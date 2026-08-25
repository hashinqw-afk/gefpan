"""Restoration chemistry — denoise, color, scratches, sharpen, upscale."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FSRCNN_URL = (
    "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb"
)
FSRCNN_PATH = MODELS / "FSRCNN_x2.pb"

MODES = (
    "auto",
    "vintage",
    "denoise",
    "color",
    "sharpen",
    "upscale",
    "portrait",
    "document",
)

MAX_SIDE = 1680
MAX_UPSCALE_IN = 1280

_sr = None
_sr_failed = False
_face = None


def _u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def _f32(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)


def mix(orig: np.ndarray, proc: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    if t <= 0:
        return orig
    if t >= 1:
        return proc
    if orig.shape != proc.shape:
        orig = cv2.resize(orig, (proc.shape[1], proc.shape[0]), interpolation=cv2.INTER_CUBIC)
    return _u8(_f32(proc) * t + _f32(orig) * (1.0 - t))


def read_image(data: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        im = bg.convert("RGB")
    elif im.mode != "RGB":
        im = im.convert("RGB")
    rgb = np.asarray(im)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def encode_jpeg(bgr: np.ndarray, quality: int = 96) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def encode_png(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def limit_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def ensure_fsrcnn() -> Path | None:
    if FSRCNN_PATH.exists() and FSRCNN_PATH.stat().st_size > 1000:
        return FSRCNN_PATH
    MODELS.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(FSRCNN_URL, timeout=45) as resp:
            data = resp.read()
        if len(data) > 1000:
            FSRCNN_PATH.write_bytes(data)
            return FSRCNN_PATH
    except Exception:
        pass
    # Some sandboxes block raw.githubusercontent.com; git still works.
    try:
        import subprocess
        dest = MODELS / "_fsrcnn_src"
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/Saafke/FSRCNN_Tensorflow.git", str(dest)],
            check=True,
            capture_output=True,
            timeout=60,
        )
        src = dest / "models" / "FSRCNN_x2.pb"
        if src.exists():
            FSRCNN_PATH.write_bytes(src.read_bytes())
        import shutil
        shutil.rmtree(dest, ignore_errors=True)
        if FSRCNN_PATH.exists() and FSRCNN_PATH.stat().st_size > 1000:
            return FSRCNN_PATH
    except Exception:
        pass
    return None


def _superres():
    global _sr, _sr_failed
    if _sr is not None or _sr_failed:
        return _sr
    path = ensure_fsrcnn()
    if path is None or not hasattr(cv2, "dnn_superres"):
        _sr_failed = True
        return None
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(path))
        sr.setModel("fsrcnn", 2)
        _sr = sr
        return _sr
    except Exception:
        _sr_failed = True
        return None


def upscale_2x(img: np.ndarray) -> tuple[np.ndarray, str]:
    src = limit_side(img, MAX_UPSCALE_IN)
    sr = _superres()
    if sr is not None:
        try:
            return sr.upsample(src), "fsrcnn"
        except Exception:
            pass
    up = cv2.resize(src, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return unsharp(up, 0.55, 1.0), "bicubic"


def gray_world(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    f = _f32(img)
    means = np.maximum(f.reshape(-1, 3).mean(axis=0), 1.0)
    scale = means.mean() / means
    scale = 1.0 + (scale - 1.0) * amount
    return _u8(f * scale)


def neutralize_lab(img: np.ndarray, amount: float = 0.65) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    a += (128.0 - float(a.mean())) * amount
    b += (128.0 - float(b.mean())) * amount
    lab = cv2.merge([l, np.clip(a, 0, 255), np.clip(b, 0, 255)]).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def auto_levels(img: np.ndarray, low: float = 0.8, high: float = 99.2, linked: bool = False) -> np.ndarray:
    f = _f32(img)
    if linked:
        y = 0.114 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.299 * f[:, :, 2]
        lo, hi = np.percentile(y, (low, high))
        if hi - lo < 12:
            return img
        g = 255.0 / (hi - lo)
        return _u8((f - lo) * g)
    out = np.empty_like(f)
    for c in range(3):
        ch = f[:, :, c]
        lo, hi = np.percentile(ch, (low, high))
        if hi - lo < 8:
            out[:, :, c] = ch
        else:
            out[:, :, c] = (ch - lo) * (255.0 / (hi - lo))
    return _u8(out)


def clahe_luma(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def s_curve(img: np.ndarray, amount: float = 0.28) -> np.ndarray:
    x = _f32(img) / 255.0
    y = x + amount * x * (1.0 - x) * (x - 0.5) * 6.0
    return _u8(y * 255.0)


def vibrance(img: np.ndarray, amount: float = 0.28) -> np.ndarray:
    f = _f32(img) / 255.0
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s + amount * (1.0 - s) * s * 2.4, 0.0, 1.0)
    out = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    return _u8(out * 255.0)


def chroma_smooth(img: np.ndarray, sigma: float = 1.35) -> np.ndarray:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    cr = cv2.GaussianBlur(cr, (0, 0), sigma)
    cb = cv2.GaussianBlur(cb, (0, 0), sigma)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)


def luma_denoise(img: np.ndarray, strength: float = 0.55) -> np.ndarray:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    d = 5 if strength < 0.55 else 7
    sigma = 12 + 38 * float(strength)
    y = cv2.bilateralFilter(y, d, sigma, max(5.0, d * 0.9))
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)


def nlm_denoise(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    h = 3.0 + 9.0 * float(strength)
    small = max(img.shape[0], img.shape[1]) > 1400
    src = cv2.resize(img, None, fx=0.65, fy=0.65, interpolation=cv2.INTER_AREA) if small else img
    clean = cv2.fastNlMeansDenoisingColored(src, None, h, h, 7, 21)
    if small:
        clean = cv2.resize(clean, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        # keep original high-frequency edges
        detail = _f32(img) - _f32(cv2.GaussianBlur(img, (0, 0), 0.8))
        clean = _u8(_f32(clean) + detail * 0.35)
    return clean


def unsharp(img: np.ndarray, amount: float = 0.7, radius: float = 1.15) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), max(0.4, radius))
    detail = _f32(img) - _f32(blur)
    return _u8(_f32(img) + detail * amount)


def _luma(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return cv2.split(ycrcb)


def _from_luma(y: np.ndarray, cr: np.ndarray, cb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.merge([_u8(y), cr, cb]), cv2.COLOR_YCrCb2BGR)


def richardson_lucy(img: np.ndarray, iterations: int = 6, sigma: float = 0.85) -> np.ndarray:
    """Deblur on luma only — brings back edges lost to soft focus and scan blur."""
    y, cr, cb = _luma(img)
    observed = np.clip(y.astype(np.float32) / 255.0, 1e-6, 1.0)
    k = int(max(3, round(sigma * 6)) | 1)
    g = cv2.getGaussianKernel(k, sigma)
    psf = g @ g.T
    psf /= float(psf.sum())
    mirror = np.flip(psf)
    estimate = observed.copy()
    for _ in range(max(1, int(iterations))):
        conv = np.clip(cv2.filter2D(estimate, -1, psf), 1e-6, None)
        estimate *= cv2.filter2D(observed / conv, -1, mirror)
        estimate = np.clip(estimate, 0.0, 1.0)
    return _from_luma(estimate * 255.0, cr, cb)


def multi_sharpen(img: np.ndarray, amount: float = 0.7) -> np.ndarray:
    """Fine + medium + coarse unsharp on luma — perceived sharpness without color fringing."""
    y, cr, cb = _luma(img)
    yf = y.astype(np.float32)
    for radius, weight in ((0.55, 0.62), (1.25, 0.38), (2.4, 0.18)):
        blur = cv2.GaussianBlur(y, (0, 0), radius).astype(np.float32)
        yf = yf + amount * weight * (yf - blur)
    return _from_luma(yf, cr, cb)


def dog_detail(img: np.ndarray, amount: float = 0.45) -> np.ndarray:
    """Difference-of-Gaussians micro-contrast — the 'unclear' midtones come forward."""
    y, cr, cb = _luma(img)
    yf = y.astype(np.float32)
    fine = cv2.GaussianBlur(y, (0, 0), 0.7).astype(np.float32)
    coarse = cv2.GaussianBlur(y, (0, 0), 2.1).astype(np.float32)
    return _from_luma(yf + amount * (fine - coarse), cr, cb)


def high_boost(img: np.ndarray, amount: float = 0.35) -> np.ndarray:
    y, cr, cb = _luma(img)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    boosted = cv2.filter2D(y.astype(np.float32), -1, kernel)
    mixed = y.astype(np.float32) * (1.0 - amount) + boosted * amount
    return _from_luma(mixed, cr, cb)


def clarify(img: np.ndarray, strength: float = 0.75) -> np.ndarray:
    """Deblur + one luma sharpen. Enough to look in focus, not painted."""
    s = float(np.clip(strength, 0.05, 1.0))
    out = richardson_lucy(img, iterations=3 + int(round(2 * s)), sigma=0.8)
    out = multi_sharpen(out, 0.32 + 0.38 * s)
    return out


def wiener_sharpen(img: np.ndarray, amount: float = 0.55) -> np.ndarray:
    """High-frequency lift with a soft Wiener-like gain."""
    f = _f32(img)
    blur = cv2.GaussianBlur(img, (0, 0), 1.4).astype(np.float32)
    residual = f - blur
    var = cv2.GaussianBlur(residual * residual, (0, 0), 2.0)
    gain = residual * (var / (var + 180.0))
    return _u8(f + gain * (1.6 * amount))


def detect_damage(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = 11 if min(img.shape[:2]) > 700 else 9
    residual = cv2.absdiff(gray, cv2.medianBlur(gray, k))

    spot_thr = max(16.0, float(np.percentile(residual, 99.15)))
    spots = (residual >= spot_thr).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(spots, 8)
    keep = np.zeros_like(spots)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if 2 <= area <= 90:
            keep[labels == i] = 255

    line_thr = max(12.0, float(np.percentile(residual, 97.2)))
    edges = (residual >= line_thr).astype(np.uint8) * 255
    min_len = max(64, int(min(img.shape[:2]) * 0.18))
    found = cv2.HoughLinesP(edges, 1, np.pi / 180.0, 36, minLineLength=min_len, maxLineGap=12)
    if found is not None:
        hh, ww = residual.shape
        for row in found:
            x1, y1, x2, y2 = [int(v) for v in np.ravel(row)[:4]]
            n = max(8, int(np.hypot(x2 - x1, y2 - y1) // 6))
            xs = np.clip(np.linspace(x1, x2, n).astype(int), 0, ww - 1)
            ys = np.clip(np.linspace(y1, y2, n).astype(int), 0, hh - 1)
            if float(residual[ys, xs].mean()) >= line_thr * 0.85:
                cv2.line(keep, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)

    if keep.mean() < 0.15:
        return keep
    return cv2.dilate(keep, np.ones((3, 3), np.uint8))


def repair_damage(img: np.ndarray, strength: float = 0.8) -> np.ndarray:
    mask = detect_damage(img)
    if mask.mean() < 0.15:
        return img
    radius = 3 if min(img.shape[:2]) > 800 else 2
    fixed = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    return mix(img, fixed, 0.82 + 0.18 * strength)


def _load_face_cascade():
    global _face
    if _face is not None:
        return _face or None
    candidates = [
        Path(__file__).resolve().parent / "cascades" / "haarcascade_frontalface_alt2.xml",
        MODELS / "haarcascade_frontalface_default.xml",
    ]
    data = getattr(cv2, "data", None)
    if data is not None:
        candidates.append(Path(data.haarcascades) / "haarcascade_frontalface_default.xml")
    for path in candidates:
        if path.exists():
            cascade = cv2.CascadeClassifier(str(path))
            if not cascade.empty():
                _face = cascade
                return _face
    _face = False
    return None


def _skin_mask(img: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    mask = cv2.medianBlur(mask, 11)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    if mask.mean() < 4:
        return np.zeros(img.shape[:2], np.float32)
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 12), 0, 1)


def face_mask(img: np.ndarray) -> np.ndarray:
    cascade = _load_face_cascade()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.float32)
    if cascade is not None:
        faces = cascade.detectMultiScale(gray, 1.08, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            pad_x, pad_y = int(w * 0.28), int(h * 0.38)
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            x1, y1 = min(img.shape[1], x + w + pad_x), min(img.shape[0], y + h + pad_y)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            cv2.ellipse(mask, (cx, cy), ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360, 1.0, -1)
    if mask.max() == 0:
        return _skin_mask(img)
    sigma = max(10.0, min(img.shape[:2]) * 0.025)
    return np.clip(cv2.GaussianBlur(mask, (0, 0), sigma), 0, 1)


def blend_masked(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)
    if m.ndim == 2:
        m = m[:, :, None]
    return _u8(_f32(overlay) * m + _f32(base) * (1.0 - m))


def lift_shadows(img: np.ndarray, amount: float = 0.22) -> np.ndarray:
    f = _f32(img) / 255.0
    lum = 0.114 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.299 * f[:, :, 2]
    lift = amount * np.power(np.clip(1.0 - lum, 0, 1), 1.35)
    return _u8((f + lift[:, :, None] * (1.0 - f)) * 255.0)


def crush_haze(img: np.ndarray, amount: float = 0.18) -> np.ndarray:
    """A light dark-channel style dehaze — good for faded landscapes."""
    f = _f32(img)
    dark = np.min(f, axis=2)
    air = float(np.percentile(dark, 99.2))
    if air < 140:
        return img
    t = 1.0 - amount * (dark / max(air, 1.0))
    t = np.clip(t, 0.45, 1.0)
    out = (f - air) / t[:, :, None] + air
    return _u8(out)


def illumination_flatten(img: np.ndarray, amount: float = 0.7) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = max(21, (min(img.shape[:2]) // 18) | 1)
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), k)
    bg = np.maximum(bg.astype(np.float32), 8.0)
    norm = _f32(img) / (bg[:, :, None] / 255.0)
    flat = _u8(norm * 0.92)
    return mix(img, flat, amount)


def _auto(img: np.ndarray, s: float) -> np.ndarray:
    out = repair_damage(img, 0.55 + 0.4 * s)
    out = gray_world(out, 0.35 + 0.4 * s)
    out = neutralize_lab(out, 0.35 + 0.35 * s)
    out = chroma_smooth(out, 0.8 + 0.8 * s)
    out = luma_denoise(out, 0.35 + 0.4 * s)
    out = auto_levels(out, 1.0, 99.1, linked=True)
    out = clahe_luma(out, 1.2 + 1.4 * s, 8)
    out = lift_shadows(out, 0.08 + 0.16 * s)
    out = s_curve(out, 0.12 + 0.22 * s)
    out = vibrance(out, 0.12 + 0.22 * s)
    out = clarify(out, 0.55 + 0.25 * s)
    return out


def _vintage(img: np.ndarray, s: float) -> np.ndarray:
    out = repair_damage(img, 0.7 + 0.3 * s)
    out = neutralize_lab(out, 0.55 + 0.4 * s)
    out = gray_world(out, 0.45 + 0.4 * s)
    out = chroma_smooth(out, 1.0 + s)
    out = luma_denoise(out, 0.4 + 0.35 * s)
    out = auto_levels(out, 0.8, 99.2, linked=True)
    out = crush_haze(out, 0.06 + 0.1 * s)
    out = clahe_luma(out, 1.2 + 1.0 * s, 8)
    out = s_curve(out, 0.14 + 0.18 * s)
    out = vibrance(out, 0.16 + 0.22 * s)
    out = lift_shadows(out, 0.1 + 0.14 * s)
    out = clarify(out, 0.5 + 0.25 * s)
    return out


def _denoise(img: np.ndarray, s: float) -> np.ndarray:
    out = chroma_smooth(img, 1.0 + 1.4 * s)
    if s > 0.35:
        out = nlm_denoise(out, s)
    else:
        out = luma_denoise(out, 0.45 + 0.5 * s)
    out = multi_sharpen(out, 0.28 + 0.25 * s)
    return out


def _color(img: np.ndarray, s: float) -> np.ndarray:
    out = repair_damage(img, 0.55 + 0.3 * s)
    out = gray_world(out, 0.45 + 0.4 * s)
    out = neutralize_lab(out, 0.4 + 0.35 * s)
    out = auto_levels(out, 0.9, 99.2, linked=True)
    out = clahe_luma(out, 1.1 + 1.1 * s, 8)
    out = vibrance(out, 0.14 + 0.22 * s)
    out = s_curve(out, 0.1 + 0.16 * s)
    out = multi_sharpen(out, 0.3 + 0.25 * s)
    return out


def _sharpen(img: np.ndarray, s: float) -> np.ndarray:
    out = luma_denoise(img, 0.2 + 0.15 * s)
    out = wiener_sharpen(out, 0.4 + 0.4 * s)
    out = clarify(out, 0.6 + 0.25 * s)
    return out


def _portrait(img: np.ndarray, s: float) -> np.ndarray:
    faces = face_mask(img)
    base = gray_world(img, 0.3 + 0.25 * s)
    base = neutralize_lab(base, 0.25 + 0.25 * s)
    skin = chroma_smooth(base, 1.1 + 0.9 * s)
    skin = luma_denoise(skin, 0.25 + 0.25 * s)
    rest = luma_denoise(base, 0.4 + 0.4 * s)
    rest = chroma_smooth(rest, 1.2 + s)
    out = blend_masked(rest, skin, faces) if faces.max() > 0 else skin
    out = auto_levels(out, 1.0, 99.0, linked=True)
    out = clahe_luma(out, 1.0 + 1.1 * s, 8)
    out = vibrance(out, 0.08 + 0.16 * s)
    sharp = clarify(out, 0.42 + 0.22 * s)
    # keep faces slightly softer than fabric / window detail
    if faces.max() > 0:
        out = blend_masked(sharp, mix(out, sharp, 0.55), faces)
    else:
        out = sharp
    return out


def _document(img: np.ndarray, s: float) -> np.ndarray:
    out = repair_damage(img, 0.75 + 0.2 * s)
    out = illumination_flatten(out, 0.18 + 0.2 * s)
    out = neutralize_lab(out, 0.4 + 0.3 * s)
    out = chroma_smooth(out, 0.8)
    out = luma_denoise(out, 0.25 + 0.2 * s)
    out = auto_levels(out, 0.7, 99.3, linked=True)
    out = clahe_luma(out, 1.1 + 1.0 * s, 8)
    out = s_curve(out, 0.06 + 0.1 * s)
    out = multi_sharpen(out, 0.4 + 0.3 * s)
    return out


def _upscale(img: np.ndarray, s: float) -> tuple[np.ndarray, str]:
    prepared = luma_denoise(img, 0.25)
    prepared = chroma_smooth(prepared, 0.9)
    up, engine = upscale_2x(prepared)
    up = multi_sharpen(up, 0.22 + 0.2 * s)
    return up, engine


_PIPELINES = {
    "auto": _auto,
    "vintage": _vintage,
    "denoise": _denoise,
    "color": _color,
    "sharpen": _sharpen,
    "portrait": _portrait,
    "document": _document,
}


def align_trace(worn: np.ndarray, trace: np.ndarray) -> tuple[np.ndarray, str]:
    """Resize the guide and, if it is the same sitting, lock it to the worn plate."""
    h, w = worn.shape[:2]
    ref = cv2.resize(trace, (w, h), interpolation=cv2.INTER_AREA)
    return ref, "trace"


def reinhard_color(src: np.ndarray, ref: np.ndarray, amount: float = 0.75) -> np.ndarray:
    """Borrow the guide's dye — not its pixels — so the worn plate keeps its drawing."""
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = s.copy()
    for i in range(3):
        sm, ss = float(s[:, :, i].mean()), float(s[:, :, i].std()) + 1e-5
        rm, rs = float(r[:, :, i].mean()), float(r[:, :, i].std()) + 1e-5
        mapped = (s[:, :, i] - sm) * (rs / ss) + rm
        out[:, :, i] = s[:, :, i] * (1.0 - amount) + mapped * amount
    return cv2.cvtColor(_u8(out), cv2.COLOR_LAB2BGR)


def fill_from_trace(worn: np.ndarray, aligned: np.ndarray, amount: float = 0.85) -> np.ndarray:
    """Where the emulsion is torn, take silver from the aligned guide."""
    mask = detect_damage(worn)
    if mask.mean() < 0.15:
        return worn
    m = mask.astype(np.float32) / 255.0
    m = cv2.GaussianBlur(m, (0, 0), 1.2)
    m = np.clip(m * amount, 0.0, 1.0)[:, :, None]
    return _u8(_f32(worn) * (1.0 - m) + _f32(aligned) * m)


def transfer_micro_detail(base: np.ndarray, ref: np.ndarray, amount: float = 0.28) -> np.ndarray:
    """A little of the guide's grain and pore — never a paste of the whole picture."""
    sigma = 1.8
    base_f = _f32(base)
    ref_f = _f32(ref)
    base_low = cv2.GaussianBlur(base, (0, 0), sigma).astype(np.float32)
    ref_low = cv2.GaussianBlur(ref, (0, 0), sigma).astype(np.float32)
    # keep the worn plate's drawing; mix only the high-frequency layer
    return _u8(base_low + (base_f - base_low) * (1.0 - amount) + (ref_f - ref_low) * amount)


def apply_trace(worn: np.ndarray, trace: np.ndarray, strength: float) -> tuple[np.ndarray, str]:
    """Rebuild the worn plate. A present-day face is matched in; the sitting stays put."""
    s = float(np.clip(strength, 0.05, 1.0))
    from .faces import reconstruct_identity

    ident, face_engine = reconstruct_identity(worn, trace, amount=0.78 + 0.18 * s)
    if face_engine:
        cleaned = repair_damage(ident, 0.45 + 0.2 * s)
        cleaned = chroma_smooth(cleaned, 0.55)
        cleaned = neutralize_lab(cleaned, 0.18 + 0.16 * s)
        cleaned = auto_levels(cleaned, 1.0, 99.2, linked=True)
        out = mix(ident, cleaned, 0.40 + 0.18 * s)
        out = unsharp(out, 0.12 + 0.10 * s, 0.9)
        return out, face_engine

    base = repair_damage(worn, 0.55 + 0.3 * s)
    base = chroma_smooth(base, 0.7)
    base = luma_denoise(base, 0.22)
    aligned, engine = align_trace(worn, trace)
    filled = fill_from_trace(base, aligned, 0.7 + 0.25 * s)
    colored = reinhard_color(filled, aligned, 0.5 + 0.35 * s)
    detailed = transfer_micro_detail(colored, aligned, 0.16 + 0.22 * s)
    out = mix(worn, detailed, 0.58 + 0.28 * s)
    out = fill_from_trace(out, aligned, 0.88)
    out = unsharp(out, 0.18 + 0.16 * s, 0.9)
    return out, engine


def restore_image(
    data: bytes,
    mode: str = "auto",
    strength: float = 0.72,
    upscale: bool = False,
    trace: bytes | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    mode = (mode or "auto").lower().strip()
    if mode not in MODES:
        mode = "auto"
    strength = float(np.clip(strength, 0.05, 1.0))

    src = read_image(data)
    work = limit_side(src, MAX_SIDE)
    engine = "opencv"
    used_trace = False

    if trace:
        try:
            guide = limit_side(read_image(trace), MAX_SIDE)
            out, engine = apply_trace(work, guide, strength)
            used_trace = True
        except Exception:
            used_trace = False

    if not used_trace:
        if mode == "upscale":
            out, engine = _upscale(work, strength)
        else:
            proc = _PIPELINES.get(mode, _auto)(work, strength)
            # keep the photograph — do not replace it with a crunchy reconstruction
            out = mix(work, proc, 0.55 + 0.28 * strength)
            if mode in ("auto", "vintage", "portrait", "document", "color"):
                out = repair_damage(out, min(1.0, strength + 0.15))
            if upscale:
                out, engine = _upscale(out, strength)

    ms = int((time.perf_counter() - t0) * 1000)
    h, w = out.shape[:2]
    return {
        "image": out,
        "jpeg": encode_jpeg(out),
        "width": w,
        "height": h,
        "mode": mode,
        "strength": strength,
        "engine": engine,
        "ms": ms,
        "trace": used_trace,
    }


def restore_file(
    src: Path,
    dest: Path,
    mode: str = "auto",
    strength: float = 0.72,
    upscale: bool = False,
    trace: Path | None = None,
) -> dict[str, Any]:
    guide = Path(trace).read_bytes() if trace else None
    result = restore_image(
        Path(src).read_bytes(),
        mode=mode,
        strength=strength,
        upscale=upscale,
        trace=guide,
    )
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(result["jpeg"])
    return {k: v for k, v in result.items() if k not in ("image", "jpeg")}
