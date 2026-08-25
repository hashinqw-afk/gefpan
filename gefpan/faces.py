"""Face identity — detect, match landmarks, rebuild the worn plate from a present photograph."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CASCADES = Path(__file__).resolve().parent / "cascades"
DETECT_SIDE = 720

_frontal = None
_profile = None
_eye = None


def _u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def _f32(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)


def _load() -> tuple[cv2.CascadeClassifier | None, cv2.CascadeClassifier | None, cv2.CascadeClassifier | None]:
    global _frontal, _profile, _eye
    if _frontal is False:
        return None, None, None
    if _frontal is not None:
        return _frontal or None, _profile or None, _eye or None

    def one(name: str) -> cv2.CascadeClassifier | None:
        path = CASCADES / name
        if not path.exists():
            return None
        cascade = cv2.CascadeClassifier(str(path))
        return None if cascade.empty() else cascade

    _frontal = one("haarcascade_frontalface_alt2.xml") or False
    _profile = one("haarcascade_profileface.xml")
    _eye = one("haarcascade_eye.xml")
    return _frontal or None, _profile, _eye


def _boxes_on(gray: np.ndarray, cascade: cv2.CascadeClassifier, min_s: int, neighbors: int = 4) -> list[tuple[int, int, int, int]]:
    found = cascade.detectMultiScale(gray, 1.08, neighbors, minSize=(min_s, min_s))
    return [tuple(int(v) for v in row) for row in found] if found is not None and len(found) else []


def _pick_face(faces: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    cx, cy = w / 2.0, h / 2.0

    def score(box: tuple[int, int, int, int]) -> float:
        x, y, bw, bh = box
        area = float(bw * bh)
        dist = ((x + bw / 2.0 - cx) / max(w, 1)) ** 2 + ((y + bh / 2.0 - cy) / max(h, 1)) ** 2
        return area * (1.0 - 0.4 * dist)

    return max(faces, key=score)


def detect_face(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest plausible face as (x, y, w, h) in the original image."""
    frontal, profile, _ = _load()
    if frontal is None:
        return None
    h, w = img.shape[:2]
    m = max(h, w)
    scale = DETECT_SIDE / float(m) if m > DETECT_SIDE else 1.0
    small = img
    if scale != 1.0:
        small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    min_s = max(24, int(min(gray.shape) * 0.07))
    faces = _boxes_on(gray, frontal, min_s, 4)
    if not faces and profile is not None:
        faces = _boxes_on(gray, profile, min_s, 3)
        flipped = cv2.flip(gray, 1)
        gw = gray.shape[1]
        for x, y, bw, bh in _boxes_on(flipped, profile, min_s, 3):
            faces.append((gw - x - bw, y, bw, bh))
    if not faces:
        return None
    x, y, bw, bh = _pick_face(faces, gray.shape)
    if bw * bh < gray.shape[0] * gray.shape[1] * 0.008:
        return None
    inv = 1.0 / scale
    box = (int(x * inv), int(y * inv), int(bw * inv), int(bh * inv))
    # a little forehead and chin — Haar sits tight on the features
    px, py0, py1 = int(box[2] * 0.06), int(box[3] * 0.10), int(box[3] * 0.16)
    x0 = max(0, box[0] - px)
    y0 = max(0, box[1] - py0)
    x1 = min(w, box[0] + box[2] + px)
    y1 = min(h, box[1] + box[3] + py1)
    return (x0, y0, max(8, x1 - x0), max(8, y1 - y0))


def _canonical(box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return np.array(
        [
            [x + 0.30 * w, y + 0.38 * h],
            [x + 0.70 * w, y + 0.38 * h],
            [x + 0.50 * w, y + 0.55 * h],
            [x + 0.36 * w, y + 0.76 * h],
            [x + 0.64 * w, y + 0.76 * h],
        ],
        np.float32,
    )


def landmarks(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Five points: left eye, right eye, nose, mouth left, mouth right."""
    pts = _canonical(box)
    _, _, eye = _load()
    if eye is None:
        return pts
    x, y, w, h = box
    roi = img[y : y + h, x : x + w]
    if roi.size == 0:
        return pts
    gray = cv2.equalizeHist(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
    min_e = max(8, min(w, h) // 12)
    found = _boxes_on(gray, eye, min_e, 3)
    found = [e for e in found if e[1] + e[3] / 2.0 < h * 0.55 and e[2] < w * 0.45]
    if len(found) < 2:
        return pts
    found = sorted(found, key=lambda e: e[0])[:2]
    left = np.array([x + found[0][0] + found[0][2] / 2.0, y + found[0][1] + found[0][3] / 2.0], np.float32)
    right = np.array([x + found[1][0] + found[1][2] / 2.0, y + found[1][1] + found[1][3] / 2.0], np.float32)
    iod = float(np.linalg.norm(right - left))
    if iod < 0.18 * w or iod > 0.62 * w:
        return pts
    mid = (left + right) / 2.0
    pts[0] = left
    pts[1] = right
    pts[2] = mid + np.array([0.0, iod * 0.55], np.float32)
    pts[3] = mid + np.array([-iod * 0.42, iod * 1.05], np.float32)
    pts[4] = mid + np.array([iod * 0.42, iod * 1.05], np.float32)
    return pts


def _oval_mask(shape: tuple[int, ...], box: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = shape[:2]
    x, y, bw, bh = box
    mask = np.zeros((h, w), np.uint8)
    cx = int(x + bw * 0.50)
    cy = int(y + bh * 0.47)
    ax = max(8, int(bw * 0.40))
    ay = max(10, int(bh * 0.50))
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    mask[:2, :] = 0
    mask[-2:, :] = 0
    mask[:, :2] = 0
    mask[:, -2:] = 0
    return mask, (int(np.clip(cx, 1, w - 2)), int(np.clip(cy, 1, h - 2)))


def _reinhard_masked(src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask > 0
    if int(m.sum()) < 80:
        return src
    sl = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    dl = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = sl.copy()
    for i in range(3):
        sm, ss = float(sl[:, :, i][m].mean()), float(sl[:, :, i][m].std()) + 1e-5
        dm, ds = float(dl[:, :, i][m].mean()), float(dl[:, :, i][m].std()) + 1e-5
        out[:, :, i] = (sl[:, :, i] - sm) * (ds / ss) + dm
    return cv2.cvtColor(_u8(out), cv2.COLOR_LAB2BGR)


def _match_softness(src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask > 0
    if int(m.sum()) < 80:
        return src
    ys = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    yd = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    ls = float(cv2.Laplacian(ys, cv2.CV_32F)[m].var())
    ld = float(cv2.Laplacian(yd, cv2.CV_32F)[m].var())
    if ls <= ld * 1.25 or ld < 1.0:
        return src
    sigma = 0.35 + min(1.6, (ls / max(ld, 1.0) - 1.0) * 0.45)
    blur = cv2.GaussianBlur(src, (0, 0), sigma)
    t = float(np.clip((ls - ld) / max(ls, 1.0), 0.12, 0.62))
    return _u8(_f32(src) * (1.0 - t) + _f32(blur) * t)


def reconstruct_identity(worn: np.ndarray, present: np.ndarray, amount: float = 0.85) -> tuple[np.ndarray, str]:
    """Put the present face onto the worn sitting. Body, clothes, and room stay put."""
    amount = float(np.clip(amount, 0.05, 1.0))
    dst_box = detect_face(worn)
    src_box = detect_face(present)
    if dst_box is None or src_box is None:
        return worn, ""

    dst_pts = landmarks(worn, dst_box)
    src_pts = landmarks(present, src_box)
    h, w = worn.shape[:2]
    matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if matrix is None:
        matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.LMEDS)
    if matrix is None:
        return worn, ""

    warped = cv2.warpAffine(
        present,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    mask, _center = _oval_mask(worn.shape, dst_box)
    if mask.max() == 0:
        return worn, ""
    if float(warped[mask > 0].mean()) < 8:
        return worn, ""

    matched = _reinhard_masked(warped, worn, mask)
    matched = _match_softness(matched, worn, mask)
    # Poisson cloning follows the worn plate's gradients and erases the
    # present face. A feathered blend keeps the matched identity.
    sigma = max(4.0, dst_box[2] * 0.048)
    feather = cv2.GaussianBlur(mask, (0, 0), sigma).astype(np.float32) / 255.0
    mix_m = np.clip(feather * (0.55 + 0.45 * amount), 0.0, 1.0)[:, :, None]
    out = _f32(matched) * mix_m + _f32(worn) * (1.0 - mix_m)
    grain = _f32(worn) - cv2.GaussianBlur(worn, (0, 0), 0.7).astype(np.float32)
    out = out + grain * mix_m * 0.28
    return _u8(out), "identity"
