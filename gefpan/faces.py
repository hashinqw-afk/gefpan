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


def available() -> bool:
    frontal, _, _ = _load()
    return frontal is not None


def _boxes_on(gray: np.ndarray, cascade: cv2.CascadeClassifier, min_s: int, neighbors: int = 4) -> list[tuple[int, int, int, int]]:
    found = cascade.detectMultiScale(gray, 1.07, neighbors, minSize=(min_s, min_s))
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
    # a little chin; keep the forehead tight so hair stays the sitting
    px, py0, py1 = int(box[2] * 0.04), int(box[3] * 0.04), int(box[3] * 0.14)
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


def _hull(box: tuple[int, int, int, int]) -> np.ndarray:
    """Jaw, cheeks, temples — enough for a pose-aware affine, not the hairline."""
    x, y, w, h = box
    return np.array(
        [
            [x + 0.18 * w, y + 0.42 * h],
            [x + 0.82 * w, y + 0.42 * h],
            [x + 0.12 * w, y + 0.68 * h],
            [x + 0.88 * w, y + 0.68 * h],
            [x + 0.28 * w, y + 0.92 * h],
            [x + 0.72 * w, y + 0.92 * h],
            [x + 0.50 * w, y + 0.98 * h],
        ],
        np.float32,
    )


def _dark_eyes(gray: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    x, y, w, h = box
    y0, y1 = y + int(h * 0.16), y + int(h * 0.52)
    x0, x1 = x + int(w * 0.10), x + int(w * 0.90)
    roi = gray[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
    if roi.size < 40:
        return []
    eq = cv2.equalizeHist(roi)
    kx = max(5, w // 16) | 1
    ky = max(3, h // 22) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
    hat = cv2.morphologyEx(eq, cv2.MORPH_BLACKHAT, kernel)
    _, th = cv2.threshold(hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_roi = float(roi.shape[0] * roi.shape[1])
    blobs: list[tuple[float, float, float]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < area_roi * 0.008 or area > area_roi * 0.22:
            continue
        moments = cv2.moments(cnt)
        if moments["m00"] < 1:
            continue
        cx = x0 + moments["m10"] / moments["m00"]
        cy = y0 + moments["m01"] / moments["m00"]
        blobs.append((cx, cy, area))
    if len(blobs) < 2:
        return []
    blobs.sort(key=lambda b: -b[2])
    top = sorted(blobs[:4], key=lambda b: b[0])
    left, right = top[0], top[-1]
    if right[0] - left[0] < 0.18 * w:
        return []
    return [(left[0], left[1]), (right[0], right[1])]


def landmarks(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Five points: left eye, right eye, nose, mouth left, mouth right."""
    pts = _canonical(box)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = box
    eyes: list[tuple[float, float]] = []
    _, _, eye = _load()
    roi = img[y : y + h, x : x + w]
    if eye is not None and roi.size:
        eq = cv2.equalizeHist(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        min_e = max(8, min(w, h) // 12)
        found = _boxes_on(eq, eye, min_e, 3)
        found = [e for e in found if e[1] + e[3] / 2.0 < h * 0.55 and e[2] < w * 0.45]
        if len(found) >= 2:
            found = sorted(found, key=lambda e: e[0])[:2]
            eyes = [
                (x + found[0][0] + found[0][2] / 2.0, y + found[0][1] + found[0][3] / 2.0),
                (x + found[1][0] + found[1][2] / 2.0, y + found[1][1] + found[1][3] / 2.0),
            ]
    if len(eyes) < 2:
        eyes = _dark_eyes(gray, box)
    if len(eyes) < 2:
        return pts
    left = np.array(eyes[0], np.float32)
    right = np.array(eyes[1], np.float32)
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


def _align_points(box: tuple[int, int, int, int], lm: np.ndarray) -> np.ndarray:
    return np.vstack([lm, _hull(box)]).astype(np.float32)


def _skin_in_box(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    roi = img[y : y + h, x : x + w]
    if roi.size == 0:
        return np.zeros(img.shape[:2], np.uint8)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    # faded prints run cooler and yellower than live skin
    skin = cv2.inRange(ycrcb, (20, 120, 70), (255, 185, 140))
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mid = cv2.inRange(gray, 40, 220)
    skin = cv2.bitwise_and(skin, mid)
    skin = cv2.medianBlur(skin, 7)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    skin = cv2.dilate(skin, np.ones((5, 5), np.uint8), iterations=1)
    full = np.zeros(img.shape[:2], np.uint8)
    full[y : y + h, x : x + w] = skin
    return full


def _face_mask(img: np.ndarray, box: tuple[int, int, int, int], pts: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x, y, bw, bh = box
    oval = np.zeros((h, w), np.uint8)
    cx = int(x + bw * 0.50)
    cy = int(y + bh * 0.46)
    ax = max(8, int(bw * 0.38))
    ay = max(10, int(bh * 0.48))
    cv2.ellipse(oval, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    hull = pts.astype(np.int32)
    if len(hull) >= 3:
        poly = np.zeros_like(oval)
        fan = cv2.convexHull(hull)
        cv2.fillConvexPoly(poly, fan, 255)
        oval = cv2.bitwise_or(oval, poly)
    skin = _skin_in_box(img, box)
    if skin.mean() > 2:
        mixed = cv2.bitwise_and(oval, cv2.dilate(skin, np.ones((11, 11), np.uint8)))
        if mixed.mean() >= oval.mean() * 0.35:
            oval = mixed
    oval[:2, :] = 0
    oval[-2:, :] = 0
    oval[:, :2] = 0
    oval[:, -2:] = 0
    return oval


def _feather(mask: np.ndarray, width: float) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if binary.max() == 0:
        return mask.astype(np.float32)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    return np.clip(dist / max(6.0, float(width)), 0.0, 1.0)


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

    dst_lm = landmarks(worn, dst_box)
    src_lm = landmarks(present, src_box)
    dst_pts = _align_points(dst_box, dst_lm)
    src_pts = _align_points(src_box, src_lm)
    h, w = worn.shape[:2]
    # full affine can take a little yaw; similarity is the safe fallback
    matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.LMEDS)
    if matrix is None:
        matrix, _ = cv2.estimateAffinePartial2D(src_lm, dst_lm, method=cv2.LMEDS)
    if matrix is None:
        return worn, ""

    warped = cv2.warpAffine(
        present,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    mask = _face_mask(worn, dst_box, dst_pts)
    if mask.max() == 0:
        return worn, ""
    if float(warped[mask > 0].mean()) < 8:
        return worn, ""

    matched = _reinhard_masked(warped, worn, mask)
    matched = _match_softness(matched, worn, mask)
    # Poisson cloning follows the worn plate's gradients and erases the
    # present face. A distance-feathered blend keeps the matched identity.
    feather = _feather(mask, max(8.0, dst_box[2] * 0.10))
    mix_m = np.clip(feather * (0.62 + 0.36 * amount), 0.0, 1.0)[:, :, None]
    out = _f32(matched) * mix_m + _f32(worn) * (1.0 - mix_m)
    grain = _f32(worn) - cv2.GaussianBlur(worn, (0, 0), 0.7).astype(np.float32)
    out = out + grain * mix_m * 0.32
    return _u8(out), "identity"
