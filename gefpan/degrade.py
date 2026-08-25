"""Age a clean photograph so the studio has worn prints to try."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .restore import encode_jpeg, limit_side

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "samples" / "source"
WORN = ROOT / "web" / "samples"
MANIFEST = WORN / "manifest.json"

RECIPES = {
    "portrait": {
        "title": "Window",
        "year": "c. 1974",
        "mode": "portrait",
        "note": "Yellowed print, hairline scratches, a little fog on the silver.",
        "fade": 0.55,
        "yellow": 0.42,
        "contrast": 0.38,
        "blur": 0.85,
        "grain": 16,
        "scratches": 14,
        "dust": 70,
        "stain": 0.22,
        "vignette": 0.35,
        "jpeg": 28,
    },
    "lake": {
        "title": "Alpine",
        "year": "c. 1968",
        "mode": "vintage",
        "note": "Bleached color, lake haze, dust in the emulsion.",
        "fade": 0.62,
        "yellow": 0.22,
        "contrast": 0.32,
        "blur": 0.7,
        "grain": 14,
        "scratches": 6,
        "dust": 120,
        "stain": 0.12,
        "vignette": 0.28,
        "haze": 0.28,
        "jpeg": 32,
    },
    "street": {
        "title": "Alley",
        "year": "c. 1981",
        "mode": "denoise",
        "note": "Pushed film — heavy grain, soft focus, a thin underexposure.",
        "fade": 0.22,
        "yellow": 0.08,
        "contrast": 0.22,
        "blur": 1.05,
        "grain": 28,
        "scratches": 3,
        "dust": 40,
        "stain": 0.0,
        "vignette": 0.42,
        "darken": 0.18,
        "jpeg": 22,
    },
    "boy": {
        "title": "Keepsake",
        "year": "c. 1976",
        "mode": "vintage",
        "note": "A handled print: crease, tea stain, the reds gone brick.",
        "fade": 0.58,
        "yellow": 0.48,
        "sepia": 0.35,
        "contrast": 0.3,
        "blur": 0.75,
        "grain": 15,
        "scratches": 10,
        "dust": 55,
        "stain": 0.38,
        "crease": True,
        "vignette": 0.32,
        "jpeg": 26,
    },
    "library": {
        "title": "Stacks",
        "year": "c. 1959",
        "mode": "color",
        "note": "Underexposed plate, low contrast, a brown fog in the shadows.",
        "fade": 0.4,
        "yellow": 0.3,
        "contrast": 0.45,
        "blur": 0.65,
        "grain": 18,
        "scratches": 8,
        "dust": 90,
        "stain": 0.16,
        "vignette": 0.5,
        "darken": 0.22,
        "jpeg": 30,
    },
    "letter": {
        "title": "Letter",
        "year": "18 Oct 1931",
        "mode": "document",
        "note": "Uneven desk light, foxing on the paper, the ink a little soft.",
        "fade": 0.2,
        "yellow": 0.35,
        "contrast": 0.28,
        "blur": 0.95,
        "grain": 10,
        "scratches": 4,
        "dust": 35,
        "stain": 0.3,
        "shadow": True,
        "vignette": 0.15,
        "jpeg": 34,
    },
}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _vignette(img: np.ndarray, amount: float) -> np.ndarray:
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt(((y - cy) / cy) ** 2 + ((x - cx) / cx) ** 2)
    fall = 1.0 - amount * np.clip((r - 0.35) / 1.15, 0, 1) ** 1.4
    return np.clip(img.astype(np.float32) * fall[:, :, None], 0, 255).astype(np.uint8)


def _scratches(img: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(n):
        light = bool(rng.random() > 0.45)
        color = (int(rng.integers(190, 255)),) * 3 if light else (int(rng.integers(8, 50)),) * 3
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
        angle = float(rng.uniform(0, np.pi))
        length = int(rng.integers(int(min(h, w) * 0.15), int(min(h, w) * 0.85)))
        x2 = int(np.clip(x1 + np.cos(angle) * length, 0, w - 1))
        y2 = int(np.clip(y1 + np.sin(angle) * length, 0, h - 1))
        thickness = int(rng.choice([1, 1, 1, 2]))
        overlay = out.copy()
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    return out


def _dust(img: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(n):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(1, 4))
        color = int(rng.integers(0, 40)) if rng.random() > 0.35 else int(rng.integers(200, 255))
        cv2.circle(out, (x, y), r, (color, color, color), -1, cv2.LINE_AA)
    return out


def _stain(img: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    if amount <= 0:
        return img
    h, w = img.shape[:2]
    stain = np.zeros((h, w), np.float32)
    blobs = 2 + int(amount * 4)
    for _ in range(blobs):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        sx = int(rng.integers(w // 8, w // 3))
        sy = int(rng.integers(h // 8, h // 3))
        y, x = np.ogrid[:h, :w]
        blob = np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
        stain += blob * float(rng.uniform(0.4, 1.0))
    stain = np.clip(stain, 0, 1) * amount
    tint = np.zeros_like(img, np.float32)
    tint[:, :, 0] = 40
    tint[:, :, 1] = 95
    tint[:, :, 2] = 160
    out = img.astype(np.float32) * (1.0 - 0.45 * stain[:, :, None]) + tint * (0.45 * stain[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)


def _crease(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()
    x1, y1 = int(rng.integers(0, w // 4)), 0
    x2, y2 = int(rng.integers(w // 2, w)), h - 1
    cv2.line(overlay, (x1, y1), (x2, y2), (28, 28, 28), 2, cv2.LINE_AA)
    cv2.line(overlay, (x1 + 2, y1), (x2 + 2, y2), (210, 205, 195), 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.55, img, 0.45, 0)


def _shadow_band(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    ramp = np.linspace(0.62, 1.0, w, dtype=np.float32)
    band = np.tile(ramp, (h, 1))
    return np.clip(img.astype(np.float32) * band[:, :, None], 0, 255).astype(np.uint8)


def degrade(img: np.ndarray, recipe: dict, seed: int = 7) -> np.ndarray:
    rng = _rng(seed)
    out = limit_side(img, 1400).copy()
    f = out.astype(np.float32)

    fade = float(recipe.get("fade", 0))
    if fade:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
        f = f * (1.0 - fade) + gray[:, :, None] * fade * 0.85 + 18 * fade

    yellow = float(recipe.get("yellow", 0))
    if yellow:
        f[:, :, 0] *= 1.0 - 0.28 * yellow
        f[:, :, 1] *= 1.0 + 0.04 * yellow
        f[:, :, 2] *= 1.0 + 0.16 * yellow
        f += np.array([6, 18, 32], np.float32) * yellow

    sepia = float(recipe.get("sepia", 0))
    if sepia:
        b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        sb = 0.272 * r + 0.534 * g + 0.131 * b
        sg = 0.349 * r + 0.686 * g + 0.168 * b
        sr = 0.393 * r + 0.769 * g + 0.189 * b
        sep = np.dstack([sb, sg, sr])
        f = f * (1.0 - sepia) + sep * sepia

    contrast = float(recipe.get("contrast", 0))
    if contrast:
        f = (f - 127.5) * (1.0 - 0.7 * contrast) + 127.5 + 12 * contrast

    darken = float(recipe.get("darken", 0))
    if darken:
        f *= 1.0 - darken

    haze = float(recipe.get("haze", 0))
    if haze:
        f = f * (1.0 - haze) + 210 * haze

    out = np.clip(f, 0, 255).astype(np.uint8)

    blur = float(recipe.get("blur", 0))
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)

    grain = float(recipe.get("grain", 0))
    if grain:
        noise = rng.normal(0, grain, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    out = _scratches(out, int(recipe.get("scratches", 0)), rng)
    out = _dust(out, int(recipe.get("dust", 0)), rng)
    out = _stain(out, float(recipe.get("stain", 0)), rng)
    if recipe.get("crease"):
        out = _crease(out, rng)
    if recipe.get("shadow"):
        out = _shadow_band(out)
    out = _vignette(out, float(recipe.get("vignette", 0)))

    q = int(recipe.get("jpeg", 40))
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def build_demo_prints() -> list[dict]:
    WORN.mkdir(parents=True, exist_ok=True)
    items = []
    order = ["portrait", "lake", "street", "boy", "library", "letter"]
    for i, key in enumerate(order):
        recipe = RECIPES[key]
        src = SOURCE / f"{key}.jpg"
        if not src.exists():
            continue
        bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        worn = degrade(bgr, recipe, seed=11 + i * 17)
        dest = WORN / f"{key}.jpg"
        dest.write_bytes(encode_jpeg(worn, 86))
        items.append(
            {
                "id": key,
                "title": recipe["title"],
                "year": recipe["year"],
                "mode": recipe["mode"],
                "note": recipe["note"],
                "file": f"/samples/{key}.jpg",
            }
        )
        print(f"worn  {dest}  {worn.shape[1]}×{worn.shape[0]}")
    MANIFEST.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return items
