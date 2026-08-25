"""Local studio server."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .degrade import build_demo_prints
from .restore import MODES, restore_image

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
SAMPLES = WEB / "samples"
MANIFEST = SAMPLES / "manifest.json"

MAX_UPLOAD = 14 * 1024 * 1024
_LAST: dict = {}

app = FastAPI(title="Gefpan", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Gefpan-Mode",
        "X-Gefpan-Width",
        "X-Gefpan-Height",
        "X-Gefpan-Ms",
        "X-Gefpan-Engine",
        "X-Gefpan-Filename",
        "X-Gefpan-Trace",
        "Content-Disposition",
    ],
)


@app.get("/api/health")
def health() -> dict:
    from .faces import available as faces_ok

    return {
        "ok": True,
        "name": "gefpan",
        "modes": list(MODES),
        "faces": faces_ok(),
    }


@app.get("/api/samples")
def samples() -> list:
    if MANIFEST.exists():
        items = json.loads(MANIFEST.read_text(encoding="utf-8"))
        traces = ROOT / "samples" / "source"
        for item in items:
            present = traces / f"{item['id']}-present.jpg"
            name = present.name if present.exists() else f"{item['id']}.jpg"
            item["trace"] = f"/traces/{name}"
            item["present"] = present.exists()
        return items
    return []


@app.post("/api/restore")
async def restore(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    strength: float = Form(0.72),
    upscale: str = Form("false"),
    trace: UploadFile | None = File(None),
) -> Response:
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "file too large (14 MB max)")
    want_upscale = str(upscale).strip().lower() in {"1", "true", "yes", "on"}
    guide = None
    if trace is not None:
        guide = await trace.read()
        if guide and len(guide) > MAX_UPLOAD:
            raise HTTPException(413, "trace file too large (14 MB max)")
        if not guide:
            guide = None
    try:
        import asyncio

        result = await asyncio.to_thread(
            restore_image,
            data,
            mode,
            float(strength),
            want_upscale,
            guide,
        )
    except Exception as exc:
        raise HTTPException(400, f"could not restore: {exc}") from exc

    raw_name = Path(file.filename or "photograph.jpg").stem or "photograph"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in raw_name).strip("-") or "photograph"
    filename = f"{safe}-restored.jpg"
    _LAST["jpeg"] = result["jpeg"]
    _LAST["filename"] = filename

    return Response(
        content=result["jpeg"],
        media_type="image/jpeg",
        headers={
            "X-Gefpan-Mode": result["mode"],
            "X-Gefpan-Width": str(result["width"]),
            "X-Gefpan-Height": str(result["height"]),
            "X-Gefpan-Ms": str(result["ms"]),
            "X-Gefpan-Engine": result["engine"],
            "X-Gefpan-Filename": filename,
            "X-Gefpan-Trace": "1" if result.get("trace") else "0",
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/download")
def download_last() -> Response:
    if "jpeg" not in _LAST:
        raise HTTPException(404, "restore a photograph first")
    name = _LAST.get("filename", "gefpan-restored.jpg")
    return Response(
        content=_LAST["jpeg"],
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
            "Cache-Control": "no-store",
        },
    )


def _ready_samples() -> None:
    if not MANIFEST.exists():
        try:
            build_demo_prints()
        except Exception:
            SAMPLES.mkdir(parents=True, exist_ok=True)


_ready_samples()

if WEB.exists():
    SAMPLES.mkdir(parents=True, exist_ok=True)
    traces = ROOT / "samples" / "source"
    if traces.exists():
        app.mount("/traces", StaticFiles(directory=str(traces)), name="traces")
    app.mount("/samples", StaticFiles(directory=str(SAMPLES)), name="samples")
    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
