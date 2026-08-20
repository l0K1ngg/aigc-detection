import os
import time
import uuid
import threading
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from detector import AIGCDetector

app = FastAPI(title="AIGC Trace Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = AIGCDetector(serve_dir="serve")
inference_lock = threading.Semaphore(1)
MAX_BYTES = 10 * 1024 * 1024


@app.get("/readyz")
def readyz():
    if not detector.is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="MODEL_NOT_READY")
    return {
        "status": "ready",
        "thresholds": detector.thresholds,
        "model_card": detector.model_card,
    }


@app.get("/v1/model-card")
def get_model_card():
    return detector.model_card


@app.post("/v1/detect")
async def detect(
    request: Request,
    file: Optional[UploadFile] = File(None)
):
    if not detector.is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"code": "MODEL_NOT_READY", "message": "Model or thresholds not loaded"}
        )

    t_start = time.perf_counter()
    request_id = "01" + uuid.uuid4().hex[:10].upper()

    content_type = request.headers.get("content-type", "")
    if file is not None:
        body = await file.read()
    elif "application/octet-stream" in content_type:
        body = await request.body()
    else:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"code": "UNSUPPORTED_FORMAT", "message": "Provide multipart form or octet-stream"}
        )

    if len(body) > MAX_BYTES:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"code": "PAYLOAD_TOO_LARGE", "message": f"Payload exceeds {MAX_BYTES} bytes"}
        )

    try:
        arr, preprocess_meta, t_prep = detector.preprocess_image(body)
    except ValueError as e:
        code = str(e)
        status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if code == "TOO_MANY_PIXELS" else status.HTTP_422_UNPROCESSABLE_ENTITY
        return JSONResponse(status_code=status_code, content={"code": code, "message": "Image processing rejected"})
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"code": "DECODE_FAILED", "message": "Image failed to decode"}
        )

    t_wait_start = time.perf_counter()
    acquired = inference_lock.acquire(timeout=2.0)
    t_wait = (time.perf_counter() - t_wait_start) * 1000.0

    if not acquired:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": "1"},
            content={"code": "QUEUE_FULL", "message": "Inference slot acquisition timed out"}
        )

    try:
        _, score, t_infer = detector.infer(arr)
    finally:
        inference_lock.release()

    verdict, thresh_info = detector.decide(score, tier="1pct")
    t_total = (time.perf_counter() - t_start) * 1000.0

    return {
        "request_id": request_id,
        "verdict": verdict,
        "score": round(score, 4),
        "thresholds": thresh_info,
        "model": {
            "version": detector.model_card.get("version", "stageA-residual-1"),
            "size_mb": detector.model_card.get("size_mb", 0.38),
            "card": "/v1/model-card",
        },
        "preprocess": preprocess_meta,
        "reliability": {
            "validated_families": ["latent-diffusion (SD v1.4/v1.5/Wukong)", "GLIDE", "Midjourney"],
            "weak_families": ["GAN (BigGAN)", "VQ-diffusion", "pixel-space diffusion (ADM)"],
            "note": "On weak_families, the error rate among decided requests is 29-43%; no_traces_detected must not be used as evidence of authenticity. See the model card.",
        },
        "timing_ms": {
            "preprocess": round(t_prep, 2),
            "queue_wait": round(t_wait, 2),
            "inference": round(t_infer, 2),
            "total": round(t_total, 2),
        },
    }


if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")