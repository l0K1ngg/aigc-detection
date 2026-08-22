# AIGC Trace Detector

A lightweight, CPU-optimized AIGC image detection service based on high-frequency residual stream extraction. The service detects generation artifacts (e.g., upsampling fingerprints from latent diffusion decoders) using a compact 3-layer CNN packaged into a 0.38 MB ONNX graph, delivering single-digit millisecond inference without requiring GPU infrastructure.

🔗 **Live Demo**: [https://aigc-detection.onrender.com/](https://aigc-detection.onrender.com/)

Full design rationale and decision logs are documented in [`design.md`](design.md).  
The HTTP API specification is detailed in [`api_contract.md`](api_contract.md).  

---

## Key Highlights

- **Lightweight & CPU-Bound**: 0.38 MB model footprint, 94k parameters, single-image inference latency ~10 ms on standard CPUs.
- **In-Graph Preprocessing**: The spatial residual operator ($1^{\text{st}}$-order differences truncated to $\pm 3$) is embedded directly into the ONNX graph, eliminating training/serving feature extraction drift.
- **Three-State Calibrated Verdicts**: Uses dev-calibrated quantile lines to output `generated`, `no_traces_detected`, or `uncertain` — explicitly avoiding misleading "authentic" or "real" classifications.
- **Metadata Shortcut Resistance**: Eliminates dataset-specific metadata exploits (PNG/JPEG format gaps, fixed short-edge heuristics) by enforcing raw 8-bit RGB decode, fixed center cropping, and zero EXIF reliance.
- **Drift Protection**: Validates execution numerics at startup via a build-time `golden.json` fingerprint; automatically fails `/readyz` if weights or runtimes change.

---

## Evaluation Benchmark & Cross-Generator Generalization

Trained on GenImage's `sdv4` (Stable Diffusion v1.4) subset and evaluated across 8 generator validation sets (100,000 images total):

| Subset / Generator | Family | Resolution | Acc (%) | AP (%) | `no_traces_detected` (%) | `uncertain` (%) | `generated` (%) |
|---|---|---|---|---|---|---|---|
| **sdv4** (in-distribution) | Latent Diffusion | 512 | 98.1 | 99.8 | 1.4 | 1.8 | 96.8 |
| **sdv5** | Latent Diffusion | 512 | 97.4 | 99.6 | 2.1 | 2.5 | 95.4 |
| **wukong** | Latent Diffusion | 512 | 96.2 | 99.1 | 3.2 | 3.6 | 93.2 |
| **midjourney** | Diffusion | 1024 | 88.5 | 94.2 | 8.6 | 6.2 | 85.2 |
| **glide** | Diffusion | 256 | 84.1 | 91.0 | 12.4 | 7.8 | 79.8 |
| **adm** | Pixel-space Diffusion | 256 | 62.3 | 68.4 | 56.1 | 9.4 | 34.5 |
| **vqdm** | VQ-Diffusion | 256 | 57.8 | 60.2 | 80.7 | 8.1 | 11.2 |
| **biggan** | GAN | 128 | 57.0 | 75.4 | 78.2 | 11.2 | 10.6 |
| **Macro Average** | — | — | **80.2** | **86.0** | — | — | — |

*Note: Real-image accuracy remains invariant at 97.9–98.3% across all subsets. Performance tracks generator family (Latent Diffusion sharing VAE decoders $\rightarrow$ GAN/VQ), rather than image resolution.*

---

## Quick Start

### 1. Run with Docker (Recommended)

```bash
# Build image (generates golden fingerprint at build time)
docker build -t aigc-detect .

# Run container on port 8000
docker run --rm -p 8000:8000 aigc-detect
```

Open `http://localhost:8000` in your browser to access the interactive detection web UI.

### 2. Local Python Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Start service with Gunicorn (Uvicorn workers)
gunicorn -c gunicorn_conf.py app:app
```

---

## API Usage

### Synchronous Detection (`POST /v1/detect`)

Send an image via `multipart/form-data` or raw binary `application/octet-stream`:

```bash
curl -X POST "http://localhost:8000/v1/detect" \
  -F "file=@sample.png"
```

**Example Response (`200 OK`)**:

```json
{
  "request_id": "01J9X8ZK3M7Q",
  "verdict": "generated",
  "score": 0.9871,
  "thresholds": {
    "tier": "1pct",
    "t_low": 0.3936,
    "t_high": 0.6389
  },
  "model": {
    "version": "stageA-residual-1",
    "size_mb": 0.38,
    "card": "/v1/model-card"
  },
  "preprocess": {
    "decoded_mode": "RGB",
    "source_min_edge": 512,
    "upscaled": false,
    "exif_orientation_present": false
  },
  "reliability": {
    "validated_families": ["latent-diffusion (SD v1.4/v1.5/Wukong)", "GLIDE", "Midjourney"],
    "weak_families": ["GAN (BigGAN)", "VQ-diffusion", "pixel-space diffusion (ADM)"],
    "note": "On weak_families, the error rate among decided requests is 29-43%; no_traces_detected must not be used as evidence of authenticity. See the model card."
  },
  "timing_ms": {
    "preprocess": 4.2,
    "queue_wait": 0.1,
    "inference": 8.9,
    "total": 13.2
  }
}
```

### Health & Readiness Check (`GET /readyz`)

```bash
curl "http://localhost:8000/readyz"
```

---

## Repository Structure

```
aigc-detect/
├── README.md                  # Project overview & quickstart
├── design.md                  # System design, data pipeline & decision records
├── api_contract.md            # HTTP API specifications & schema contracts
├── LICENSE                    # MIT License
├── .gitignore
├── .dockerignore
├── Dockerfile                 # Multi-stage image build with golden fingerprinting
├── requirements.txt
├── app.py                     # FastAPI application & route endpoints
├── detector.py                # Preprocessing + ONNX inference + calibration logic
├── gunicorn_conf.py           # Multi-process concurrency settings (pinned 1 thread/worker)
│
├── static/
│   └── index.html             # Web-based visualization UI (no-green neutral palette)
│
├── serve/                     # Committed model artifacts (< 1 MB)
│   ├── model.onnx             # 0.38 MB ONNX model (in-graph residual op)
│   ├── thresholds.json        # Calibrated three-state threshold bounds
│   └── model_card.json        # Model capability & limitation registry
│                              # (golden.json is auto-generated inside Docker)
│
├── colab/                     # Training & data extraction pipelines (reproducibility)
└── docs/
    └── zh/                    # Complete Chinese documentation mirrors
```

---

## Limitations & Ethical Notice

1. **No Authenticity Certification**: A verdict of `no_traces_detected` means that no specific generative artifacts were identified by the model; it is **not proof of image authenticity** (error rates on out-of-distribution generator families such as BigGAN/VQDM reach 29–43%).
2. **Dataset Notice**: This repository does not distribute any images from GenImage, ImageNet, or OpenFake datasets. All datasets must be obtained from their respective creators.

---

## License

This project is licensed under the [MIT License](LICENSE).
