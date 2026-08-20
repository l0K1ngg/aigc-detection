# Detection Service API Contract v1

## 0. A naming decision: there is no `real` verdict

The model detects **generation traces**, not authenticity. Calibration shows that on
generator families absent from training (biggan / vqdm / adm), 56–81% of fakes land
in the low-score region, and the error rate among confidently decided requests
reaches 29–43%. A low score therefore means "no traces detected" — not "this image
is real." On unseen generators those two statements differ by about forty
percentage points.

The contract accordingly forbids the words `real` / `authentic`. The verdict values
are:

| verdict | Means | Does NOT mean |
|---|---|---|
| `generated` | generation traces detected | — |
| `no_traces_detected` | no known traces found | **the image is authentic** |
| `uncertain` | score falls inside the calibrated uncertainty band | — |

This is not cautious wording; it aligns the contract with measured capability. Any
downstream logic that treats `no_traces_detected` as "authentic" will be wrong at
roughly that rate on unseen generators.

## 1. Synchronous detection

```
POST /v1/detect
Content-Type: multipart/form-data | application/octet-stream
```

**Response 200**

```json
{
  "request_id": "01J9X8ZK3M7Q",
  "verdict": "generated",
  "score": 0.9871,
  "thresholds": { "tier": "1pct", "t_low": 0.3936, "t_high": 0.6389 },
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
  "timing_ms": { "preprocess": 6.1, "queue_wait": 0.1, "inference": 1.4, "total": 10.2 }
}
```

`score` = `sigmoid(logit)` = P(generated). **The raw score is always returned** —
it is never hidden when the verdict is `uncertain`. Downstream consumers may have
their own risk preferences and need the raw value.

`preprocess.upscaled` = true means the source's short edge was <224 and the image
was enlarged. This is a known reliability risk factor (the weak families' fakes are
exactly the small-sized ones); downstream may weight accordingly.

## 2. Asynchronous detection

```
POST /v1/jobs          -> 202 { "job_id": "...", "status": "queued" }
GET  /v1/jobs/{job_id} -> 200 { "status": "queued|running|done|failed", "result": {...} }
```

When `status=done`, `result` has the same shape as the synchronous response. Batch
submission goes through `POST /v1/jobs:batch`; each entry succeeds or fails
independently — one bad image never rolls back the batch.

*(Not yet implemented; the synchronous endpoint covers current use. Recorded here so
the shape is fixed before implementation.)*

## 3. Error codes

| HTTP | code | Trigger |
|---|---|---|
| 400 | `DECODE_FAILED` | not decodable as an image (includes truncated files) |
| 400 | `UNSUPPORTED_FORMAT` | decoded, but the format is not on the allowlist |
| 413 | `PAYLOAD_TOO_LARGE` | exceeds `max_bytes` (default 10 MB) |
| 413 | `TOO_MANY_PIXELS` | exceeds `max_pixels` (default 50M, decompression-bomb guard) |
| 422 | `IMAGE_TOO_SMALL` | short edge < 32; enlarging to 224 is meaningless |
| 429 | `QUEUE_FULL` | queue watermark exceeded; carries `Retry-After` |
| 503 | `MODEL_NOT_READY` | model or threshold config not loaded |

## 4. The service reads no metadata

The evaluation benchmark contains a metadata shortcut (container format PNG alone
classifies with 100% accuracy — see the corresponding section of `design.md`). The
service **consumes decoded pixels only**; container format, EXIF, filename, and
dimensions never participate in a decision. The fields in the `preprocess` block are
returned as diagnostics only.

EXIF orientation is **not corrected** — training did not correct it, so the service
doing so would be a train/serve inconsistency. When an orientation tag exists, it is
reported honestly via `exif_orientation_present`.

## 5. Threshold tiers

Calibration uses only the dev set (shards held out from sdv4 train), never any test
subset: a deployed system cannot hold samples from future generators, and
calibrating on test sets amounts to peeking.

- **1% tier (default)**: `t_low = 0.3936` / `t_high = 0.6389`; three states are live.
- **5% tier**: the two lines overlap, meaning that error target is loose enough for a
  single threshold — equivalent to a fixed cutoff at 0.5; no uncertainty band needed.

The dev set has only 3,115 fakes; the 1% quantile rests on ~31 samples and is
already at the edge of statistical reliability. A stricter tier (0.1%) would need
~10× the dev data, which requires retraining — not offered.

## 6. Identified but not implemented

**A score-independent OOD gate.** The current failure mode is that
out-of-distribution inputs score in the low region, indistinguishable from reals; a
scalar score cannot express "I have not seen an input like this." A workable design:
take the 128-dim GAP feature, fit mean and covariance on dev, and route inputs past
a Mahalanobis-distance threshold straight to `uncertain`.

Not implemented because it needs a second output head plus re-calibration, and
ranked below finishing the service path. Recorded here so it is not mistaken for an
unknown problem.
