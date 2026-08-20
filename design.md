# Design and Decision Record

This document records the full rationale behind the data pipeline, evaluation
protocol, calibration method, and service architecture. It is written for a reader
who wants to know where every number came from and why every choice was made the
way it was. The API contract lives in `api_contract.md`; the results summary lives
in `README.md`.

It is organized chronologically and includes the directions that were abandoned —
the reasons for abandoning them are part of the content.

---

## 1. Evaluation protocol: why GenImage in the end

### 1.1 The abandoned direction

The initial choice was OpenFake v2 (2.49M rows / 3.44 TB): newer generators, broader
real-image sources. An audit killed it. On a 3,000-image sample, real images were
98.9% JPEG while fakes were 63.4% PNG (a 63-point gap), median dimensions differed
by 31.7%, and 25% of the training set was >2048px — almost all of it real.

The problem was not the shortcut itself — GenImage's shortcut is worse — but the
**absence of a reference frame**. However clean a self-built evaluation set is, its
numbers cannot be compared against anything published, and a reader has no way to
judge whether 82 is good or bad.

### 1.2 The final protocol

Standard practice in this literature: train only on GenImage's sdv4 (Stable
Diffusion v1.4) subset; test on the validation sets of all 8 generators; report
unweighted macro-average mAcc/mAP.

The cost is accepting GenImage's inherent biases (Section 3). The benefit is that
the numbers line up cell-for-cell against CNNDet / UnivFD / NPR / AIDE / B-Free /
SAFE / CoD / PGC. **Comparability was prioritized over cleanliness** — because this
is a project that outside readers need to be able to evaluate.

---

## 2. Data pipeline

### 2.1 Scale and source

The raw data is 8 subsets of multi-part zip archives, ~600 GB total. `unzip` cannot
read multi-part archives; 7z reads them directly (no need to first merge into a
single zip, saving a full copy of disk), and supports `-i!"*/val/*"` to extract
only validation sets and `-i@list` for precise sampling.

Only sdv4 in full plus the validation sets of the other 7 subsets are needed:
~50 GB / 200k images after extraction.

### 2.2 Integrity scan

The full scan uses `PIL.Image.load()` — a real decode — not `verify()`, which only
checks headers and passes truncated files.

Result: 758 of sdv4's 112k images were corrupt (0.677%, `UnidentifiedImageError`);
the other 7 subsets' validation sets had zero. The corruption was 100% on the
generated side. The cause was not investigated; the files were blocklisted and
excluded. Impact: the training set becomes 49,242 : 50,000, a 0.76% imbalance,
negligible.

### 2.3 Shard shuffling: a non-obvious trap

The source directory structure lists generated images before real ones. With
**inter-shard random routing alone**, each shard's internal write order remains the
source order — first half all generated, second half all real — and sequential
reads produce single-class batches.

Two levels of shuffling are required: random routing across shards, plus a shuffle
buffer inside every shard writer. Verification: per-shard generated-image fraction
0.415–0.593; adjacent-label transitions inside shards at 68–108% of the random
expectation; no single-class batches at batch size 16 under sequential reads.

### 2.4 What the geometry census found

The initial scan recorded only the longest edge; a follow-up added the short-edge
distribution and produced an important fact:

| Subset | Fake short edge | Real short edge, p50 |
|---|---|---|
| biggan | 128 (p1 = p99 = max) | 375 |
| adm / glide / vqdm | 256 (constant) | 375 |
| sdv4 / sdv5 / wukong | 512 (constant) | 375 |
| midjourney | 1024 (constant) | 375 |

**Each subset's fakes have a constant size.** Two consequences:

1. It is a metadata shortcut (Section 3).
2. It dictates the preprocessing strategy. In the sdv4 training set, the 471 images
   with short edge <224 are **100% real** (fakes are constant 512). Under the naive
   "upscale only images that are too small" policy, "has been upscaled" and "is
   real" become synonymous in training — the model can learn "interpolation
   smoothness → real." At test time every biggan fake is 128 and must be upscaled,
   so that policy collapses the biggan column systematically.

   Training therefore uses a **random scale factor** (log-uniform 0.5–2.0) instead
   of a fixed target edge, so both classes experience both up- and down-sampling.

   > A later result revised the weight of this judgment: biggan's failure is fully
   > explained by generator-family differences (glide and adm/vqdm both produce
   > 256px fakes yet differ by 70 points). The measure is kept, but it is not the
   > main cause of the biggan collapse.

3. It also killed the "pre-crop to 320 to save space" idea: 26.76% of images are
   <320, and adm/glide/vqdm fakes are only 256 — there is nothing to crop.

Also: adm's 6,000 fakes are all RGBA with fully opaque alpha (verified), so
`convert('RGB')` is side-effect-free; each subset has ~100–880 grayscale images and
a handful of CMYK, all converted to RGB.

---

## 3. The metadata shortcut: quantified, controlled, and bounded

> This section is one of the project's main findings.

### 3.1 The observation

Two rules that look at no pixels, evaluated on all 100,000 validation images:

| Rule | Criterion | mAcc |
|---|---|---|
| R1 | container format is PNG → predict generated | **100.00%** |
| R2 | short edge exactly in {128, 256, 512, 1024} → predict generated | 99.94% |
| R1 ∪ R2 | either fires | 99.94% |

R1 is 100.00% on every one of the 8 subsets: fakes are 100% PNG, reals are 100%
JPEG, across all 16 (subset × label) groups without exception.

A detail worth recording: R1 ∪ R2 is **lower** than R1 alone. R1 is already at the
ceiling; the union can only add R2's false positives on real images (a few ImageNet
photos happen to have short edge exactly 256 or 512). **Combining shortcuts is not
monotone improvement** — rule baselines need per-rule verification too.

### 3.2 What it does and does not mean

It does **not** mean published methods cheat, nor that their numbers are invalid.
Pixel-level models receive decoded RGB tensors; container format never enters the
network, and absolute size disappears after cropping.

It **does** mean the label is structurally bound to two physical quantities, both of
which leave visible traces in pixels:

1. **Compression artifacts.** Reals are JPEG-compressed ImageNet photos carrying
   8×8 blocking and chroma subsampling; fakes are lossless PNGs carrying neither.
2. **Resampling traces.** Fakes have constant sizes and reals a natural
   distribution, so under any resize-to-fixed-input preprocessing, the scaling
   ratio correlates with the label.

For methods that live on high-frequency statistics (residual streams, the NPR
family), both traces land exactly on their only signal channel. The conclusion must
therefore be stated as: **numbers measured under the standard GenImage protocol are
a composite of "generation-trace detection" and "compression/resampling-artifact
discrimination," not a pure measure of the former.**

This is not a new finding — GenImage++ (NeurIPS 2025, arXiv 2506.00874, "Breaking
Latent Prior Bias") targets exactly this class of problem. The contribution here is
not pointing it out but **quantifying it to concrete numbers and building
verifiable controls into the pipeline.**

### 3.3 Controls

| Measure | Shortcut removed | Where |
|---|---|---|
| Uniform decode to 8-bit RGB; drop container info and alpha | format (R1) | dataloader / service |
| Fixed 224×224 crop; never expose original size to the network | absolute size (R2) | transform |
| Random scale **factor** in training, not fixed target edge | resampling–label correlation | transform |
| Random JPEG recompression + Gaussian blur (p=0.1 each) | compression-artifact gap | transform |
| Service reads no format / EXIF / filename / dimensions for decisions | all metadata shortcuts | service layer |

The service does **not** apply EXIF orientation correction: training didn't, so the
service doing it would be a train/serve inconsistency — and rotation changes the
directionality of high-frequency statistics (the residual stream's horizontal and
vertical differences are anisotropic). When an orientation tag is present, the
response reports it honestly.

### 3.4 Limits

The complete fix is switching benchmarks (GenImage++ / OpenFake, whose real images
are contemporaneous with the generators), which is outside this project's
comparison target. GenImage's real images have irreversibly lost information; any
post-processing can only narrow the gap, never restore it.

These two rules also target GenImage's distribution format specifically. Once data
is redistributed, transcoded, or screenshotted, the metadata shortcut vanishes —
which shows it is an artifact of dataset construction, not an intrinsic property of
generated images.

---

## 4. Model

### 4.1 Selection

The single **residual-stream** branch of PGC (arXiv 2605.21207). The paper's
ablation shows this branch alone reaches 93.4 mAcc — above AIDE and B-Free — and
explicitly warns that naive two-stream concatenation regresses to 92.1, so the
single-stream baseline must stand first.

Stage B (DINOv2-Large LoRA + peak-guided calibration) was **deliberately dropped**.
Not for compute — for positioning: the project's selling point is a load-testable
inference service, with the model as a replaceable component. The three-layer CNN
is an asset for service metrics, not a compromise — 0.38 MB, clean ONNX export,
millisecond inference. DINOv2-Large would destroy the latency and size column.

### 4.2 Implementation and deviations

Residual operator: first-order RGB differences (horizontal + vertical) truncated to
±3 and normalized, 6 channels; three conv layers (32 / 64 / 128, stride 1/2/2, each
with BN + ReLU); GAP; single linear head.

Training: T4, bs 128 / lr 5e-4 with cosine decay / 3 epochs, 13.7 min/epoch
(~110 img/s, dataloader bound by 2 vCPUs). Dev accuracy peaked at 98.10 in epoch 1
and fell back to 97.25 in epoch 2.

> **Deviation notice.** The paper uses bs 32 / lr 5e-5 (not adopted, compute
> limits); the residual operator was reconstructed from common practice, not
> verified line-by-line against the authors' code. These results are a
> **self-implemented baseline under the same protocol, not a faithful reproduction
> of PGC.**
>
> Also worth stating: the 11-point gap is almost certainly not the learning rate.
> Dev accuracy hit 98.10 in epoch 1 and regressed in epoch 2 — in-distribution
> convergence came early; what is missing is cross-generator generalization, a
> property of the features themselves.

### 4.3 The structure of the results

Per-subset numbers are in the README. Three points worth recording:

1. **Real-image accuracy is a constant.** 97.9–98.3 across all 8 subsets — nearly
   identical, because all 8 real sets come from the same ImageNet pool. Hence
   `mAcc ≈ (98 + fake-accuracy) / 2`; the whole table reduces to one column. It is
   also a sanity check: a stable 2% false-positive rate on reals, no
   subset-specific pipeline bug.

2. **Failure tracks generator family, not resolution.** glide and adm/vqdm all
   produce 256px fakes yet differ by 70 points. Re-sorted by family the pattern is
   clean: latent diffusion (sharing SD's VAE decoder) → diffusion → pixel-space
   diffusion → VQ → GAN, monotonically decreasing. The residual stream learned the
   upsampling fingerprint of SD's VAE decoder.

3. **The AP–acc gap distinguishes two failure kinds.** biggan: acc 56.97 but AP
   75.36 (18-point gap) — ranking information exists, the scores are shifted down.
   vqdm: acc 57.77 / AP 60.19 (2-point gap, AP barely 10 above random) — the signal
   barely exists. The former looks like a calibration problem, the latter a
   capability problem.

   > **That inference was later falsified.** See Section 5.3 — AP is invariant to
   > monotone score shifts and cannot say which side of an absolute threshold the
   > scores fall on.

---

## 5. Calibration and the three-state verdict

### 5.1 Dev-set-only calibration

Thresholds are calibrated only on the dev set (2 shards held out from sdv4 train),
never on any test subset. A deployed system never holds samples from a future
generator; calibrating on test sets is peeking. The 8 test subsets are used solely
to **report how that calibration performs**.

By the same principle, the sdv4 validation set never participates in model
selection — otherwise that column (usually the highest of the eight) carries
selection bias.

### 5.2 What the two lines mean

- `t_high` = the (1−α) quantile of dev real scores → above it, `generated`
  (controls false positives)
- `t_low` = the α quantile of dev fake scores → below it, `no_traces_detected`
  (controls misses)
- between them: `uncertain`

α = 1%: `t_low = 0.3936` / `t_high = 0.6389`; the three states are live.

α = 5%: `t_high = 0.3015` while `t_low = 0.6432` — the lines **overlap**. Overlap
means "this error target is loose enough that any single threshold inside the
interval satisfies both sides" — **not** that the model can't support it. This tier
degenerates to a single threshold of 0.5.

Statistical bound: the dev set has only 3,115 fakes, so the 1% quantile rests on
~31 samples — already at the edge of reliability. A stricter tier (0.1%) needs ~10×
the dev data, which requires retraining; not offered.

### 5.3 The real failure mode revealed by calibration

The three-state distribution table is in the README. It falsified the inference in
Section 4.3, point 3:

**78.2% of biggan fakes are ruled `no_traces_detected`**; only 11.2% land in
`uncertain`. vqdm: 80.7%.

The mistake was over-reading AP. **AP measures ranking within a subset, and ranking
is invariant to any monotone score shift** — it cannot say which side of an absolute
threshold the scores fall on. biggan's fakes crowd the low-score region together
with its reals, heavily overlapped. This is not a calibration problem; it is a
representation problem: **no threshold scheme on this scalar score can fix it.**

Two conclusions:

1. Three-state design catches borderline samples, not out-of-distribution samples.
   **Silent failure is real.**
2. The middle verdict must be named `no_traces_detected`, never `real`
   (see `api_contract.md`, Section 0).

### 5.4 The remedy not implemented

A score-independent OOD gate: take the 128-dim GAP feature, fit mean and covariance
on dev, route inputs whose Mahalanobis distance exceeds a dev quantile straight to
`uncertain`, ignoring the score. Requires a second output head and re-calibration;
ranked below finishing the service path, so not done.

Deliberately **not** using input resolution as an OOD feature — that would carry the
dataset's metadata shortcut into the service.

---

## 6. From model to service

### 6.1 ONNX export: the residual operator lives in the graph

The residual operator is the model's only feature extractor. A second
implementation on the service side would drift: an off-by-one in boundary handling
makes production scores silently disagree with offline evaluation, with no error
raised anywhere. So it was folded into the graph; the service only decodes, crops
to 224, and feeds uint8 NHWC.

Cost: the training formulation (`zeros_like` + slice assignment) exports to
ScatterND and was rewritten with `F.pad`. The export pipeline **refuses to write the
model until both formulations verify numerically equivalent** (measured max
absolute difference: 0.000e+00).

Verification: max |Δlogit| between PyTorch and ONNX Runtime = 6.68e-6 across batch
sizes 1/2/8/32 plus 256 real images, zero verdict flips. Random inputs do not cover
the real residual distribution (real images have highly correlated neighboring
pixels, differences near 0; random inputs saturate the ±3 truncation), so real
images are mandatory in the check.

Preprocessing parameters are written into ONNX metadata and read from the model
file at startup — `model_card.json` and the code cannot disagree.

### 6.2 The golden fingerprint

A logit computed on a fixed-seed synthetic input, stored in `golden.json`,
generated at build time and re-verified at startup (tolerance 1e-4). A swapped
model file, a changed in-graph preprocessing step, or an ORT upgrade that shifts
numerics — any of these makes `/readyz` return 503 immediately, instead of letting
production scores drift quietly.

### 6.3 Compute-bound, therefore no dynamic batching

Measured (2 vCPU, ORT CPU provider):

| Scheme | Throughput | p50 latency |
|---|---|---|
| Serial, one image at a time | 71.6 img/s | 13.7 ms |
| Dynamic batching (4 decode threads, batch ≤32) | **56.7 img/s** | **69.9 ms** |

Of the 13.7 ms end-to-end, decode+preprocess is 3.3 ms and inference 10.4 ms —
**inference is 76%**.

Why: only 94k parameters, but the first conv runs at full 224×224 with stride 1 —
~1.1 GFLOPs total (61% of ResNet-18). Back-solving, 1.10 GFLOP / 0.0104 s ≈
106 GFLOP/s, near the AVX2+FMA peak of two cores. **Small parameter count does not
mean small compute.**

In a compute-bound regime, batching has no call overhead to amortize; extra threads
only contend for the same cores. The service therefore uses synchronous
single-image inference with multi-process horizontal scaling, and pins ORT's
`intra_op_num_threads` to 1 (default is core count; W workers without the pin
reproduce exactly the contention that sank batching).

### 6.4 int8 quantization: feasible, not adopted

The residual operator **must be excluded from quantization**: it differences in the
0–255 range and then truncates to ±3. If the quantizer sets the scale from the Sub
output's raw range (±255), the int8 step is ~2 while the signal is only ±3 — seven
useful levels collapse to three. So `op_types_to_quantize` was limited to
Conv/Gemm/MatMul; QDQ wraps only the convolutions. Calibration used dev shards
only, Percentile 99.999.

Accuracy: mAcc 81.58 → 81.88 (sampled evaluation), verdict flip rate 2.04% —
lossless on the accuracy side.

Error character: the maximum score deviation was **identically 0.1208 across six
subsets**. Inverting: `sigmoid(-0.495) − 0.5 = 0.1203` — a constant logit offset of
about **-0.5**. Systematic bias, not random noise, hence fully absorbable by
re-calibration (int8 lines: `t_low = 0.2466` / `t_high = 0.5512`).

**Not adopted.** The gain does not cover the operational complexity of two coexisting
threshold sets — and mixing them fails in exactly the worst direction: the -0.5
offset pushes weak-family fakes further into `no_traces_detected`.

### 6.5 Backpressure instead of unbounded queuing

One inference slot per worker; waiting longer than 2 s for the slot returns 429
with `Retry-After`. A compute-bound service with unbounded queuing blows p99 to
seconds under overload — better to fail fast and let upstream retry.
`detect_rejected_total` thereby becomes a meaningful capacity signal.

The response body and logs time `preprocess / queue_wait / inference` separately:
under load, rising `queue_wait` with flat `inference` means demand exceeds capacity
(healthy); rising `inference` means core contention.

### 6.6 No capacity load test

The test environment had 2 cores with the load generator on the same host. Under a
compute-bound workload, every CPU cycle the client takes directly lowers the
service's throughput and inflates its self-reported inference time. The resulting
capacity numbers are neither reproducible nor citable — so the measurement was not
made.

**The single-request latency numbers remain valid**: they measure how long one image
takes on one core, and the evidence behind one architecture decision — neither
depends on machine size. Hence: single-request latency is reported; QPS and
capacity are not.

---

## 7. One front-end design decision

There is **no green in the interface**. `no_traces_detected` renders in neutral
gray, not a green checkmark.

Green means "pass / safe," and that state is wrong roughly four times in ten on the
weak generator families. The contract refuses to call it `real`; the visual layer
equally must not make it look like clearance.

The signature element is the calibration scale: `t_low` and `t_high` drawn to true
proportion on a 0→1 axis, the three intervals labeled, the score as a needle on
top. Thresholds are fetched from `/readyz`, never hard-coded in the page —
consistent with the anti-drift design throughout.

The box on the thumbnail is the region the model actually sees, computed from the
preprocessing contract (short side to 256, center-crop 224 — i.e. 87.5% of the
short side); on elongated images the trimmed sides are clearly visible.

---

## 8. Decision log

| Decision | Basis | Outcome |
|---|---|---|
| Benchmark: OpenFake → GenImage | self-built eval has no reference frame | numbers comparable cell-for-cell with published tables |
| Drop Stage B (DINOv2 + peak calibration) | selling point is the service, not the model | latency/size column preserved |
| Random scale factor in training | small training images are 100% real | kept; measured impact limited |
| Residual operator inside the ONNX graph | dual implementations drift silently | equivalence verified at 0.000e+00 |
| Remove dynamic batching | batching cut throughput 71.6 → 56.7 | multi-process, threads pinned to 1 |
| Keep fp32, skip int8 | gain doesn't cover dual-threshold complexity | single source of truth for thresholds |
| No OOD gate (option A) | ranked below finishing the service path | documented honestly as a known defect |
| No capacity load test | 2-core same-host numbers aren't citable | single-request latency only |

### Predictions falsified by measurement

Recorded because the falsification process is itself valuable:

1. **"biggan's 18 points are recoverable by calibration"** — wrong. AP is invariant
   to monotone shifts and says nothing about absolute thresholds. Measured: 78.2%
   of biggan fakes ruled `no_traces_detected`.
2. **"The service is decode-bound"** — wrong. Inference is 76% of end-to-end. Small
   parameter count does not mean small compute.
3. **"Scale jitter is the key to the biggan collapse"** — limited effect. The main
   cause is generator family (glide vs vqdm, both 256px, differ by 70 points).
