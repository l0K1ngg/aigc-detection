## 0. 设计原则

1. **推理进程与 Web 进程物理分离。** API 层永不加载模型权重。否则「延迟优化」无从谈起。
2. **一切结果可追溯。** 每条结果必须能答出：哪个模型版本、哪个阈值、多久算完的。
3. **同一张图不算第二遍。** 去重不是优化项，是核心设计，缓存命中率是要写进简历的数字。

---

## 1. 系统边界

```
                    ┌──────────────────────┐
   Client / Agent ──▶  FastAPI 接入层       │  无模型、无 GPU、可横向扩
                    │  鉴权 / 限流 / 校验    │
                    └──────────┬───────────┘
                               │ enqueue
                    ┌──────────▼───────────┐
                    │  Redis               │  队列 + 去重缓存 + 结果缓存
                    └──────────┬───────────┘
                               │ consume
                    ┌──────────▼───────────┐
                    │  Inference Worker    │  ONNX Runtime，动态 batching
                    │  (GPU)               │  唯一持有权重的进程
                    └──────────┬───────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
   ┌───────────┐        ┌───────────┐         ┌────────────┐
   │ Postgres  │        │  MinIO    │         │ Prometheus │
   │ 任务/结果  │        │ 图片/热力图 │         │  指标       │
   └───────────┘        └───────────┘         └────────────┘
```

**关键约束**：Worker 是唯一 import torch / onnxruntime 的地方。API 层的 requirements.txt 里不应出现深度学习框架。这条在 CI 里加一个检查。

---

## 2. 接口契约

Base path: `/api/v1`。所有响应 `Content-Type: application/json`。

### 2.1 提交检测任务

```
POST /api/v1/detections
Content-Type: multipart/form-data
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | binary | 二选一 | 图片文件，≤ 10 MB |
| `image_url` | string | 二选一 | 远程图片地址（需通过 SSRF 白名单校验）|
| `explain` | bool | 否 | 是否生成 Grad-CAM 热力图，默认 `false`（生成会显著增加耗时，别默认开）|
| `model_version` | string | 否 | 指定模型版本，默认取 registry 中 `is_active` 的那个 |

支持格式：`image/jpeg`、`image/png`、`image/webp`。**按 magic bytes 判定，不信 Content-Type，不信扩展名。**

**201 Created**

```json
{
  "task_id": "0193f2a1-...",
  "status": "QUEUED",
  "dedup": "MISS",
  "poll_url": "/api/v1/detections/0193f2a1-...",
  "created_at": "2026-08-13T09:12:33Z"
}
```

去重命中时直接返回 `200 OK`，`status` 为 `SUCCEEDED`，`dedup` 为 `EXACT` 或 `NEAR`，并携带完整 `result` 对象。

### 2.2 查询任务

```
GET /api/v1/detections/{task_id}
```

**200 OK**

```json
{
  "task_id": "0193f2a1-...",
  "status": "SUCCEEDED",
  "created_at": "2026-08-13T09:12:33Z",
  "finished_at": "2026-08-13T09:12:34Z",
  "queue_wait_ms": 213,
  "inference_ms": 47,
  "result": {
    "label": "AI_GENERATED",
    "score_ai": 0.9412,
    "calibrated_score": 0.8873,
    "threshold": 0.5,
    "model_version": "effnet-b0-v3",
    "backend": "onnxruntime-cuda",
    "heatmap_url": "/api/v1/detections/0193f2a1-.../heatmap"
  },
  "image": {
    "sha256": "e3b0c442...",
    "width": 1024,
    "height": 1024,
    "mime": "image/png"
  }
}
```

`label` 枚举：`AI_GENERATED` / `HUMAN_MADE` / `UNCERTAIN`。

> **设计决策**：保留 `UNCERTAIN`。校准分落在 `[0.4, 0.6]` 区间时返回它。二分类硬判在这个任务上是不诚实的，面试时这一点可以主动讲。

任务未完成时 `result` 为 `null`，`status` 为 `PENDING` / `QUEUED` / `RUNNING`。

### 2.3 热力图

```
GET /api/v1/detections/{task_id}/heatmap
→ 302 Found，Location 指向 MinIO 预签名 URL（有效期 15 分钟）
```

未生成时返回 `404` + 错误码 `HEATMAP_NOT_GENERATED`。

### 2.4 同步端点（给 Agent 用）

```
POST /api/v1/tools/detect-image
```

同步阻塞，超时 5 秒；超时返回 `202 Accepted` + `task_id`，降级为异步。请求体用 JSON（`image_url` 或 base64），响应体是扁平结构，方便 function calling 直接消费：

```json
{
  "is_ai_generated": true,
  "confidence": 0.887,
  "verdict": "UNCERTAIN 以上，建议人工复核",
  "model_version": "effnet-b0-v3"
}
```

这个端点的 OpenAPI schema 就是它的工具定义，`/openapi.json` 直接可用。

### 2.5 其他

| 端点 | 用途 |
|---|---|
| `GET /api/v1/detections?cursor=&limit=` | 游标分页列表 |
| `GET /api/v1/models` | 模型版本列表 + 各自离线指标 |
| `GET /healthz` | 存活探针，不查依赖 |
| `GET /readyz` | 就绪探针，查 Postgres / Redis / MinIO |
| `GET /metrics` | Prometheus 抓取 |

### 2.6 错误响应

统一格式，HTTP 状态码 + 业务错误码双轨：

```json
{
  "error": {
    "code": "IMAGE_TOO_LARGE",
    "message": "Image exceeds 10 MB limit",
    "request_id": "req_01hq..."
  }
}
```

| code | HTTP | 触发条件 |
|---|---|---|
| `INVALID_IMAGE_FORMAT` | 400 | magic bytes 校验失败 |
| `IMAGE_TOO_LARGE` | 413 | 超过 10 MB |
| `IMAGE_DIMENSION_INVALID` | 400 | 任一边 < 64px 或 > 8192px |
| `URL_NOT_ALLOWED` | 400 | SSRF 白名单拦截 |
| `TASK_NOT_FOUND` | 404 | — |
| `HEATMAP_NOT_GENERATED` | 404 | 提交时 `explain=false` |
| `MODEL_VERSION_NOT_FOUND` | 400 | — |
| `RATE_LIMITED` | 429 | 令牌桶耗尽，带 `Retry-After` |
| `INFERENCE_FAILED` | 500 | Worker 异常，已重试 2 次 |
| `QUEUE_FULL` | 503 | 队列积压超阈值，带 `Retry-After` |

---

## 3. 任务状态机

```
PENDING ──▶ QUEUED ──▶ RUNNING ──┬──▶ SUCCEEDED
                          │       └──▶ FAILED
                          └──▶ QUEUED（重试，最多 2 次）
```

- `PENDING`：已落库，图片还没传完 MinIO
- `FAILED` 是终态，不再自动重试
- 无 `CANCELLED`：任务生命周期以秒计，取消功能纯属过度设计

---

## 4. 数据模型（Postgres）

### `image_asset`

图片实体，**与任务解耦**——同一张图被提交 100 次，这里只有一行。

| 列 | 类型 | 约束 |
|---|---|---|
| `id` | uuid | PK |
| `sha256` | char(64) | **UNIQUE**，去重主键 |
| `phash` | bigint | 感知哈希，建索引 |
| `object_key` | text | MinIO 对象键 |
| `mime` | text | |
| `width` / `height` | int | |
| `size_bytes` | bigint | |
| `created_at` | timestamptz | 默认 now() |

### `detection_task`

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | uuid | PK，也是对外的 task_id |
| `image_id` | uuid | FK → image_asset |
| `status` | text | 状态枚举 |
| `model_version` | text | 提交时快照，不随 registry 变动 |
| `explain` | bool | |
| `source` | text | `upload` / `url` / `agent` |
| `client_id` | text | 限流与统计维度 |
| `dedup` | text | `MISS` / `EXACT` / `NEAR` |
| `retry_count` | smallint | |
| `error_code` | text | nullable |
| `created_at` / `queued_at` / `started_at` / `finished_at` | timestamptz | 四个时间戳，用来算排队时长和端到端延迟 |

索引：`(status, created_at)` 给运维查积压；`(image_id, model_version)` 给去重查历史结果。

### `detection_result`

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | uuid | PK |
| `task_id` | uuid | FK，**UNIQUE**（一任务一结果）|
| `label` | text | 三值枚举 |
| `score_ai` | real | 模型原始输出 |
| `calibrated_score` | real | 温度缩放后 |
| `threshold` | real | 判定时用的阈值，快照 |
| `model_version` | text | |
| `backend` | text | `onnxruntime-cuda` / `tensorrt` / `pytorch` |
| `heatmap_key` | text | nullable |
| `inference_ms` | int | 纯推理耗时，不含排队和 IO |
| `batch_size` | smallint | 该次推理实际凑到的 batch 大小 |
| `created_at` | timestamptz | |

> `threshold` 和 `model_version` 冗余存储是刻意的。模型换代后，历史结果必须还能解释清楚当时是怎么判的。

### `model_registry`

| 列 | 类型 | 说明 |
|---|---|---|
| `version` | text | PK，如 `effnet-b0-v3` |
| `arch` | text | |
| `weights_key` / `onnx_key` | text | MinIO |
| `input_size` | int | |
| `calibration_temp` | real | 温度缩放参数 |
| `threshold` | real | |
| `metrics` | jsonb | 离线评测结果，见 §6 |
| `is_active` | bool | 部分唯一索引保证只有一个 true |
| `trained_at` | timestamptz | |

---

## 5. 去重与缓存策略

三层，从便宜到贵：

1. **Redis 精确层**：`sha256:{hash}:{model_version}` → `task_id`。命中直接返回，标 `EXACT`。TTL 7 天。
2. **Postgres 精确层**：Redis 未命中时查 `image_asset.sha256`，命中则回填 Redis。
3. **感知哈希层**：phash 汉明距离 ≤ 6 视为同图（缩放、轻度压缩、重编码）。命中标 `NEAR`，**结果照抄但在响应里注明来源图 sha256**。

> 第 3 层是加分项也是风险点：距离阈值定错会误判。先实现 1、2，第 3 层留到 Week 3，且必须有一组人工验证样本证明阈值合理。

MinIO 对象键：`images/{sha256[:2]}/{sha256[2:4]}/{sha256}.{ext}`，两级前缀散列避免单目录过大。热力图：`heatmaps/{task_id}.png`。

---

## 6. 指标埋点（对应简历数字，现在就定好）

**Prometheus 侧**

| 指标 | 类型 | 标签 |
|---|---|---|
| `detection_requests_total` | Counter | `status`, `dedup`, `source` |
| `detection_e2e_seconds` | Histogram | `explain` |
| `detection_queue_wait_seconds` | Histogram | — |
| `detection_inference_seconds` | Histogram | `backend`, `model_version` |
| `detection_batch_size` | Histogram | — |
| `inference_queue_depth` | Gauge | — |
| `gpu_utilization_percent` | Gauge | — |

**模型 registry.metrics 里必须有的字段**

```json
{
  "in_domain": { "auc": 0.0, "acc": 0.0, "f1": 0.0 },
  "cross_generator": {
    "<训练集外的生成器A>": { "auc": 0.0 },
    "<训练集外的生成器B>": { "auc": 0.0 }
  },
  "robustness": {
    "jpeg_q75": { "auc": 0.0 },
    "resize_0.5": { "auc": 0.0 }
  },
  "ece": 0.0
}
```

> **跨生成器泛化和 JPEG 鲁棒性是这个领域的真问题。** 只报 in-domain AUC 的简历，面试官一问就穿。哪怕数字掉得难看也要测，掉得难看本身就是可以聊的内容。

---

## 7. 仓库结构

```
aigc-detect/
├── docs/
│   ├── design.md              ← 本文件
│   └── architecture.png
├── api/                       ← FastAPI，无 torch 依赖
│   ├── routers/
│   ├── schemas/               ← Pydantic，接口契约的唯一真源
│   ├── services/
│   └── requirements.txt
├── worker/                    ← 推理，唯一持有权重
│   ├── engine/                ← onnx runtime 封装、动态 batching
│   ├── explain/               ← Grad-CAM
│   └── requirements.txt
├── training/                  ← 离线，不进生产镜像
│   ├── data/                  ← 数据集准备与切分
│   ├── train.py
│   ├── evaluate.py            ← 产出 §6 的 metrics json
│   └── export_onnx.py
├── migrations/                ← Alembic
├── bench/                     ← locust 压测脚本
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## 8. Step 1 收尾清单

- [x] 接口契约
- [x] 数据模型
- [x] 错误码表
- [x] 指标埋点清单
- [ ] 数据集选定（含许可确认）与切分方案 —— 必须留出**训练时完全没见过的生成器**做泛化测试集
- [ ] 架构图导出
- [ ] 仓库初始化 + Alembic 首版迁移 + Pydantic schema 落地

