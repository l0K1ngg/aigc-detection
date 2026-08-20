FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 构建期动态计算基准 Logit 并生成 golden.json 指纹文件
RUN python -c "import onnxruntime as ort, numpy as np, json, os; \
    s = ort.InferenceSession('serve/model.onnx', providers=['CPUExecutionProvider']); \
    x = np.zeros((1, 224, 224, 3), dtype=np.uint8); \
    out = float(s.run(None, {s.get_inputs()[0].name: x})[0].flatten()[0]); \
    json.dump({'golden_logit': out, 'tolerance': 1e-4}, open('serve/golden.json', 'w')); \
    print('Generated golden fingerprint logit:', out)"

EXPOSE 8000

CMD ["gunicorn", "-c", "gunicorn_conf.py", "app:app"]