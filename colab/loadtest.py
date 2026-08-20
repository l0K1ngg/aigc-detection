"""
Benchmark harness for single-request vs dynamic batching latency.
NOTE: Full capacity load testing was deliberately omitted due to same-host
compute contention on 2 vCPUs (see design.md Section 6.6).
"""
import time
import numpy as np
import onnxruntime as ort

def benchmark_single_request(model_path: str, iterations: int = 500):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
    
    dummy = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    inp_name = session.get_inputs()[0].name
    
    # Warmup
    for _ in range(50):
        session.run(None, {inp_name: dummy})
        
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, {inp_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    print(f"Single-image CPU latency (ms): p50={p50:.2f}, p95={p95:.2f}, p99={p99:.2f}")

if __name__ == "__main__":
    benchmark_single_request("serve/model.onnx")