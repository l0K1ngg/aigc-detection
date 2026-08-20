import multiprocessing
import os

bind = 0.0.0.08000
workers = int(os.getenv(WEB_CONCURRENCY, max(1, multiprocessing.cpu_count())))
worker_class = uvicorn.workers.UvicornWorker
threads = 1
keepalive = 5
timeout = 30
loglevel = info
accesslog = -
errorlog = -