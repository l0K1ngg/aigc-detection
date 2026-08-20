import multiprocessing
import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
worker_class = "uvicorn.workers.UvicornWorker"
threads = 1
keepalive = 5
timeout = 30
loglevel = "info"
accesslog = "-"
errorlog = "-"
