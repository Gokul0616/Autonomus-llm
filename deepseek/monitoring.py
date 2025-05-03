# monitoring.py
import time
import psutil
import prometheus_client
from prometheus_client import Gauge, Counter

class ModelMonitor:
    def __init__(self):
        self.inference_latency = Gauge('model_inference_latency', 'Latency in milliseconds')
        self.memory_usage = Gauge('model_memory_usage', 'RAM usage in MB')
        self.requests = Counter('model_requests', 'Total inference requests')
        
    def track(self, fn):
        def wrapper(*args, **kwargs):
            start = time.time()
            self.requests.inc()
            
            result = fn(*args, **kwargs)
            
            # Record metrics
            self.inference_latency.set((time.time()-start)*1000)
            self.memory_usage.set(psutil.Process().memory_info().rss/1024/1024)
            
            return result
        return wrapper

    def start_server(self, port=8000):
        prometheus_client.start_http_server(port)