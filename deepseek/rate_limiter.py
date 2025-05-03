# rate_limiter.py
import time
from collections import deque

class RateLimiter:
    def __init__(self, rpm=100):
        self.rpm = rpm
        self.times = deque()
        
    def __call__(self):
        now = time.time()
        while self.times and now - self.times[0] > 60:
            self.times.popleft()
            
        if len(self.times) >= self.rpm:
            raise RateLimitExceeded()
        
        self.times.append(now)

class RateLimitExceeded(Exception):
    pass