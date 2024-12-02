import time
from functools import wraps
from logger_config import logger
from collections import deque
from threading import Lock
import statistics

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.request_times = deque(maxlen=window_size)
        self.lock = Lock()
        self.start_time = time.time()
        self.total_requests = 0
    
    def add_request_time(self, execution_time):
        with self.lock:
            self.request_times.append(execution_time)
            self.total_requests += 1
    
    def get_stats(self):
        with self.lock:
            if not self.request_times:
                return {
                    "avg_response_time": 0,
                    "min_response_time": 0,
                    "max_response_time": 0,
                    "total_requests": self.total_requests,
                    "requests_per_second": 0
                }
            
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            stats = {
                "avg_response_time": statistics.mean(self.request_times),
                "min_response_time": min(self.request_times),
                "max_response_time": max(self.request_times),
                "total_requests": self.total_requests,
                "requests_per_second": self.total_requests / elapsed_time if elapsed_time > 0 else 0
            }
            return stats

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Log execution time
            logger.info(f"{func.__name__} execution time: {execution_time:.2f} seconds")
            
            # Add to performance monitor
            performance_monitor.add_request_time(execution_time)
            
            # Log current performance stats
            stats = performance_monitor.get_stats()
            logger.info(f"Performance Stats: "
                       f"Avg Response Time: {stats['avg_response_time']:.2f}s, "
                       f"TPS: {stats['requests_per_second']:.2f}, "
                       f"Total Requests: {stats['total_requests']}")
            
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    return wrapper
