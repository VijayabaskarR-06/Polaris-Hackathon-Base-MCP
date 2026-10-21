"""A small thread-safe LRU cache."""
import threading
from collections import OrderedDict


class LRU:
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.data = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.data:
                return None
            self.data.move_to_end(key)
            return self.data[key]

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            self.data.move_to_end(key)
            while len(self.data) > self.capacity:
                self.data.popitem(last=False)
        return value

    def __len__(self):
        return len(self.data)


class KeyLocks:
    """One lock per key, so concurrent identical requests compute once."""

    def __init__(self):
        self.locks = {}
        self.guard = threading.Lock()

    def get(self, key):
        with self.guard:
            lock = self.locks.get(key)
            if lock is None:
                if len(self.locks) > 10000:
                    self.locks.clear()
                lock = self.locks[key] = threading.Lock()
            return lock
