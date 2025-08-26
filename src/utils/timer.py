import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    t0 = time.time()
    yield
    print(f"[TIMER] {name}: {time.time()-t0:.2f}s")
