import time
from contextlib import contextmanager


def printa(cls: object, msg: str):
    """stands for print-authored"""
    print(f"[{type(cls).__name__}] {msg}")


@contextmanager
def timeit(name: str = "elapsed"):
    """
    with timeit("routine"):
        routine()
    >> "routine: 0.5s"
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{name}: {end - start:.4f} s")
