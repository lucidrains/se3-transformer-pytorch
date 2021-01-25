import os
import sys
import time
import pickle
import gzip
import contextlib
from functools import wraps, lru_cache
from filelock import FileLock

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# default dtype context manager

@contextlib.contextmanager
def torch_default_dtype(dtype):
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(prev_dtype)

# benchmark tool

def benchmark(fn):
    def inner(*args, **kwargs):
        start = time.time()
        res = fn(*args, **kwargs)
        diff = time.time() - start
        return diff, res
    return inner

# caching functions

def cache(cache, key_fn):
    def cache_inner(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            key_name = key_fn(*args, **kwargs)
            if key_name in cache:
                return cache[key_name]
            res = fn(*args, **kwargs)
            cache[key_name] = res
            return res

        return inner
    return cache_inner

# cache in directory

def cached_dir(dirname, maxsize=128):
    '''
    Cache a function with a directory

    :param dirname: the directory path
    :param maxsize: maximum size of the RAM cache (there is no limit for the directory cache)
    '''
    def decorator(func):

        @lru_cache(maxsize=maxsize)
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                os.makedirs(dirname)
            except FileExistsError:
                pass

            indexfile = os.path.join(dirname, "index.pkl")
            lock = FileLock(os.path.join(dirname, "mutex"))

            with lock:
                try:
                    with open(indexfile, "rb") as file:
                        index = pickle.load(file)
                except FileNotFoundError:
                    index = {}

                key = (args, frozenset(kwargs), func.__defaults__)

                try:
                    filename = index[key]
                except KeyError:
                    index[key] = filename = f"{len(index)}.pkl.gz"
                    with open(indexfile, "wb") as file:
                        pickle.dump(index, file)

            filepath = os.path.join(dirname, filename)

            try:
                with lock:
                    with gzip.open(filepath, "rb") as file:
                        result = pickle.load(file)
            except FileNotFoundError:
                print(f"compute {filename}... ", end="", flush = True)
                result = func(*args, **kwargs)
                print(f"save {filename}... ", end="", flush = True)

                with lock:
                    with gzip.open(filepath, "wb") as file:
                        pickle.dump(result, file)

                print("done")

            return result
        return wrapper
    return decorator
