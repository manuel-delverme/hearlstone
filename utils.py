import pickle
import sys
import os
from functools import lru_cache


def disk_cache(f):
    @lru_cache(maxsize=1024)
    def wrapper(*args, **kwargs):
        fid = f.__name__
        cache_file = "cache/{}".format(fid)
        if args:
            if not os.path.exists(cache_file):
                os.makedirs(cache_file)
            fid = fid + "/" + "::".join(str(arg) for arg in args)
            cache_file = "cache/{}".format(fid)
        cache_file += ".pkl"
        try:
            with open(cache_file, "rb") as fin:
                retr = pickle.load(fin)
        except FileNotFoundError:
            retr = f(*args, **kwargs)
            with open(cache_file, "wb") as fout:
                pickle.dump(retr, fout)
        return retr

    return wrapper
