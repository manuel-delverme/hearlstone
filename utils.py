import pickle
import sys
import os
from functools import lru_cache
import hashlib

def to_tuples(list_of_lists):
    tuple_of_tuples = []
    for item in list_of_lists:
        if isinstance(item, list):
            item = to_tuples(item)
        tuple_of_tuples.append(item)
    return  tuple(tuple_of_tuples)

def disk_cache(f):
    @lru_cache(maxsize=1024)
    def wrapper(*args, **kwargs):
        fid = f.__name__
        cache_file = "cache/{}".format(fid)
        if args:
            if not os.path.exists(cache_file):
                os.makedirs(cache_file)
            fid = fid + "/" + "::".join(str(arg) for arg in args).replace("/", "_")
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
