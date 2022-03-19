#!/usr/bin/env python3

from functools import wraps
import os.path

import glog as log
import numpy as np
import pandas as pd

import FullOuterSampler.datasets as datasets

CACHE_DIR = "./cache"

TOY_TABLES = {
    "A": pd.DataFrame({"x": [1, 2, 3]}),
    "B": pd.DataFrame({
        "x": [1, 2, 2, 2, 4],
        "y": [10, 20, 20, 30, 30],
        "z": [100, 100, 100, 100, 200],
    }),
    "C": pd.DataFrame({"y": [10, 20, 20, 40]}),
    "D": pd.DataFrame({"z": [100, 100, 200, 300]}),
}


def load(filename, description):
    # +@ add file name encoder
    save_path = os.path.join(CACHE_DIR, datasets.filename_encoder(filename))
    log.info(f"Loading cached {description} from {save_path}  -  encoded {filename}")
    return pd.read_feather(save_path)


def save_result(filename, subdir=None, description="result"):

    def decorator(func):

        @wraps(func)
        def wrapper(*fargs, **kwargs):
            os.makedirs(CACHE_DIR, exist_ok=True)
            # +@ use filename Encoder
            if subdir is not None:
                os.makedirs(os.path.join(CACHE_DIR, subdir), exist_ok=True)
                save_path = os.path.join(CACHE_DIR, subdir, datasets.filename_encoder(filename))
            else:
                save_path = os.path.join(CACHE_DIR, datasets.filename_encoder(filename))
            if os.path.exists(save_path):
                log.info(f"Loading cached {description} from {save_path} - encoded {filename}")
                ret = pd.read_feather(save_path)
            else:
                log.info(f"Creating {description}.")
                ret = func(*fargs, **kwargs)
                log.info(f"Saving {description} to {save_path}")
                ret.to_feather(save_path)
            return ret

        return wrapper

    return decorator


# +@ change parameter
def load_table(table, df_dict , **kwargs):

#     usecols = kwargs.get("usecols")
    # +@ load usecols
#     if usecols is not None:
#         usecols = datasets.get_use_column(dataset,table,usecols)
#     kwargs.update({"usecols": usecols})
#     if usecols is None:
    usecols = ["ALL"]

    #XXX error??
    #@save_result("{}-{}.df".format(table, "-".join(usecols)),
    #             description=f"dataframe of `{table}`")
    def work():
        print(table, kwargs)
        return df_dict[table]
#         return pd.read_csv(os.path.join(data_dir, f"{table}.csv"),
#                            escapechar="\\",
#                            low_memory=False,
#                            **kwargs)

    return work()

