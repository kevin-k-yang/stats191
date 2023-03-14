import pickle as pkl
import sys

import pandas as pd
import statsmodels.formula.api as smf


def read_pickle(file):
    with open(file, 'rb') as file:
        result = pkl.load(file)
    return result


def run_func(f, fp, *args, **kwargs):
    try:
        result = read_pickle(fp)
    except FileNotFoundError:
        result = f(*args, **kwargs)
        with open(fp, 'wb') as file:
            pkl.dump(result, file)
    return result


def run_ols(formula, data, f=sys.stdout):
    result = smf.ols(formula=formula, data=data).fit()
    print(result.summary(), file=f)
    return result


def load_csv(name):
    file = name + ".csv"
    cache_name = name + ".pkl"
    return run_func(pd.read_csv, cache_name, file)
