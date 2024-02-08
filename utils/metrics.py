import numpy as np


def cumavg(m):
    cumsum = np.cumsum(m)
    return cumsum / np.arange(1, cumsum.size + 1)


def RSE(pred, true):
    if np.var(true)>0:
        return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
            np.sum((true - true.mean()) ** 2)
        )
    else:
        return np.sqrt(np.sum((true - pred) ** 2))


def CORR(pred, true):
    mean_x = np.mean(pred)
    mean_y = np.mean(true)
    numerator = np.sum((pred - mean_x) * (true - mean_y))
    if np.var(true) >0:
        denominator = np.sqrt(np.sum((pred - mean_x) ** 2) * np.sum((true - mean_y) ** 2))
    else:
        denominator = np.sqrt(np.sum((pred - mean_x) ** 2))
    r = numerator / denominator
    return r


def metric(pred, true):
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    return rse, corr
