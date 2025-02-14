import logging

import numpy as np

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s [%(levelname)s] [%(filename)s %(funcName)20s] %(message)s"),
)


def get_pdf(arr, bins):
    y, bins = np.histogram(arr, bins=bins, density=False)
    y = y.astype(float)
    y = y / np.sum(y * np.diff(bins))

    offset = (bins[1] - bins[0]) / 2
    x = bins[:-1] + offset

    return x, y


def logspace_moving_average(v, nbins):
    bins = np.unique(np.logspace(0, np.log10(v.size), nbins, endpoint=True).astype(int))
    out = np.zeros(len(bins) - 1)
    for i in range(1, len(bins) - 1):
        out[i] = np.mean(v[bins[i - 1] : bins[i + 1]])
    return out[1:]
