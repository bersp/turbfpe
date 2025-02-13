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


def calc_incs_1tau(v, tau):
    """v.shape = (time, ...)"""

    tsize = v.shape[0]
    inc = v[tau:tsize] - v[0 : tsize - tau]
    return inc.flatten()


def calc_incs_2tau(v, tau0, tau1):
    """
    v.shape = (time, ...)
    tau0 > tau1
    """
    tsize = v.shape[0]

    d_01 = tau0 - tau1

    inc0 = v[tau0:tsize] - v[0 : tsize - tau0]
    inc1 = v[tau1 : tsize - d_01] - v[0 : tsize - tau0]

    inc0, inc1 = inc0.flatten(), inc1.flatten()

    return inc0, inc1


def calc_incs_3tau(v, tau0, tau1, tau2):
    """
    v.shape = (time, ...)
    tau0 > tau1 > tau2
    """
    tsize = v.shape[0]

    d_01 = tau0 - tau1
    d_02 = tau0 - tau2

    inc0 = v[tau0:tsize] - v[0 : tsize - tau0]
    inc1 = v[tau1 : tsize - d_01] - v[0 : tsize - tau0]
    inc2 = v[tau2 : tsize - d_02] - v[0 : tsize - tau0]

    inc0, inc1, inc2 = inc0.flatten(), inc1.flatten(), inc2.flatten()

    return inc0, inc1, inc2
