import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..utils.mpl_utils import *
from .markov_auxiliar_functions import (
    calculate_indep_incr_non_square_data,
    calculate_indep_incr_square_data,
    wilcoxon_test_multiple_bins,
)


def compute_wilcoxon_test(
    data, fs, nbins, taylor_hyp_vel, indep_scale, n_interv_sec=1, output_filename=None
):
    # number of statistically independent intervals
    indep_scale_us = round(fs * indep_scale / taylor_hyp_vel)
    n_interv = data.shape[1] // indep_scale_us - 1

    # calculate delta_arr
    n_increments = 10 * nbins
    taylor_scale = 1 / fs
    delta_arr = (
        np.logspace(
            np.log10(taylor_scale * 1e6),
            np.log10((indep_scale - taylor_scale / 2.5) * 1e6),
            # If there is some problem with this, you can try with
            # replacing the second value of the logspace with
            # np.log10((indep_scale-taylor_scale/2.5)*1e6)
            n_increments,
        )
        * 1e-6
    )
    delta_arr = np.unique(np.floor(delta_arr * fs / taylor_hyp_vel).astype(int))
    delta_arr = delta_arr[delta_arr != 0]

    # calculate where starts the independent intervals
    start_interv_idx = np.arange(0, indep_scale_us * n_interv, indep_scale_us)

    # if all data in the array have the same effective scale
    data_is_square = np.sum(np.isnan(data[:, -1])) == 0

    start_interv_sec_list = [
        start_interv_idx + n
        for n in range(0, indep_scale_us, indep_scale_us // n_interv_sec)
    ]

    wt_arr = np.zeros((len(start_interv_sec_list), delta_arr.size))
    for ii, delta in enumerate(tqdm(delta_arr)):
        for jj, start_interv_sec in enumerate(start_interv_sec_list):
            # do the wilcoxon test for the delta selected
            # at different independent intervals.

            if data_is_square:
                inc0, inc1, inc2 = calculate_indep_incr_square_data(
                    data, start_interv_sec, delta
                )
            else:
                inc0, inc1, inc2 = calculate_indep_incr_non_square_data(
                    data, start_interv_sec, delta
                )

            # bins1
            count1, bins1_tmp = np.histogram(inc1, bins=nbins)
            bins1_width = bins1_tmp[1] - bins1_tmp[0]

            bins1 = [
                (bins1_tmp[i], bins1_tmp[i] + bins1_width) for i in range(count1.size)
            ]

            # bin0 (only one, i.e. idx_c0)
            idx_c0 = np.abs(inc0) < bins1_width / 2

            # mean of the wilcoxon test stats over all bins1
            tmp = wilcoxon_test_multiple_bins(inc1, inc2, bins1, idx_c0)
            wt_arr[jj, ii] = tmp

    if output_filename:
        np.save(output_filename, np.vstack((delta_arr, wt_arr)))


def plot_wilcoxon_test(filename, EM_scale):
    # TODO: Improve plot and add options to see this in different units
    delta_arr, *wt_arr = np.load(filename)
    wt_arr = np.mean(wt_arr, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    for ax in (ax1, ax2):
        ax.axhline(1, c="k")
        ax.axvline(EM_scale, c="k", ls="--")
        ax.set_xlabel(r"$\Delta$ [samp]")
        ax.set_ylabel("Wilcoxon-Test")
    ax1.semilogy(delta_arr, wt_arr, "o")
    ax2.loglog(delta_arr, wt_arr, "o")
    plt.show()
