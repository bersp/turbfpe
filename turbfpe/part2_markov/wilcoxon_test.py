from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .markov_auxiliar_functions import (
    calc_incs_3tau,
    compute_indep_incs_non_square_data,
    compute_indep_incs,
    distribution,
)

SQRT_2_DIV_PI = np.sqrt(2 / np.pi)


def compute_wilcoxon_test(data, fs, nbins, taylor_hyp_vel, indep_scale, end_scale, n_interv_sec=1):

    to_us = fs / taylor_hyp_vel # convert to unit of samples

    # number of statistically independent intervals
    indep_scale_us = round(indep_scale*to_us)
    n_interv = data.shape[1] // indep_scale_us - 1

    # calculate delta_arr_us (in unit of samples)
    n_increments = 10 * nbins
    start_scale = taylor_hyp_vel / fs  # this is 1 in unit of samples
    delta_arr_us = (
        np.logspace(
            np.log10(start_scale),
            np.log10(end_scale),
            n_increments,
        )
    ) * to_us
    delta_arr_us = np.unique(np.floor(delta_arr_us).astype(int))
    delta_arr_us = delta_arr_us[delta_arr_us != 0]


    # calculate where starts the independent intervals
    start_interv_idx = np.arange(0, indep_scale_us * n_interv, indep_scale_us)

    
    if n_interv_sec >= indep_scale_us:
        raise ValueError("n_inter_sec shoud be smallen than indep_scale_us")

    start_interv_sec_list = [
        start_interv_idx + n
        for n in range(0, n_interv_sec, 1)
        # for n in range(0, indep_scale_us, indep_scale_us // n_interv_sec)
    ]

    wt_arr = np.zeros((len(start_interv_sec_list), delta_arr_us.size))
    for ii, delta in enumerate(tqdm(delta_arr_us)):
        for jj, start_interv_sec in enumerate(start_interv_sec_list):
            # do the wilcoxon test for the delta selected
            # at different independent intervals.

            inc0, inc1, inc2 = compute_indep_incs(
                data, start_interv_sec, delta
            )

            # bins1
            count1, bins1_tmp = np.histogram(inc1, bins=nbins)
            bins1_width = bins1_tmp[1] - bins1_tmp[0]

            bins1 = [
                (bins1_tmp[i], bins1_tmp[i] + bins1_width) for i in range(count1.size)
            ]

            # bin0 (only one, i.e. idx_c0)
            # idx_c0 = np.abs(inc0) < bins1_width / 2
            idx_c0 = np.abs(inc0) < 2*bins1_width

            # mean of the wilcoxon test stats over all bins1
            tmp = wilcoxon_test_multiple_bins(inc1, inc2, bins1, idx_c0)
            wt_arr[jj, ii] = tmp

    return delta_arr_us, wt_arr


def wilcoxon_test_2samp(p1, p2):
    """
    Test if p(df)1 it is statistically distributed as p(df)2.
    """
    m = p1.size
    n = p2.size

    sp1 = np.sort(p1)
    sp2 = np.sort(p2)

    # Q = np.sum(sp2[:, np.newaxis] > sp1) is faster but memory inefficient
    Q = 0
    for val2 in sp2:
        Q += np.sum(val2 > sp1)

    Q_mean = n * m / 2
    Q_sigma = np.sqrt(n * m * (n + m + 1) / 12)
    T = np.abs(Q - Q_mean) / (Q_sigma * SQRT_2_DIV_PI)

    return T


def wilcoxon_test_multiple_bins(inc1, inc2, bins1, idx_c0):
    T_list = []
    for b1 in bins1:
        # - P(u_2|u_1) i.e. inc2 only where inc1 belongs to the nth bin
        idx_c1 = (inc1 > b1[0]) & (inc1 < b1[1])
        inc2_c1 = inc2[idx_c1]

        # - P(u_2|u_1,u_0=0)
        inc2_c1_c0 = inc2[idx_c1 & idx_c0]

        if inc2_c1.size > 40_000:  # we don't need that much data
            inc2_c1 = np.random.choice(inc2_c1, 40_000, replace=False)
        elif (
            inc2_c1.size < 30
        ):  # we need that much data (to be sure that we have enough data to calculate mean and have a Gaussian behaviour)
            continue  # skip this bin

        if inc2_c1_c0.size > 20_000:
            inc2_c1_c0 = np.random.choice(inc2_c1_c0, 20_000, replace=False)
        elif inc2_c1_c0.size < 30:
            continue

        # - test if P(u_2|u_1,u_0=0) is compatible with P(u_2|u_1)
        T = wilcoxon_test_2samp(inc2_c1, inc2_c1_c0)
        T_list.append(T)
    return np.mean(T_list)


def plot_wilcoxon_test(data, delta_arr_us, wt_arr, markov_scale_us):
    # TODO: Improve plot and add options to see this in different units
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.axhline(1, c="k")
    ax1.axvline(markov_scale_us, c="C3", ls="--", lw=2, label=r"$\Delta_\mathrm{EM}$")
    ax1.set_xlabel(r"$\Delta~\mathrm{[samp]}$")
    ax1.set_ylabel(r"$\mathrm{Standardized}$" + "\n" + r"$\mathrm{Wilcoxon-Test}$")

    for i, wt in enumerate(wt_arr):
        ax1.loglog(delta_arr_us, wt, ".-k", alpha=0.2)


    wt_arr_mean = np.nanmean(wt_arr, axis=0)
    ax1.loglog(delta_arr_us, wt_arr_mean, ".-", color="#C95E61", lw=2)

    ax1.legend()

    _plot_pdf_on_markov_scale(ax2, data, markov_scale_us, u0=0)


def _plot_pdf_on_markov_scale(ax, data, markov_scale_us, u0):
    incs0, incs1, incs2 = calc_incs_3tau(
        data, 3 * markov_scale_us, 2 * markov_scale_us, markov_scale_us
    )

    P_2I1, *_, bin_x_edges, bin_y_edges, dx, dy, _, counts1 = distribution(
        incs2, incs1, bins=150
    )

    P_2I1[:, counts1 < 400] = np.nan
    P_2I1 = np.clip(P_2I1, None, 5 * np.nanstd(P_2I1))

    x = bin_x_edges[:-1] + dx
    y = bin_y_edges[:-1] + dy
    X, Y = np.meshgrid(x, y)

    levels = (
        np.round(
            np.logspace(
                np.log10(0.01 * np.nanmax(P_2I1) * 1e6),
                np.log10(0.90 * np.nanmax(P_2I1) * 1e6),
                10,
            )
        )
        / 1e6
    )

    ax.set_xlabel(r"$u_{\tau_1=2\Delta_{EM}}$")
    ax.set_ylabel(r"$u_{\tau_2=\Delta_{EM}}$")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.6)

    ax.contour(X, Y, P_2I1, levels=levels, cmap="gray_r", linewidths=2)

    # -- P(u_2|u_1,u_0)
    idx = (incs0 > u0 - dx / 2) & (incs0 < u0 + dx / 2)
    incs2I0, incs1I0 = incs2[idx], incs1[idx]

    P_2I1I0, *_, counts1I0 = distribution(
        incs2I0, incs1I0, bins=[bin_x_edges, bin_y_edges]
    )
    P_2I1I0[:, (counts1I0 < 80)] = np.nan
    P_2I1I0 = np.clip(P_2I1I0, None, 5 * np.nanstd(P_2I1I0))

    ax.contour(X, Y, P_2I1I0, levels=levels, cmap="Reds", linewidths=2)

    # legend
    legend_elements = [
        Line2D([0], [0], color="k", lw=3, label="$P(u_2|u_1)$"),
        Line2D([0], [0], color="C3", lw=3, label=f"$P(u_2|u_1|u_0={u0})$"),
    ]
    ax.legend(handles=legend_elements)
