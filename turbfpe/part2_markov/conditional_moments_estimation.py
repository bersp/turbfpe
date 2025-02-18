import numpy as np
from tqdm import tqdm

from ..utils.storing_clases import (
    ConditionalMoments,
    ConditionalMomentsGroup,
    DensityFunctions,
    DensityFunctionsGroup,
)
from .markov_auxiliar_functions import (
    calc_incs_2tau,
    compute_mean_values_per_bin,
    distribution,
)


def compute_conditional_moments_estimation(
    data,
    fs,
    highest_freq,
    markov_scale_us,
    nbins,
    int_scale,
    taylor_scale,
    n_scale_steps,
    taylor_hyp_vel,
):
    steps_con_moment_us = np.round(
        np.linspace(markov_scale_us, 2 * markov_scale_us, 5)
    ).astype(int)

    smallest_scale = (
        (1 / highest_freq) + steps_con_moment_us[-1] / fs
    ) * taylor_hyp_vel

    scales_tmp = np.unique(
        np.logspace(
            np.log10(max(smallest_scale, taylor_scale) * 1e6),
            np.log10(int_scale * 1e6),
            n_scale_steps,
        ).astype(int)
        * 1e-6
    )

    scales_us = np.unique(np.round(scales_tmp / taylor_hyp_vel * fs)).astype(int)
    scales_us = scales_us[scales_us > steps_con_moment_us[-1]]
    scales = scales_us / fs * taylor_hyp_vel

    cond_moments_group = ConditionalMomentsGroup()
    density_funcs_group = DensityFunctionsGroup()
    for scale_idx in tqdm(range(scales.size)):
        scale_us = scales_us[scale_idx]

        # Initialize variables for storing computation results
        M11 = np.full((nbins, steps_con_moment_us.size), np.nan)
        M21 = np.full((nbins, steps_con_moment_us.size), np.nan)
        M31 = np.full((nbins, steps_con_moment_us.size), np.nan)
        M41 = np.full((nbins, steps_con_moment_us.size), np.nan)
        M1_err = np.full((nbins, steps_con_moment_us.size), np.nan)
        M2_err = np.full((nbins, steps_con_moment_us.size), np.nan)

        for jj in range(0, steps_con_moment_us.size)[::-1]:
            inc0, inc1 = calc_incs_2tau(
                data, tau0=scale_us, tau1=scale_us - steps_con_moment_us[jj]
            )

            (
                P_1I0,
                P_0I1,
                P_1n0,
                P_1,
                P_0,
                bin1_edges,
                bin0_edges,
                bin1_width,
                bin0_width,
                counts1,
                counts0,
            ) = distribution(
                x_data=inc1,
                y_data=inc0,
                num_bin=nbins,
            )

            mp_bin0 = compute_mean_values_per_bin(inc0, counts0, bin0_edges)
            mp_bin1 = compute_mean_values_per_bin(inc1, counts1, bin1_edges)
            mean_per_bin_1I0 = mp_bin1[:, np.newaxis] - mp_bin0[np.newaxis, :]

            # Compute moments
            dd = bin1_width
            M11[:, jj] = dd * np.nansum(mean_per_bin_1I0 * P_1I0, axis=0)
            M21[:, jj] = dd * np.nansum((mean_per_bin_1I0**2) * P_1I0, axis=0)
            M31[:, jj] = dd * np.nansum((mean_per_bin_1I0**3) * P_1I0, axis=0)
            M41[:, jj] = dd * np.nansum((mean_per_bin_1I0**4) * P_1I0, axis=0)

            # Compute errors of the moments
            M1_err[:, jj] = np.sqrt(np.abs((M21[:, jj] - M11[:, jj] ** 2) / counts0))
            M2_err[:, jj] = np.sqrt(np.abs((M41[:, jj] - M21[:, jj] ** 2) / counts0))

        cond_moments = ConditionalMoments(
            scale=scales[scale_idx],
            scale_us=scale_us,
            scale_short_us=scale_us - steps_con_moment_us[0],
            M11=M11,
            M21=M21,
            M31=M31,
            M41=M41,
            M1_err=M1_err,
            M2_err=M2_err,
        )
        cond_moments_group.add(cond_moments)

        density_funcs = DensityFunctions(
            scale=scales[scale_idx],
            scale_us=scale_us,
            scale_short_us=scale_us - steps_con_moment_us[0],
            P_1=P_1,
            P_0=P_0,
            P_1I0=P_1I0,
            P_0I1=P_0I1,
            P_1n0=P_1n0,
            bin1_edges=bin1_edges,
            bin0_edges=bin0_edges,
            bin1_width=bin1_width,
            bin0_width=bin0_width,
            counts1=counts1,
            counts0=counts0,
            mean_per_bin1=mp_bin1,
            mean_per_bin0=mp_bin0,
            inc1=inc1,
            inc0=inc0,
        )
        density_funcs_group.add(density_funcs)

    return cond_moments_group, density_funcs_group
