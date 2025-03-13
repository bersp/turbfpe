import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from ..utils.logger_setup import logger
from .markov_auxiliar_functions import (
    calc_incs_2tau,
    compute_mean_values_per_bin,
    distribution,
)


def compute_km_coeffs_estimation_stp_opti_one_scale(
    data,
    km_coeffs_est,
    fs,
    nbins,
    tol,
    taylor_scale,
    taylor_hyp_vel,
):
    D1 = km_coeffs_est.D1
    D2 = km_coeffs_est.D2
    scale_short_us = km_coeffs_est.scale_short_us
    scale_us = km_coeffs_est.scale_us
    valid_idxs = km_coeffs_est.valid_idxs

    inc0, inc1 = calc_incs_2tau(data, tau0=scale_us, tau1=scale_short_us)

    P_1I0, *_, bin1_edges, bin0_edges, _, _, counts1, counts0 = distribution(
        x_data=inc1,
        y_data=inc0,
        bins=nbins,
    )
    mean_per_bin0 = compute_mean_values_per_bin(inc0, counts0, bin0_edges)
    mean_per_bin1 = compute_mean_values_per_bin(inc1, counts1, bin1_edges)

    # Prepare initial guess init_D1_D2_values (concatenate D1 and abs(D2)
    init_D1_D2_values = np.concatenate([D1[valid_idxs], np.abs(D2[valid_idxs])])

    # Upper and lower bounds (percent of the D1, D2 range)
    D1_valid = D1[valid_idxs]
    D2_valid = D2[valid_idxs]
    ub = np.concatenate([
        D1_valid + tol * (D1_valid.max() - D1_valid.min()),
        np.abs(D2_valid) + tol * (D2_valid.max() - D2_valid.min()),
    ])
    lb = np.concatenate([
        D1_valid - tol * (D1_valid.max() - D1_valid.min()),
        np.zeros_like(D2_valid),
    ])

    # Define the objective function
    def objective_func(D1_D2_values):
        return calc_divergence(
            D1_D2_values,
            mean_per_bin0,
            mean_per_bin1,
            P_1I0,
            valid_idxs,
            taylor_scale,
            taylor_hyp_vel,
            fs,
            scale_short_us,
            scale_us,
        )

    bounds = [(float(lb[i]), float(ub[i])) for i in range(len(init_D1_D2_values))]

    def callback(xk):
        print(f"Iteration: {callback.iter}, x: {xk}, f(x): {objective(xk)}")
        callback.iter += 1

    opt_result = minimize(
        fun=objective_func,
        x0=init_D1_D2_values,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300, "gtol": 1e-8},
        callback=callback,
    )

    D1_D2_values_opt = opt_result.x

    # Store results
    n_valid = sum(valid_idxs)
    km_coeffs_est.set_D1_opti(D1_D2_values_opt[:n_valid])
    km_coeffs_est.set_D2_opti(D1_D2_values_opt[n_valid : 2 * n_valid])
    km_coeffs_est.set_valid_idxs(valid_idxs)

    return km_coeffs_est


def compute_km_estimation_stp_opti(
    data,
    km_coeffs_est_group,
    fs,
    nbins,
    tol,
    taylor_scale,
    taylor_hyp_vel,
):
    km_coeffs_est_group = km_coeffs_est_group.copy()

    logger.info("Short time propagator optimization...")
    for scale_idx in tqdm(
        range(len(km_coeffs_est_group)),
        desc="# scales",
        bar_format=r"{desc}: |{bar}{r_bar}",
    ):
        km_coeffs_est = km_coeffs_est_group[scale_idx]

        km_coeffs_est = compute_km_coeffs_estimation_stp_opti_one_scale(
            data,
            km_coeffs_est,
            fs,
            nbins,
            tol,
            taylor_scale,
            taylor_hyp_vel,
        )
        km_coeffs_est_group[scale_idx] = km_coeffs_est

    return km_coeffs_est_group


def short_time_prop(
    y,
    x,
    scale_short_us,
    scale_us,
    D1,
    D2,
    taylor_scale,
    taylor_hyp_vel,
    fs,
):
    """
    This function looks somewhat unusual, but it is just an
    efficient way (making heavy use of NumPy broadcasting) to write
    the propagator $p_{stp}$ (Eq. 55 of https://doi.org/10.1063/5.0107974).

    This is not strictly necessary, but it significantly speeds up the optimization.
    """
    taylor_length_us = taylor_scale / taylor_hyp_vel * fs
    delta = abs(scale_us - scale_short_us) / taylor_length_us

    D2_delta = D2 * delta

    diff = x[:, None] - y[None, :] - (D1 * delta)[None, :]
    mask_good = D2_delta > 0

    P = np.full((len(x), len(y)), 1e-15, dtype=float)

    D2_delta_good = D2_delta[mask_good]
    denom2_good = 2.0 * np.sqrt(np.pi * D2_delta_good)

    small_mask = denom2_good < 1e-15

    diff_good = diff[:, mask_good]
    exponent_good = -(diff_good**2) / (4.0 * D2_delta_good)

    denom2_good_no_small = denom2_good.copy()
    denom2_good_no_small[small_mask] = np.inf

    P_vals = (1.0 / denom2_good_no_small) * np.exp(exponent_good)

    P[:, mask_good] = P_vals

    P[:, mask_good][:, small_mask] = 1e-15

    P[P < 0] = 0.0

    return P


def calc_divergence(
    D1_D2_values,
    mean_per_bin0,
    mean_per_bin1,
    P_1I0,
    valid_idxs,
    taylor_scale,
    taylor_hyp_vel,
    fs,
    scale_short_us,
    scale_us,
):
    n_valid = len(D1_D2_values) // 2
    D1 = D1_D2_values[:n_valid]
    D2 = np.abs(D1_D2_values[n_valid:])

    # Compute short-time propagator
    P_n = short_time_prop(
        y=mean_per_bin0[valid_idxs],
        x=mean_per_bin1[valid_idxs],
        scale_short_us=scale_short_us,
        scale_us=scale_us,
        D1=D1,
        D2=D2,
        taylor_scale=taylor_scale,
        taylor_hyp_vel=taylor_hyp_vel,
        fs=fs,
    )  # shape => (len(x_in), len(y_in))

    P_1I0_in = P_1I0[np.ix_(valid_idxs, valid_idxs)]

    # Extract only valid elements
    mask = (P_n > 0) & (P_1I0_in > 0)
    Pn_masked = P_n[mask]
    P1I0_in_masked = P_1I0_in[mask]

    # Compute log values once
    log_Pn = np.log(Pn_masked)
    log_P1I0_in = np.log(P1I0_in_masked)

    # term_1 => (P_n + P_1I0_in) * (log(P_n) - log(P_1I0_in))^2
    sum_P = Pn_masked + P1I0_in_masked
    diff_log_sq = (log_Pn - log_P1I0_in) ** 2
    term_1 = sum_P * diff_log_sq

    # term_2 => (P_n + P_1I0_in) * (log(P_n)^2 + log(P_1I0_in)^2)
    log_sq_sum = log_Pn**2 + log_P1I0_in**2
    term_2 = sum_P * log_sq_sum

    # Accumulate sums
    d_M_1 = term_1.sum()
    d_M_2 = term_2.sum()

    # Avoid division by zero => large penalty
    if d_M_2 == 0.0:
        return 1e10

    d_M = d_M_1 / d_M_2
    return d_M
