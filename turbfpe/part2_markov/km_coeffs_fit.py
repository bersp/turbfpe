from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from ..utils.storing_clases import KMCoeffs
from .markov_auxiliar_functions import get_Di_for_all_incs_and_scales


def model_D1(u, r, a11, c11, b11):
    """D1(u, r) = [a11 * r^b11 + c11] * u"""
    d11 = a11 * (r**b11) + c11
    return d11 * u


def model_D2(u, r, a20, c20, a21, c21, a22, c22, b20, b21, b22):
    """
    D2(u, r) = [a20 * r^b20 + c20]
             + [a21 * r^b21 + c21] * u
             + [a22 * r^b22 + c22] * u^2
    """
    d20 = a20 * (r**b20) + c20
    d21 = a21 * (r**b21) + c21
    d22 = a22 * (r**b22) + c22
    return d20 + d21 * u + d22 * (u**2)


def fit_D1(
    u_vals: np.ndarray,
    r_vals: np.ndarray,
    z_vals: np.ndarray,
    z_err_vals: np.ndarray,
    initial_guess=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit: D1(u, r) = [a11 * r^b11 + c11] * u
    Returns (popt, pcov)
    """
    u_data = u_vals.ravel()
    r_data = r_vals.ravel()
    z_data = z_vals.ravel()
    z_err_data = z_err_vals.ravel()

    valid_idxs = ~np.isnan(u_data)
    u_data = u_data[valid_idxs]
    r_data = r_data[valid_idxs]
    z_data = z_data[valid_idxs]
    z_err_data = z_err_data[valid_idxs]

    def _model(data, a11, c11, b11):
        uu, rr = data
        return model_D1(uu, rr, a11, c11, b11)

    if initial_guess is None:
        initial_guess = [0, 0, -1]  # a11, c11, b11

    bounds = ([-2, -2, -np.inf], [2, 2, 0])

    popt, pcov = curve_fit(
        _model, (u_data, r_data), z_data, p0=initial_guess, bounds=bounds
    )
    return popt, pcov


def fit_D2(
    u_vals: np.ndarray,
    r_vals: np.ndarray,
    z_vals: np.ndarray,
    z_err_vals: np.ndarray,
    initial_guess=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit: D2(u, r) = [a20*r^b20 + c20]
                  + [a21*r^b21 + c21]*u
                  + [a22*r^b22 + c22]*u^2
    Returns (popt, pcov)
    """
    u_data = u_vals.ravel()
    r_data = r_vals.ravel()
    z_data = z_vals.ravel()
    z_err_data = z_err_vals.ravel()

    valid_idxs = ~np.isnan(u_data)
    u_data = u_data[valid_idxs]
    r_data = r_data[valid_idxs]
    z_data = z_data[valid_idxs]
    z_err_data = z_err_data[valid_idxs]

    def _model(data, a20, c20, a21, c21, a22, c22, b20, b21, b22):
        uu, rr = data
        return model_D2(uu, rr, a20, c20, a21, c21, a22, c22, b20, b21, b22)

    if initial_guess is None:
        initial_guess = [0, 0, 0, 0, 0, 0, 0, 0, -1e-4]

    bounds = (
        [-1, -1, -1, -1, -1, -1, 0, -np.inf, -np.inf],
        [1, 1, 1, 1, 1, 1, np.inf, np.inf, 0],
    )

    # Overwrite errors as ones if needed
    # z_err_data = np.ones_like(z_err_data)

    popt, pcov = curve_fit(
        _model,
        (u_data, r_data),
        z_data,
        p0=initial_guess,
        sigma=z_err_data,
        bounds=bounds,
    )
    return popt, pcov


def compute_km_coeffs_fit(
    density_funcs_group, km_coeffs_est_group, nbins: int, taylor_scale: float
) -> tuple[KMCoeffs, KMCoeffs]:
    """
    1) Prepares data for D1_opti, D2_opti
    2) Prepares data for D1, D2
    3) Fits D1 and D2 in both scenarios
    4) Returns km_coeffs and km_coeffs_noopt
    """

    # --- 1) Data for optimized fits ---
    u_opt, r_opt, D1_opt, D1_err_opt, D2_opt, D2_err_opt = (
        get_Di_for_all_incs_and_scales(
            density_funcs_group,
            km_coeffs_est_group,
            nbins,
            taylor_scale,
            use_Di_opti=True,
        )
    )

    # --- 2) Data for non-optimized fits ---
    u_noopt, r_noopt, D1_noopt, D1_err_noopt, D2_noopt, D2_err_noopt = (
        get_Di_for_all_incs_and_scales(
            density_funcs_group,
            km_coeffs_est_group,
            nbins,
            taylor_scale,
            use_Di_opti=False,
        )
    )

    # --- 3) Fit both sets (D1, D2) for optimized
    popt_D1_opt, pcov_D1_opt = fit_D1(u_opt, r_opt, D1_opt, D1_err_opt)
    popt_D2_opt, pcov_D2_opt = fit_D2(u_opt, r_opt, D2_opt, D2_err_opt)

    # --- 3) Fit both sets (D1, D2) for non-optimized
    popt_D1_noopt, pcov_D1_noopt = fit_D1(u_noopt, r_noopt, D1_noopt, D1_err_noopt)
    popt_D2_noopt, pcov_D2_noopt = fit_D2(u_noopt, r_noopt, D2_noopt, D2_err_noopt)

    # --- 4) Extract best-fit parameters ---

    # Compute 95% intervals for the "optimized" fits
    D1_conf_opt, D2_conf_opt = None, None
    if pcov_D1_opt is not None:
        err_D1_opt = 1.96 * np.sqrt(np.diag(pcov_D1_opt))
        D1_conf_opt = (popt_D1_opt - err_D1_opt, popt_D1_opt + err_D1_opt)
    if pcov_D2_opt is not None:
        err_D2_opt = 1.96 * np.sqrt(np.diag(pcov_D2_opt))
        D2_conf_opt = (popt_D2_opt - err_D2_opt, popt_D2_opt + err_D2_opt)

    # Create the km_coeffs object for the optimized fit
    km_coeffs = _gen_km_coeffs_object(
        popt_D1_opt, popt_D2_opt, D1_conf_opt, D2_conf_opt
    )

    # Compute 95% intervals for the "non-optimized" fits
    D1_conf_noopt, D2_conf_noopt = None, None
    if pcov_D1_noopt is not None:
        err_D1_no = 1.96 * np.sqrt(np.diag(pcov_D1_noopt))
        D1_conf_noopt = (popt_D1_noopt - err_D1_no, popt_D1_noopt + err_D1_no)
    if pcov_D2_noopt is not None:
        err_D2_no = 1.96 * np.sqrt(np.diag(pcov_D2_noopt))
        D2_conf_noopt = (popt_D2_noopt - err_D2_no, popt_D2_noopt + err_D2_no)

    # Create the km_coeffs object for the non-optimized fit
    km_coeffs_noopt = _gen_km_coeffs_object(
        popt_D1_noopt, popt_D2_noopt, D1_conf_noopt, D2_conf_noopt
    )

    return km_coeffs, km_coeffs_noopt


def _gen_km_coeffs_object(popt_d1, popt_d2, conf_d1, conf_d2) -> KMCoeffs:
    """
    Create a km_coeffs instance from the parameter arrays and
    their confidence intervals (each param has [lower, upper]).
    We reorder them to match the km_coeffs layout:
        a11, b11, c11,
        a20, b20, c20, a21, b21, c21, a22, b22, c22
    """
    # popt_d1 => a11=0, c11=1, b11=2
    a11_val = popt_d1[0]
    b11_val = popt_d1[2]
    c11_val = popt_d1[1]

    # popt_d2 => a20=0, c20=1, a21=2, c21=3, a22=4, c22=5, b20=6, b21=7, b22=8
    a20_val = popt_d2[0]
    b20_val = popt_d2[6]
    c20_val = popt_d2[1]

    a21_val = popt_d2[2]
    b21_val = popt_d2[7]
    c21_val = popt_d2[3]

    a22_val = popt_d2[4]
    b22_val = popt_d2[8]
    c22_val = popt_d2[5]

    # If confidence intervals exist, reorder them the same way
    if conf_d1 is not None:
        # conf_d1[0] => lower bounds, conf_d1[1] => upper bounds
        # indices: 0->a11, 1->c11, 2->b11
        a11_conf = np.array([conf_d1[0][0], conf_d1[1][0]], dtype=np.float64)
        b11_conf = np.array([conf_d1[0][2], conf_d1[1][2]], dtype=np.float64)
        c11_conf = np.array([conf_d1[0][1], conf_d1[1][1]], dtype=np.float64)
    else:
        # fallback if we have no confidence data
        a11_conf = np.array([np.nan, np.nan], dtype=np.float64)
        b11_conf = np.array([np.nan, np.nan], dtype=np.float64)
        c11_conf = np.array([np.nan, np.nan], dtype=np.float64)

    if conf_d2 is not None:
        a20_conf = np.array([conf_d2[0][0], conf_d2[1][0]], dtype=np.float64)
        b20_conf = np.array([conf_d2[0][6], conf_d2[1][6]], dtype=np.float64)
        c20_conf = np.array([conf_d2[0][1], conf_d2[1][1]], dtype=np.float64)

        a21_conf = np.array([conf_d2[0][2], conf_d2[1][2]], dtype=np.float64)
        b21_conf = np.array([conf_d2[0][7], conf_d2[1][7]], dtype=np.float64)
        c21_conf = np.array([conf_d2[0][3], conf_d2[1][3]], dtype=np.float64)

        a22_conf = np.array([conf_d2[0][4], conf_d2[1][4]], dtype=np.float64)
        b22_conf = np.array([conf_d2[0][8], conf_d2[1][8]], dtype=np.float64)
        c22_conf = np.array([conf_d2[0][5], conf_d2[1][5]], dtype=np.float64)
    else:
        a20_conf = np.array([np.nan, np.nan], dtype=np.float64)
        b20_conf = np.array([np.nan, np.nan], dtype=np.float64)
        c20_conf = np.array([np.nan, np.nan], dtype=np.float64)
        a21_conf = np.array([np.nan, np.nan], dtype=np.float64)
        b21_conf = np.array([np.nan, np.nan], dtype=np.float64)
        c21_conf = np.array([np.nan, np.nan], dtype=np.float64)
        a22_conf = np.array([np.nan, np.nan], dtype=np.float64)
        b22_conf = np.array([np.nan, np.nan], dtype=np.float64)
        c22_conf = np.array([np.nan, np.nan], dtype=np.float64)

    return KMCoeffs(
        a11=a11_val,
        b11=b11_val,
        c11=c11_val,
        a20=a20_val,
        b20=b20_val,
        c20=c20_val,
        a21=a21_val,
        b21=b21_val,
        c21=c21_val,
        a22=a22_val,
        b22=b22_val,
        c22=c22_val,
        a11_conf=a11_conf,
        b11_conf=b11_conf,
        c11_conf=c11_conf,
        a20_conf=a20_conf,
        b20_conf=b20_conf,
        c20_conf=c20_conf,
        a21_conf=a21_conf,
        b21_conf=b21_conf,
        c21_conf=c21_conf,
        a22_conf=a22_conf,
        b22_conf=b22_conf,
        c22_conf=c22_conf,
    )
