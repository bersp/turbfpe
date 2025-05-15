import numpy as np
from scipy.optimize import curve_fit, minimize

from ..utils.logger_setup import logger
from ..utils.storing_clases import KMCoeffs
from .entropy_computation import get_entropies


def compute_km_coeffs_ift_opti(
    data,
    km_coeffs_stp_opti,
    scales_for_optimization,
    tol_D1,
    tol_D2,
    iter_max,
    fs,
    smallest_scale,
    largest_scale,
    scale_subsample_step_us,
    taylor_scale,
    taylor_hyp_vel,
    available_ram_gb,
):
    overlap_trajs_flag = 0  # don't use overlap trajs to optimize

    scales_dimless = scales_for_optimization / taylor_scale
    n_scales = scales_dimless.size
    d11 = km_coeffs_stp_opti.eval_d11(scales_dimless)
    d20 = km_coeffs_stp_opti.eval_d20(scales_dimless)
    d21 = km_coeffs_stp_opti.eval_d21(scales_dimless)
    d22 = km_coeffs_stp_opti.eval_d22(scales_dimless)
    x0 = np.concatenate([d11, d20, d21, d22])
    lower_bound = np.empty_like(x0)
    upper_bound = np.empty_like(x0)
    lower_bound[0:n_scales] = x0[0:n_scales] - np.abs(x0[0:n_scales] * tol_D1)
    upper_bound[0:n_scales] = x0[0:n_scales] + np.abs(x0[0:n_scales] * tol_D1)
    for i in range(1, 4):
        start = i * n_scales
        end = (i + 1) * n_scales
        lower_bound[start:end] = x0[start:end] - np.abs(x0[start:end] * tol_D2)
        upper_bound[start:end] = x0[start:end] + np.abs(x0[start:end] * tol_D2)
    upper_bound[0:n_scales] = np.minimum(upper_bound[0:n_scales], 0)
    lower_bound[n_scales : 2 * n_scales] = np.maximum(
        lower_bound[n_scales : 2 * n_scales], 0
    )
    lower_bound[3 * n_scales : 4 * n_scales] = np.maximum(
        lower_bound[3 * n_scales : 4 * n_scales], 0
    )
    _, optimization_history = ift_run_optimization(
        x0,
        lower_bound,
        upper_bound,
        iter_max,
        scales_dimless,
        n_scales,
        km_coeffs_stp_opti,
        largest_scale,
        smallest_scale,
        data,
        fs,
        scale_subsample_step_us,
        taylor_scale,
        taylor_hyp_vel,
        overlap_trajs_flag,
        available_ram_gb,
    )
    history_fval = np.array(optimization_history["error"])
    best_index = np.argmin(history_fval)
    best_x = optimization_history["x_iter"][best_index]
    popt_d11 = fit_d1j(scales_dimless, best_x[0:n_scales], km_coeffs_stp_opti)
    popt_d20, popt_d21, popt_d22 = fit_d2j(
        scales_dimless, best_x, n_scales, km_coeffs_stp_opti
    )

    km_coeffs_ift_opti = KMCoeffs(
        a11=popt_d11[0],
        b11=popt_d11[2],
        c11=popt_d11[1],
        a20=popt_d20[0],
        b20=popt_d20[2],
        c20=popt_d20[1],
        a21=popt_d21[0],
        b21=popt_d21[2],
        c21=popt_d21[1],
        a22=popt_d22[0],
        b22=popt_d22[2],
        c22=popt_d22[1],
    )

    return km_coeffs_ift_opti, optimization_history


def ift_objective_function(
    x0,
    scales_dimless,
    n_scales,
    km_coeffs,
    largest_scale,
    smallest_scale,
    data,
    fs,
    scale_subsample_step_us,
    taylor_scale,
    taylor_hyp_vel,
    overlap_trajs_flag,
    available_ram_gb,
):
    # Fit parameters.
    popt_d11 = fit_d1j(scales_dimless, x0[:n_scales], km_coeffs)
    popt_d20, popt_d21, popt_d22 = fit_d2j(scales_dimless, x0, n_scales, km_coeffs)

    # Create km_coeffs object.
    km_coeffs = KMCoeffs(
        a11=popt_d11[0],
        b11=popt_d11[2],
        c11=popt_d11[1],
        a20=popt_d20[0],
        b20=popt_d20[2],
        c20=popt_d20[1],
        a21=popt_d21[0],
        b21=popt_d21[2],
        c21=popt_d21[1],
        a22=popt_d22[0],
        b22=popt_d22[2],
        c22=popt_d22[1],
    )

    # Compute entropy.
    _, _, total_entropy = get_entropies(
        data,
        km_coeffs,
        fs,
        smallest_scale,
        largest_scale,
        scale_subsample_step_us,
        taylor_scale,
        taylor_hyp_vel,
        overlap_trajs_flag,
        available_ram_gb,
    )

    # Remove values that would cause overflow in exp(-total_entropy).
    threshold = -np.log(np.finfo(np.float64).max)
    total_entropy = total_entropy[total_entropy >= threshold]
    if total_entropy.size == 0:
        return 1e10, np.nan

    # Compute exp(-total_entropy) once.
    exp_neg = np.exp(-total_entropy)
    ift = np.nanmean(exp_neg)

    # Precompute cumulative sums so that slice means can be computed quickly.
    cumsum_exp = np.cumsum(exp_neg)
    size = total_entropy.size

    # Fast average using cumulative sums.
    def get_mean(n):
        return cumsum_exp[n - 1] / n

    # Use cumulative sum to compute error for each portion.
    def compute_error(portion):
        n = int(np.ceil(portion * size))
        return np.abs(1 - get_mean(n)) if n > 5000 else 0.0

    error_60 = compute_error(0.6)
    error_70 = compute_error(0.7)
    error_80 = compute_error(0.8)
    error_90 = compute_error(0.9)
    error_100 = np.abs(1 - ift)

    error = error_60 + error_70 + error_80 + error_90 + error_100

    return error, ift


def ift_run_optimization(
    x0,
    lower_bound,
    upper_bound,
    iter_max,
    scales_dimless,
    n_scales,
    km_coeffs,
    largest_scale,
    smallest_scale,
    data,
    fs,
    scale_subsample_step_us,
    taylor_scale,
    taylor_hyp_vel,
    overlap_trajs_flag,
    available_ram_gb,
):
    bounds = [(_l, _u) for _l, _u in zip(lower_bound, upper_bound)]
    optimization_history = {"x_iter": [], "error": [], "ift": []}

    def objective(x):
        error, ift = ift_objective_function(
            x,
            scales_dimless,
            n_scales,
            km_coeffs,
            largest_scale,
            smallest_scale,
            data,
            fs,
            scale_subsample_step_us,
            taylor_scale,
            taylor_hyp_vel,
            overlap_trajs_flag,
            available_ram_gb,
        )
        optimization_history["x_iter"].append(np.copy(x))
        optimization_history["error"].append(error)
        optimization_history["ift"].append(ift)
        return error

    def callback(_):
        iter_n = len(optimization_history["error"])
        ift = optimization_history["ift"][-1]
        error = optimization_history["error"][-1]
        logger.info(
            "\n"
            + "-" * 40
            + f"\n# Evaluations: {iter_n}\nIFT: {ift}\nError: {error}\n"
            + "-" * 40
            + "\n"
        )

    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": iter_max, "gtol": 1e-8},
        callback=callback,
    )
    optimization_history["n_iter"] = np.arange(len(optimization_history["error"]))
    return res, optimization_history


def fit_d1j(scales_dimless, y, km_coeffs):
    p0 = [km_coeffs.a11, km_coeffs.c11, km_coeffs.b11]
    lower_bounds = [-2, -2, -np.inf]
    upper_bounds = [2, 2, 0]
    bounds = (lower_bounds, upper_bounds)
    popt_d11, _ = curve_fit(
        model_d11, scales_dimless, y, p0=p0, bounds=bounds, maxfev=10_000
    )
    return popt_d11


def fit_d2j(scales_dimless, y, n_scales, km_coeffs):
    y_d20 = y[n_scales : 2 * n_scales]
    y_d21 = y[2 * n_scales : 3 * n_scales]
    y_d22 = y[3 * n_scales : 4 * n_scales]
    p0_d20 = [km_coeffs.a20, km_coeffs.c20, km_coeffs.b20]
    bounds_d20 = ([-1, -1, 0], [1, 1, np.inf])
    popt_d20, _ = curve_fit(
        model_d20, scales_dimless, y_d20, p0=p0_d20, bounds=bounds_d20, maxfev=10_000
    )
    p0_d21 = [km_coeffs.a21, km_coeffs.c21, km_coeffs.b21]
    bounds_d21 = ([-1, -1, -np.inf], [1, 1, np.inf])
    popt_d21, _ = curve_fit(
        model_d21, scales_dimless, y_d21, p0=p0_d21, bounds=bounds_d21, maxfev=10_000
    )
    p0_d22 = [km_coeffs.a22, km_coeffs.c22, km_coeffs.b22]
    bounds_d22 = ([-1, -1, -np.inf], [1, 1, 0])
    popt_d22, _ = curve_fit(
        model_d22, scales_dimless, y_d22, p0=p0_d22, bounds=bounds_d22, maxfev=10_000
    )
    return popt_d20, popt_d21, popt_d22


def model_d11(scales_dimless, a11, c11, b11):
    return a11 * scales_dimless**b11 + c11


def model_d20(scales_dimless, a20, c20, b20):
    return a20 * scales_dimless**b20 + c20


def model_d21(scales_dimless, a21, c21, b21):
    return a21 * scales_dimless**b21 + c21


def model_d22(scales_dimless, a22, c22, b22):
    return a22 * scales_dimless**b22 + c22
