import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize
from tqdm import tqdm

from ..utils.general import get_pdf
from ..utils.logger_setup import logger
from ..utils.storing_clases import Entropies, KMCoeffs


# --- Entropy calculation functions ---
def compute_entropy(
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
):
    medium_entropy, system_entropy, total_entropy = _compute_entropy(
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

    entropies = Entropies(
        medium_entropy=medium_entropy,
        system_entropy=system_entropy,
        total_entropy=total_entropy,
    )

    return (
        entropies,
        None,  # indep_scales,
        None,  # incs_start_idxs,
        None,  # indep_scales_idxs,
        None,  # scale_subsample_step_us,
        None,  # incs_for_all_scales,
        None,  # lagrangian,
        None,  # action,
        None,  # momentum,
        None,  # hamiltonian,
        None,  # hamiltonian_derivative1,
        None,  # hamiltonian_derivative2,
    )


def _compute_entropy(
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
):
    """
    Computes the total entropy production (total_entropy = medium_entropy + system_entropy) for cascade trajs

    Parameters
    ----------
    data : ndarray
        2D array with the data.
    fs : float
        Sampling frequency.
    km_coeffs : KMCoeffs
        Dataclass with Kramers-Moyal km_coeffs.
    smallest_scale : float
        The dimensionless smallest scale to be used.
    largest_scale : float
        The dimensionless largest scale to be used.
    scale_subsample_step_us : int
        Step size for subsampling scales.
    taylor_scale : float
        Taylor length scale (used to normalize other scales).
    taylor_hyp_vel : float
        Velocity used for distance-to-time conversion.
    overlap_trajs_flag : float, optional
    available_ram_gb : float, optional
        Available RAM in GB for chunking. Default is 8 GB.

    Returns
    -------
    medium_entropy : ndarray
        Medium entropy production for each trajectory.
    system_entropy : ndarray
        System (Shannon) entropy production for each trajectory.
    total_entropy : ndarray
        Total entropy production = medium_entropy + system_entropy.
    indep_scales : ndarray
        Scale vector (dimensionless) from largest to smallest in the cascade.
    incs_start_idxs : ndarray or None
        Indices in data corresponding to the start of each trajectory.
    indep_scales_idxs : ndarray
        Indices for the scales used.
    scale_subsample_step_us : int
        Step size used for subsampling scales.
    indep_incs_for_all_scales : ndarray or None
        Matrix of increments for each trajectory.
    lagrangian : ndarray or None
        Lagrangian.
    action : ndarray or None
        Action functional (path integral of Lagrangian).
    momentum : ndarray or None
        Conjugate variable (Hamiltonian formalism).
    hamiltonian : ndarray or None
        Hamiltonian.
    hamiltonian_derivative1 : ndarray or None
        Auxiliary Hamiltonian derivative.
    hamiltonian_derivative2 : ndarray or None
        Auxiliary Hamiltonian derivative.
    """

    to_us = fs / taylor_hyp_vel

    # Use dimensionless scales.
    largest_scale_dimless = largest_scale / taylor_scale
    smallest_scale_dimless = smallest_scale / taylor_scale

    num_points = int(np.ceil(1.1 * largest_scale * to_us))
    all_indep_scales_dimless = np.arange(num_points) / to_us / taylor_scale

    # Find indices closest to the target scales.
    # For largest_scale, we want the last occurrence of the minimum difference.
    largest_index = all_indep_scales_dimless.size - np.argmin(
        np.abs(all_indep_scales_dimless[::-1] - largest_scale_dimless)
    )
    # For smallest_scale, we take the first occurrence.
    smallest_index = np.argmin(
        np.abs(all_indep_scales_dimless - smallest_scale_dimless)
    )

    trajectory_chunk_length = largest_index

    # Take only the independent scales between the smallest and the largest scale.
    indep_scales_idxs = np.arange(smallest_index, largest_index + 1)[::-1]
    indep_scales = all_indep_scales_dimless[indep_scales_idxs]

    # Choose between overlapping or non-overlapping increments
    if overlap_trajs_flag:
        incs_for_all_scales_iterator = get_overlap_incs(
            data, indep_scales_idxs, trajectory_chunk_length, available_ram_gb
        )
    else:
        incs_for_all_scales_iterator = get_non_overlap_idep_incs(
            data, indep_scales_idxs, trajectory_chunk_length, available_ram_gb
        )

    medium_entropy, system_entropy, total_entropy = compute_entropies_for_all_scales(
        km_coeffs, incs_for_all_scales_iterator, indep_scales, scale_subsample_step_us
    )

    # This works but for the moment I will not returned it
    # Lagrangian and Hamiltonian quantities
    # (
    # lagrangian,
    # action,
    # momentum,
    # hamiltonian,
    # hamiltonian_derivative1,
    # hamiltonian_derivative2,
    # ) = compute_path_integral(
    # incs_deriv_central,
    # incs_central,
    # scales_central,
    # scale_spacing_central,
    # km_coeffs,
    # )

    return medium_entropy, system_entropy, total_entropy


def compute_entropies_for_all_scales(
    km_coeffs, incs_for_all_scales_iterator, indep_scales, scale_subsample_step_us
):
    medium_entropy_l, system_entropy_l, total_entropy_l = [], [], []
    for incs_for_all_scales in incs_for_all_scales_iterator:
        # Independent increments for scales separated by (probably) one markovian step

        sampled_scales = indep_scales[::scale_subsample_step_us]
        sampled_incs = incs_for_all_scales[:, ::scale_subsample_step_us]

        # Midpoint derivative (and properly evaluated incs and scales in those points)
        scales_central, scale_spacing_central, incs_central, incs_deriv_central = (
            compute_central_derivative(sampled_incs, sampled_scales)
        )

        # Entropy
        medium_entropy = compute_medium_entropy(
            incs_deriv_central,
            incs_central,
            scales_central,
            scale_spacing_central,
            km_coeffs,
        )

        system_entropy = compute_system_entropy(incs_central)

        total_entropy = medium_entropy + system_entropy
        total_entropy[~np.isfinite(total_entropy)] = np.nan

        medium_entropy_l.append(medium_entropy)
        system_entropy_l.append(system_entropy)
        total_entropy_l.append(total_entropy)

    medium_entropy = np.concatenate(medium_entropy_l)
    system_entropy = np.concatenate(system_entropy_l)
    total_entropy = np.concatenate(total_entropy_l)

    return medium_entropy, system_entropy, total_entropy


def get_non_overlap_idep_incs(
    data: np.ndarray,
    indep_scales_idxs: np.ndarray,
    trajectory_chunk_length: int,
    available_ram_gb: float,
):
    # TODO: Generalize this to data with other shapes
    data = data.flatten()
    data_len = len(data)
    step_size = trajectory_chunk_length + 1
    starts = np.arange(0, data_len, step_size)

    valid_1 = (starts + indep_scales_idxs[0]) < data_len
    valid_2 = (starts + indep_scales_idxs[-1]) < data_len
    valid_starts = starts[valid_1 & valid_2]

    max_chunk_size = _compute_max_chunk_size(
        available_ram_gb, n_scales=len(indep_scales_idxs)
    )

    if max_chunk_size < 1:
        raise MemoryError("Not enough RAM available for even one trajectory chunk.")

    for i in range(0, len(starts), max_chunk_size):
        chunk_starts = valid_starts[i : i + max_chunk_size]
        chunk_incs = (
            data[chunk_starts[:, None] + indep_scales_idxs] - data[chunk_starts, None]
        )
        yield chunk_incs


def get_overlap_incs(
    data: np.ndarray,
    indep_scales_idxs: np.ndarray,
    trajectory_chunk_length: int,
    available_ram_gb: float,
):
    data = data.flatten()
    data_len = len(data)

    # For overlapping trajectories, use a step of 1
    starts = np.arange(0, data_len - trajectory_chunk_length)

    max_chunk_size = _compute_max_chunk_size(
        available_ram_gb, n_scales=len(indep_scales_idxs)
    )

    if max_chunk_size < 1:
        raise MemoryError("Not enough RAM available for even one trajectory chunk.")

    for i in tqdm(range(0, len(starts), max_chunk_size)):
        chunk_starts = starts[i : i + max_chunk_size]
        chunk_incs = (
            data[chunk_starts[:, None] + indep_scales_idxs] - data[chunk_starts, None]
        )
        yield chunk_incs


def compute_central_derivative(sampled_incs: np.ndarray, sampled_scales: np.ndarray):
    scale_spacing_raw = (sampled_scales[2:] - sampled_scales[:-2]) / 2.0
    scale_spacing_central = -np.tile(scale_spacing_raw, (sampled_incs.shape[0], 1))

    # midpoint increments (Stratonovich convention)
    incs_central = sampled_incs[:, 1:-1]
    incs_deriv_central = (sampled_incs[:, 2:] - sampled_incs[:, :-2]) / (
        2.0 * scale_spacing_central
    )
    scales_central = np.tile(sampled_scales[1:-1], (sampled_incs.shape[0], 1))

    return scales_central, scale_spacing_central, incs_central, incs_deriv_central


def compute_medium_entropy(
    incs_deriv_central: np.ndarray,
    incs_central: np.ndarray,
    scales_central: np.ndarray,
    delta_scales: np.ndarray,
    km_coeffs: dict,
):
    D1 = _D1(incs_central, scales_central, km_coeffs)
    D2 = _D2(incs_central, scales_central, km_coeffs)
    D2_derivative = _D2_diff(incs_central, scales_central, km_coeffs)
    F = D1 - (D2_derivative / 2.0)
    FD = F / D2
    Sm_tmp = incs_deriv_central * FD * delta_scales
    Sm_1 = np.nansum(Sm_tmp, axis=1)
    return Sm_1


def compute_system_entropy(incs_central):
    nbins = 301
    large_scale_incs = incs_central[:, 0]
    small_scale_incs = incs_central[:, -1]
    large_scale_incs = large_scale_incs[~np.isnan(large_scale_incs)]
    small_scale_incs = small_scale_incs[~np.isnan(small_scale_incs)]

    large_scale_pdf, bin_edges = np.histogram(
        large_scale_incs, bins=nbins, density=True
    )
    large_scale_bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    small_scale_pdf, _ = np.histogram(small_scale_incs, bins=bin_edges, density=True)
    small_scale_bin_centers = large_scale_bin_centers.copy()

    system_entropy = []
    for i in range(incs_central.shape[0]):
        large_scale_increment = incs_central[i, 0]
        small_scale_increment = incs_central[i, -1]

        valid_large_scale = large_scale_pdf > 0
        valid_small_scale = small_scale_pdf > 0
        entropy_large_scale = np.interp(
            large_scale_increment,
            large_scale_bin_centers[valid_large_scale],
            large_scale_pdf[valid_large_scale],
            left=np.nan,
            right=np.nan,
        )
        entropy_small_scale = np.interp(
            small_scale_increment,
            small_scale_bin_centers[valid_small_scale],
            small_scale_pdf[valid_small_scale],
            left=np.nan,
            right=np.nan,
        )

        system_entropy_i = -np.log(entropy_small_scale / entropy_large_scale)
        system_entropy.append(system_entropy_i)
    system_entropy = np.array(system_entropy)
    return system_entropy


def compute_path_integral(
    incs_deriv_central: np.ndarray,
    incs_central: np.ndarray,
    scales_central: np.ndarray,
    delta_scales: np.ndarray,
    km_coeffs: dict,
):
    D1 = _D1(incs_central, scales_central, km_coeffs)
    D2 = _D2(incs_central, scales_central, km_coeffs)
    D1_derivative = _D1_diff(incs_central, scales_central, km_coeffs)
    D2_derivative = _D2_diff(incs_central, scales_central, km_coeffs)
    D2_second_derivative = _D2_diff_diff(incs_central, scales_central, km_coeffs)
    lagrangian = (
        ((incs_deriv_central - D1 + (D2_derivative / 2.0)) ** 2) / (4.0 * D2)
    ) - (D1_derivative / 2.0)
    action = np.nansum(lagrangian * delta_scales, axis=1)
    momentum = (incs_deriv_central - D1 + (D2_derivative / 2.0)) / (2.0 * D2)
    hamiltonian = (
        D2 * momentum**2
        + (D1 - (D2_derivative / 2.0)) * momentum
        - (D1_derivative / 2.0)
    )
    hamiltonian_derivative1 = (2.0 * D2 * momentum) + D1 - (D2_derivative / 2.0)
    hamiltonian_derivative2 = -(
        (momentum**2 * D2_derivative)
        + (D1_derivative - (D2_second_derivative / 2.0)) * momentum
        - (D1_derivative / 2.0)
    )
    hamiltonian_derivative2 = -hamiltonian_derivative2
    # tmpvar = np.cumsum(lagrangian * delta_scales, axis=1)
    # return lagrangian, action, momentum, hamiltonian, tmpvar, hamiltonian_derivative1, hamiltonian_derivative2
    return (
        lagrangian,
        action,
        momentum,
        hamiltonian,
        hamiltonian_derivative1,
        hamiltonian_derivative2,
    )


def _D1(incs: np.ndarray, scales: np.ndarray, kmcoeffs: KMCoeffs) -> np.ndarray:
    """
    Compute D1 using the KMCoeffs object.

    Uses:
      D1(u, r) = [a11 * r^(b11) + c11] * u
    where u is represented by 'incs' and r by 'scales'.
    """
    D1 = kmcoeffs.eval_D1(incs, scales)
    D1[~np.isfinite(D1)] = np.nan
    return D1


def _D1_diff(_: np.ndarray, scales: np.ndarray, kmcoeffs: KMCoeffs) -> np.ndarray:
    """
    Compute the derivative of D1 with respect to u (incs).
    """
    D1_diff = kmcoeffs.a11 * (scales**kmcoeffs.b11) + kmcoeffs.c11
    D1_diff[~np.isfinite(D1_diff)] = np.nan
    return D1_diff


def _D2(incs: np.ndarray, scales: np.ndarray, kmcoeffs: KMCoeffs) -> np.ndarray:
    D2 = kmcoeffs.eval_D2(incs, scales)
    D2[~np.isfinite(D2)] = np.nan
    D2[D2 <= 0] = np.nan
    return D2


def _D2_diff(incs: np.ndarray, scales: np.ndarray, kmcoeffs: KMCoeffs) -> np.ndarray:
    """
    Compute the derivative of D2 with respect to u (incs).
    """
    d21 = kmcoeffs.a21 * (scales**kmcoeffs.b21) + kmcoeffs.c21
    d22 = kmcoeffs.a22 * (scales**kmcoeffs.b22) + kmcoeffs.c22
    D2_diff = d21 + 2.0 * d22 * incs
    D2_diff[~np.isfinite(D2_diff)] = np.nan
    return D2_diff


def _D2_diff_diff(_: np.ndarray, scales: np.ndarray, kmcoeffs: KMCoeffs) -> np.ndarray:
    """
    Compute the second derivative of D2 with respect to u (incs).
    """
    d22 = kmcoeffs.a22 * (scales**kmcoeffs.b22) + kmcoeffs.c22
    D2_dd = 2.0 * d22
    D2_dd[~np.isfinite(D2_dd)] = np.nan
    return D2_dd


def _compute_max_chunk_size(available_ram_gb, n_scales):
    available_ram_bytes = available_ram_gb * 1.07e9
    bytes_per_value = 8  # assuming float64

    max_chunk_size = int(np.floor(available_ram_bytes / (n_scales * bytes_per_value)))
    max_chunk_size = max_chunk_size // 5  # need for temporal allocations

    return max_chunk_size


# --- IFT Optimization functions ---
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
    overlap_trajs_flag,
    available_ram_gb,
):
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
            + "-" * 50
            + f"\n# Evaluations: {iter_n}\nIFT: {ift}\nError: {error}\n"
            + "-" * 50
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
    _, _, total_entropy = _compute_entropy(
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
    n90 = int(np.ceil(0.9 * size))
    error_90 = np.abs(1 - get_mean(n90)) if n90 > 0 else 0.0
    error_100 = np.abs(1 - ift)

    error = error_60 + error_70 + error_80 + error_90 + error_100

    return error, ift


def fit_d1j(scales_dimless, y, km_coeffs):
    p0 = [km_coeffs.a11, km_coeffs.c11, km_coeffs.b11]
    lower_bounds = [-2, -2, -np.inf]
    upper_bounds = [2, 2, 0]
    bounds = (lower_bounds, upper_bounds)
    popt_d11, _ = curve_fit(
        model_d11, scales_dimless, y, p0=p0, bounds=bounds, maxfev=1000
    )
    return popt_d11


def fit_d2j(scales_dimless, y, n_scales, km_coeffs):
    y_d20 = y[n_scales : 2 * n_scales]
    y_d21 = y[2 * n_scales : 3 * n_scales]
    y_d22 = y[3 * n_scales : 4 * n_scales]
    p0_d20 = [km_coeffs.a20, km_coeffs.c20, km_coeffs.b20]
    bounds_d20 = ([-1, -1, 0], [1, 1, np.inf])
    popt_d20, _ = curve_fit(
        model_d20, scales_dimless, y_d20, p0=p0_d20, bounds=bounds_d20, maxfev=1000
    )
    p0_d21 = [km_coeffs.a21, km_coeffs.c21, km_coeffs.b21]
    bounds_d21 = ([-1, -1, -np.inf], [1, 1, np.inf])
    popt_d21, _ = curve_fit(
        model_d21, scales_dimless, y_d21, p0=p0_d21, bounds=bounds_d21, maxfev=1000
    )
    p0_d22 = [km_coeffs.a22, km_coeffs.c22, km_coeffs.b22]
    bounds_d22 = ([-1, -1, -np.inf], [1, 1, 0])
    popt_d22, _ = curve_fit(
        model_d22, scales_dimless, y_d22, p0=p0_d22, bounds=bounds_d22, maxfev=1000
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


# --- Plot functions ---
def plot_entropy_and_ift(entropies, nbins):
    medium_entropy = entropies.medium_entropy
    system_entropy = entropies.system_entropy
    total_entropy = entropies.total_entropy
    # l, u = -5, 9
    # idx = (l < total_entropy) & (total_entropy < u)
    # idx = np.abs(total_entropy) < 5*np.nanstd(np.abs(total_entropy))
    # medium_entropy = medium_entropy[idx]
    # system_entropy = system_entropy[idx]
    # total_entropy = total_entropy[idx]

    n_samps = 20
    sample_sizes = np.round(
        np.logspace(1, np.log10(total_entropy.size), n_samps)
    ).astype(int)
    ft_values = np.empty(n_samps)
    ft_errors = np.empty(n_samps)
    for i, size in enumerate(sample_sizes):
        slice_data = np.exp(-total_entropy[:size])
        ft_values[i] = np.nanmean(slice_data)
        ft_errors[i] = np.nanstd(slice_data) / np.sqrt(size)

    # Create one figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    hist_min = np.nanmin([medium_entropy, system_entropy, total_entropy])
    hist_max = np.nanmax([medium_entropy, system_entropy, total_entropy])
    bins = np.linspace(hist_min, hist_max, nbins)
    centers_med, pdf_med = get_pdf(medium_entropy, bins)
    centers_sys, pdf_sys = get_pdf(system_entropy, bins)
    centers_tot, pdf_tot = get_pdf(total_entropy, bins)

    # Plot the PDFs on the second axis
    ax1.plot(centers_med, pdf_med, label=r"$\Delta S_{\mathrm{med}}$", linewidth=1.5)
    ax1.plot(centers_sys, pdf_sys, label=r"$\Delta S_{\mathrm{sys}}$", linewidth=1.5)
    ax1.plot(
        centers_tot, pdf_tot, "--k", label=r"$\Delta S_{\mathrm{tot}}$", linewidth=1.5
    )
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.4)
    ax1.set_xlabel("Entropy")
    ax1.set_ylabel("PDF")
    mean_ds = np.nanmean(total_entropy)
    ax1.set_title(r"$\langle \Delta S_{\mathrm{tot}}\rangle = $" f"{mean_ds:0.2f}")
    ax1.axvline(0, color="k", linestyle="-")
    ax1.legend()

    ax2.errorbar(
        sample_sizes,
        ft_values,
        yerr=ft_errors,
        fmt="ok",
        linewidth=1.2,
        label=r"$\langle e^{-\Delta S_{\mathrm{tot}}}\rangle_N$",
    )
    ax2.axhline(1.0, color="k", linestyle="--", linewidth=1.0, label="ift = 1")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.4)
    ax2.set_xlabel(r"$N$")
    ax2.set_ylabel(r"$\langle e^{-\Delta S_{\mathrm{tot}}}\rangle_N$")
    mean_val = np.nanmean(np.exp(-total_entropy))
    ax2.set_title(
        r"$\langle e^{-\Delta S_{\mathrm{tot}}}\rangle_{\max(N)} =$ "
        f"{mean_val:0.2f}",
        fontsize=12,
    )
    ax2.legend()

    plt.show()
