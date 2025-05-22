import numpy as np
from tqdm import tqdm

from ..utils.storing_clases import Entropies, KMCoeffs


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
    medium_entropy, system_entropy, total_entropy, idx_track = get_entropies(
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
        idx_track=idx_track,
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


def get_entropies(
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
    Computes the medium_entropy, system_entropy and total_entropy production for cascade trajs.

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
    indep_scales : ndarray (COMMENTED)
        Scale vector (dimensionless) from largest to smallest in the cascade.
    incs_start_idxs : ndarray or None (COMMENTED)
        Indices in data corresponding to the start of each trajectory.
    indep_scales_idxs : ndarray (COMMENTED)
        Indices for the scales used.
    scale_subsample_step_us : int (COMMENTED)
        Step size used for subsampling scales.
    indep_incs_for_all_scales : ndarray or None (COMMENTED)
        Matrix of increments for each trajectory.
    lagrangian : ndarray or None (COMMENTED)
        Lagrangian.
    action : ndarray or None (COMMENTED)
        Action functional (path integral of Lagrangian).
    momentum : ndarray or None (COMMENTED)
        Conjugate variable (Hamiltonian formalism).
    hamiltonian : ndarray or None (COMMENTED)
        Hamiltonian.
    hamiltonian_derivative1 : ndarray or None (COMMENTED)
        Auxiliary Hamiltonian derivative.
    hamiltonian_derivative2 : ndarray or None (COMMENTED)
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

    medium_entropy, system_entropy, total_entropy, idx_track = (
        compute_entropies_for_all_scales(
            km_coeffs,
            incs_for_all_scales_iterator,
            indep_scales,
            scale_subsample_step_us,
        )
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

    return medium_entropy, system_entropy, total_entropy, idx_track


def compute_entropies_for_all_scales(
    km_coeffs, incs_for_all_scales_iterator, indep_scales, scale_subsample_step_us
):
    medium_entropy_l, system_entropy_l, total_entropy_l = [], [], []
    idx_track_l = []
    for incs_for_all_scales, incs_for_all_scales_idx in incs_for_all_scales_iterator:
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

        idx_track_l.append(incs_for_all_scales_idx)

    medium_entropy = np.concatenate(medium_entropy_l)
    system_entropy = np.concatenate(system_entropy_l)
    total_entropy = np.concatenate(total_entropy_l)

    idx_track = np.concatenate(idx_track_l, axis=0)

    return medium_entropy, system_entropy, total_entropy, idx_track


def get_non_overlap_idep_incs(
    data: np.ndarray,
    indep_scales_idxs: np.ndarray,
    trajectory_chunk_length: int,
    available_ram_gb: float,
):
    orig_shape = data.shape
    flat_idx = np.flatnonzero(~data.mask.ravel())
    data = data.compressed()  # TODO: Generalize this to data with other shapes

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
        chunk_coords = np.stack(
            np.unravel_index(flat_idx[chunk_starts], orig_shape), axis=1
        )
        yield chunk_incs, chunk_coords


def get_overlap_incs(
    data: np.ndarray,
    indep_scales_idxs: np.ndarray,
    trajectory_chunk_length: int,
    available_ram_gb: float,
):
    # TODO: Generalize this to data with other shapes
    data = data.compressed()

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
