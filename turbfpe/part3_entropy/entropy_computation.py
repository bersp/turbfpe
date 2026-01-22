from __future__ import annotations

import numpy as np

from ..utils.logger_setup import logger
from ..utils.storing_classes import Entropies, KMCoeffs


def compute_entropy(
    data,
    km_coeffs,
    fs,
    smallest_scale,
    largest_scale,
    scale_subsample_step_us,
    taylor_scale,
    taylor_hyp_vel,
    compute_entropy_steps=False,
):
    (
        incs_for_all_scales,
        indep_scales,
        scale_subsample_step_us,
        indep_scales_idxs,
        traj_start_pairs,
    ) = compute_scales(
        data,
        fs,
        smallest_scale,
        largest_scale,
        scale_subsample_step_us,
        taylor_scale,
        taylor_hyp_vel,
    )

    medium_entropy, system_entropy, total_entropy, idx_track = (
        compute_entropies_for_all_scales(
            km_coeffs=km_coeffs,
            incs_for_all_scales=incs_for_all_scales,
            indep_scales=indep_scales,
            scale_subsample_step_us=scale_subsample_step_us,
            indep_scales_idxs=indep_scales_idxs,
            traj_start_pairs=traj_start_pairs,
            compute_entropy_steps=compute_entropy_steps,
        )
    )
    entropies = Entropies(
        medium_entropy=medium_entropy,
        system_entropy=system_entropy,
        total_entropy=total_entropy,
        idx_track=idx_track,
    )
    return entropies


def compute_scales(
    data,
    fs,
    smallest_scale,
    largest_scale,
    scale_subsample_step_us,
    taylor_scale,
    taylor_hyp_vel,
):
    to_us = fs / taylor_hyp_vel

    # Dimensionless scales
    largest_scale_dimless = largest_scale / taylor_scale
    smallest_scale_dimless = smallest_scale / taylor_scale

    num_points = int(np.ceil(1.1 * largest_scale * to_us))
    all_indep_scales_dimless = np.arange(num_points) / to_us / taylor_scale

    # Find the indices closest to the target scales.
    # For largest_scale, we want the last occurrence of the minimum difference.
    rev_idx = np.argmin(np.abs(all_indep_scales_dimless[::-1] - largest_scale_dimless))
    largest_index = all_indep_scales_dimless.size - 1 - rev_idx
    # For smallest_scale, we take the first occurrence.
    smallest_index = np.argmin(
        np.abs(all_indep_scales_dimless - smallest_scale_dimless)
    )

    trajectory_chunk_length = largest_index

    # Independent scales between smallest and largest (descending)
    indep_scales_idxs = np.arange(smallest_index, largest_index + 1)[::-1]
    indep_scales = all_indep_scales_dimless[indep_scales_idxs]

    # Increments for all rows
    incs_for_all_scales, traj_start_pairs = get_non_overlap_idep_incs(
        data,
        indep_scales_idxs,
        trajectory_chunk_length,
    )

    return (
        incs_for_all_scales,
        indep_scales,
        scale_subsample_step_us,
        indep_scales_idxs,
        traj_start_pairs,
    )


def compute_entropies_for_all_scales(
    km_coeffs: KMCoeffs,
    incs_for_all_scales: np.ndarray,
    indep_scales: np.ndarray,
    scale_subsample_step_us: int,
    indep_scales_idxs: np.ndarray,
    traj_start_pairs: np.ndarray,  # (N, 2) -> (traj_idx, start_time)
    compute_entropy_steps,
):
    """
    Compute entropies on the full set of increments and build index tracking.

    Returns
    -------
    medium_entropy : ndarray, shape (N, n_steps)
    system_entropy : ndarray, shape (N, n_steps)
    total_entropy : ndarray, shape (N, n_steps)
    idx_track : ndarray, shape (N, n_steps, 3, 2) with pairs (traj_idx, time_idx)
    """
    # Precompute sampled/central columns on the full indep_scales grid
    full_cols = np.arange(indep_scales.size)
    sampled_cols = full_cols[::scale_subsample_step_us]
    central_cols = sampled_cols[1:-1]  # matches incs_central columns

    # Subsample increments
    sampled_scales = indep_scales[::scale_subsample_step_us]
    sampled_incs = incs_for_all_scales[:, ::scale_subsample_step_us]

    # Midpoint derivative (and properly evaluated incs and scales in those points)
    scales_central, scale_spacing_central, incs_central, incs_deriv_central = (
        compute_central_derivative(sampled_incs, sampled_scales)
    )

    # Entropies
    if compute_entropy_steps:
        medium_entropy = compute_medium_entropy_w_steps(
            incs_deriv_central,
            incs_central,
            scales_central,
            scale_spacing_central,
            km_coeffs,
        )

        system_entropy = compute_system_entropy_w_steps(incs_central)

        idx_track = _build_index_track(
            traj_start_pairs=traj_start_pairs,
            indep_scales_idxs=indep_scales_idxs,
            central_scale_column_indices=central_cols,
        )
    else:
        medium_entropy = compute_medium_entropy(
            incs_deriv_central,
            incs_central,
            scales_central,
            scale_spacing_central,
            km_coeffs,
        )

        system_entropy = compute_system_entropy(incs_central)

        idx_track = np.array([])

    total_entropy = medium_entropy + system_entropy
    total_entropy[~np.isfinite(total_entropy)] = np.nan

    return medium_entropy, system_entropy, total_entropy, idx_track


def get_non_overlap_idep_incs(
    data: np.ma.MaskedArray,
    indep_scales_idxs: np.ndarray,
    trajectory_chunk_length: int,
):
    """
    Build non-overlapping increments.

    Returns
    -------
    incs_all : ndarray, shape (n_casc_trajs, n_full_scales)
        Increments for every cascade trajectory across the full independent scale grid.
    traj_start_pairs : ndarray, shape (n_casc_trajs, 2)
        (row_index, start_time_index) for each cascade trajectory.
    """

    mask = np.ma.getmaskarray(data)
    n_trajs, n_times = data.shape
    valid_len = n_times - mask.sum(axis=1)

    step_size = trajectory_chunk_length + 1
    max_offset = int(indep_scales_idxs[0])

    incs_list = []
    starts_list = []

    trajs_discarted_len = []
    trajs_tails_discarted_len = []
    total_valid_samples = 0
    for r in range(n_trajs):
        n_avail = int(valid_len[r])
        total_valid_samples += n_avail
        if n_avail <= max_offset:
            trajs_discarted_len.append(n_avail)
            continue

        # Non-overlapping starts. Tail is discarded.
        limit = n_avail - max_offset
        starts = np.arange(0, limit, step_size, dtype=int)

        trajs_tails_discarted_len.append(n_avail - (starts[-1] + max_offset + 1))

        # Build increments for row r
        # Shape: (n_starts, n_full_scales)
        row_vals = data[r].filled(np.nan)  # valid prefix has real data; mask is tail
        incs_r = (
            row_vals[starts[:, None] + indep_scales_idxs[None, :]]
            - row_vals[starts, None]
        )

        incs_list.append(incs_r)
        # start pairs (traj_idx, time_idx) for each start
        starts_r = np.column_stack([np.full(starts.size, r, dtype=int), starts])
        starts_list.append(starts_r)

    if len(incs_list) == 0:
        # No valid cascades
        n_full_scales = indep_scales_idxs.size
        return np.empty((0, n_full_scales), dtype=float), np.empty((0, 2), dtype=int)

    incs_all = np.vstack(incs_list)
    traj_start_pairs = np.vstack(starts_list)

    n_samples_discarted = sum(trajs_discarted_len) + sum(trajs_tails_discarted_len)

    logger.info(
        "Discarded samples (short trajectories + discarded tails):: "
        f"{n_samples_discarted/total_valid_samples:.1%} ({n_samples_discarted:.2E} / {total_valid_samples:.2E})"
    )

    logger.info(
        "Discarded trajectories (len(traj) < largest_scale_us): "
        f"{len(trajs_discarted_len)/n_trajs:.1%} ({len(trajs_discarted_len)} / {n_trajs})"
    )

    logger.info(
        f"Discarded samples from truncated tail (sum_traj [len(traj) - N*largest_scale_us]): {sum(trajs_tails_discarted_len):.2E}"
    )

    logger.info(
        f"Mean discarded tail per kept trajectory / largest_scale_us: "
        f"{sum(trajs_tails_discarted_len) / len(trajs_tails_discarted_len) / max_offset:.1%}"
    )

    logger.info(
        f"Total number of cascade trajectories: {incs_all.shape[0]}"
    )

    return incs_all, traj_start_pairs


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
    km_coeffs: KMCoeffs,
):
    D1 = _D1(incs_central, scales_central, km_coeffs)
    D2 = _D2(incs_central, scales_central, km_coeffs)
    D2_derivative = _D2_diff(incs_central, scales_central, km_coeffs)
    F = D1 - (D2_derivative / 2.0)
    FD = F / D2
    ent_steps = incs_deriv_central * FD * delta_scales
    ent = np.nansum(ent_steps, axis=1)
    return ent


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


def compute_medium_entropy_w_steps(
    incs_deriv_central: np.ndarray,
    incs_central: np.ndarray,
    scales_central: np.ndarray,
    delta_scales: np.ndarray,
    km_coeffs: KMCoeffs,
):
    D1 = _D1(incs_central, scales_central, km_coeffs)
    D2 = _D2(incs_central, scales_central, km_coeffs)
    D2_derivative = _D2_diff(incs_central, scales_central, km_coeffs)
    F = D1 - (D2_derivative / 2.0)
    FD = F / D2
    ent_steps = incs_deriv_central * FD * delta_scales
    ent_steps[:, 0] *= 2
    ent_steps[:, -1] *= 2
    return 0.5 * (ent_steps[:, :-1] + ent_steps[:, 1:])


def compute_system_entropy_w_steps(incs_central):
    nbins = 301
    # Build common bins from the largest scale (col 0)
    ref = incs_central[:, 0]
    ref = ref[~np.isnan(ref)]
    _, bin_edges = np.histogram(ref, bins=nbins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # PDFs for all scales on the same bins
    n_scales = incs_central.shape[1]
    pdfs = np.empty((n_scales, bin_centers.size))
    valids = np.empty_like(pdfs, dtype=bool)
    for k in range(n_scales):
        x = incs_central[:, k]
        x = x[~np.isnan(x)]
        pdfs[k], _ = np.histogram(x, bins=bin_edges, density=True)
        valids[k] = pdfs[k] > 0

    # Stepwise entropies: -log p_{k+1}(x_{i,k+1}) / p_k(x_{i,k})
    N = incs_central.shape[0]
    ent_steps = np.full((N, n_scales - 1), np.nan)

    for i in range(N):
        for k in range(n_scales - 1):
            # denom: p_k at x_{i,k}
            denom = (
                np.interp(
                    incs_central[i, k],
                    bin_centers[valids[k]],
                    pdfs[k][valids[k]],
                    left=np.nan,
                    right=np.nan,
                )
                if np.any(valids[k])
                else np.nan
            )

            # num: p_{k+1} at x_{i,k+1}
            num = (
                np.interp(
                    incs_central[i, k + 1],
                    bin_centers[valids[k + 1]],
                    pdfs[k + 1][valids[k + 1]],
                    left=np.nan,
                    right=np.nan,
                )
                if np.any(valids[k + 1])
                else np.nan
            )

            ent_steps[i, k] = (
                -np.log(num / denom)
                if np.isfinite(num) and np.isfinite(denom)
                else np.nan
            )

    return ent_steps


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


def _build_index_track(
    traj_start_pairs: np.ndarray,
    indep_scales_idxs: np.ndarray,
    central_scale_column_indices: np.ndarray,
) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]]:
    """
    Build index-tracking triplets [trajectory_start, scale_k, scale_kp1],
    each as (traj_idx, time_idx) pairs for every step between adjacent central scales.

    Parameters
    ----------
    traj_start_pairs : (N, 2)
        (traj_idx, start_time_idx) for each cascade trajectory.
    indep_scales_idxs : (n_full_scales,)
        Absolute offsets on the full independent scale grid (descending).
    central_scale_column_indices : (n_central_cols,)
        Column indices on the full scale grid used for central differences.

    Returns
    -------
    idx_track : (N, n_steps, 3, 2)
        Triplets [trajectory_start, scale_k, scale_kp1].
    """

    N = traj_start_pairs.shape[0]
    n_steps = int(central_scale_column_indices.size) - 1
    if n_steps < 1:
        return np.empty((N, 0, 3, 2), dtype=int)

    traj_indices = traj_start_pairs[:, 0]
    start_times = traj_start_pairs[:, 1]

    i_offsets = indep_scales_idxs[central_scale_column_indices[:-1]]
    i_plus_1_offsets = indep_scales_idxs[central_scale_column_indices[1:]]

    i_times = start_times[:, None] + i_offsets[None, :]
    i_plus_1_times = start_times[:, None] + i_plus_1_offsets[None, :]

    traj_indices_rep = np.repeat(traj_indices[:, None], n_steps, axis=1)

    trajectory_start = np.repeat(
        traj_start_pairs[:, None, :], n_steps, axis=1
    )  # (N,n_steps,2)
    scale_k = np.stack([traj_indices_rep, i_times], axis=2).astype(int)  # (N,n_steps,2)
    scale_kp1 = np.stack([traj_indices_rep, i_plus_1_times], axis=2).astype(int)

    idx_track = np.stack(
        [trajectory_start, scale_k, scale_kp1], axis=2
    )  # (N,n_steps,3,2)
    return idx_track
