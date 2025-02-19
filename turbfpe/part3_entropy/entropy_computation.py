import matplotlib.pyplot as plt
import numpy as np

from ..utils.storing_clases import KMCoeffs, Entropies
from ..utils.general import get_pdf


def compute_entropy(
    data,
    km_coeffs,
    fs,
    smallest_scale,
    largest_scale,
    scale_subsample_step,
    taylor_scale,
    taylor_hyp_vel,
):
    """
    Computes the total entropy production (total_entropy = medium_entropy + system_entropy) for cascade trajs

    (always using the independent, non-overlapping logic).

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
    scale_subsample_step : int
        Step size for subsampling scales.
    taylor_scale : float
        Taylor length scale (used to normalice other scales).
    taylor_hyp_vel : float
        Valocity used for distance to time conversion.

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
    scale_subsample_step : int
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

    # Use dimensionless scales.
    largest_scale_dimless = largest_scale / taylor_scale
    smallest_scale_dimless = smallest_scale / taylor_scale

    num_points = int(np.ceil(1.1 * largest_scale * fs / taylor_hyp_vel))
    all_indep_scales = (np.arange(num_points) * taylor_hyp_vel) / (fs * taylor_scale)

    # Find indices closest to the target scales.
    # For largest_scale, we want the last occurrence of the minimum difference.
    largest_index = all_indep_scales.size - np.argmin(
        np.abs(all_indep_scales[::-1] - largest_scale_dimless)
    )
    # For smallest_scale, we take the first occurrence.
    smallest_index = np.argmin(np.abs(all_indep_scales - smallest_scale_dimless))

    trajectory_chunk_length = largest_index

    # Take only the independen scales between the smallest and the largest scale.
    indep_scales_idxs = np.arange(smallest_index, largest_index + 1)[::-1]
    indep_scales = all_indep_scales[indep_scales_idxs]

    # Independent (non-overlapping) increments
    indep_incs_for_all_scales, incs_start_idxs = get_non_overlap_idep_incs(
        data, indep_scales_idxs, trajectory_chunk_length
    )

    # Indepentent increments for scales separated by (problably) one markovian step
    sampled_scales = indep_scales[::scale_subsample_step]
    sampled_incs = indep_incs_for_all_scales[:, ::scale_subsample_step]

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

    entropies = Entropies(
        medium_entropy=medium_entropy,
        system_entropy=system_entropy,
        total_entropy=total_entropy,
    )

    # Lagrnagian and hamiltonian quantities
    (
        lagrangian,
        action,
        momentum,
        hamiltonian,
        hamiltonian_derivative1,
        hamiltonian_derivative2,
    ) = compute_path_integral(
        incs_deriv_central,
        incs_central,
        scales_central,
        scale_spacing_central,
        km_coeffs,
    )

    return (
        entropies,
        indep_scales,
        incs_start_idxs,
        indep_scales_idxs,
        scale_subsample_step,
        indep_incs_for_all_scales,
        lagrangian,
        action,
        momentum,
        hamiltonian,
        hamiltonian_derivative1,
        hamiltonian_derivative2,
    )


def get_non_overlap_idep_incs(
    data: np.ndarray,
    indep_scales_idxs: np.ndarray,
    trajectory_chunk_length: int,
):
    # TODO: Generalize this to data with other shapes
    data = data.flatten()
    data_len = len(data)
    step_size = trajectory_chunk_length + 1
    starts = np.arange(0, data_len, step_size)

    valid_1 = (starts + indep_scales_idxs[0]) < data_len
    valid_2 = (starts + indep_scales_idxs[-1]) < data_len
    valid_starts = starts[valid_1 & valid_2]

    indep_incs_for_all_scales = (
        data[valid_starts[:, None] + indep_scales_idxs] - data[valid_starts, None]
    )
    return indep_incs_for_all_scales, valid_starts


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


def plot_entropy_pdfs(entropies, nbins):
    medium_entropy = entropies.medium_entropy
    system_entropy = entropies.system_entropy
    total_entropy = entropies.total_entropy

    N = len(total_entropy)
    NN = 20
    sample_sizes = np.round(np.logspace(1, np.log10(N), NN)).astype(int)
    ft_values = np.empty(NN)
    ft_errors = np.empty(NN)
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
    ax1.plot(centers_tot, pdf_tot, "k", label=r"$\Delta S_{\mathrm{tot}}$", linewidth=1.5)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.4)
    ax1.set_xlabel("Entropy")
    ax1.set_ylabel("PDF")
    mean_ds = np.nanmean(total_entropy)
    ax1.set_title(r"$\langle \Delta S_{\mathrm{tot}}\rangle = $" f"{mean_ds:0.2f}")
    ax1.axvline(0, color="k", linestyle="-")
    ax1.legend()


    # Plot the errorbar plot on the first axis
    ax2.errorbar(
        sample_sizes,
        ft_values,
        yerr=ft_errors,
        fmt="ok",
        linewidth=1.2,
        label=r"$\langle e^{-\Delta S_{\mathrm{tot}}}\rangle_N$",
    )
    ax2.axhline(1.0, color="k", linestyle="--", linewidth=1.0, label="IFT = 1")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.4)
    ax2.set_xlabel(r"$N$")
    ax2.set_ylabel(r"$\langle e^{-\Delta S_{\mathrm{tot}}}\rangle_N$")
    mean_val = np.nanmean(np.exp(-total_entropy))
    ax2.set_title(r"$\langle e^{-\Delta S_{\mathrm{tot}}}\rangle_{\max(N)} =$ " f"{mean_val:0.2f}", fontsize=12)
    ax2.legend()

    plt.show()
