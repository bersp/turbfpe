import numpy as np
import matplotlib.pyplot as plt

from turbfpe.utils.storing_clases import (
    ConditionalMomentsGroup,
    DensityFunctionsGroup,
    KMCoeffsEstimationGroup,
    KMCoeffsEstimation,
)


def compute_km_coeffs_estimation(
    cond_moments_group: ConditionalMomentsGroup,
    density_funcs_group: DensityFunctionsGroup,
    fs: float,
    markov_scale_us: int,
    nbins: int,
    taylor_scale: float,
    taylor_hyp_vel: float,
) -> KMCoeffsEstimationGroup:
    """
    Calculate Kramers–Moyal coefficients (D1, D2, D3, D4) for each scale.
    """

    steps_con_moment_us = np.round(
        np.linspace(markov_scale_us, 2 * markov_scale_us, 5)
    ).astype(int)
    steps_normlalization_const_us = taylor_scale * fs / taylor_hyp_vel
    steps_us = steps_con_moment_us / steps_normlalization_const_us

    km_coeffs_est_group = KMCoeffsEstimationGroup()
    for idx, cond_moments in enumerate(cond_moments_group):
        # Preallocate arrays for coefficients and error estimates per bin
        D1 = np.full(nbins, np.nan)
        D1_err = np.full(nbins, np.nan)
        D2 = np.full(nbins, np.nan)
        D2_no_correction = np.full(nbins, np.nan)
        D2_err = np.full(nbins, np.nan)
        D3 = np.full(nbins, np.nan)
        D4 = np.full(nbins, np.nan)

        for i in range(nbins):
            # Only process bins where the moment errors are positive
            # ... is this necessary?
            if np.all(cond_moments.M1_err[i, :] > 0) and np.all(
                cond_moments.M2_err[i, :] > 0
            ):
                # Compute normalized steps for fitting

                # --- Fit for D1 (intercept method) ---
                m11_normalized = cond_moments.M11[i, :] / steps_us
                weights_m11 = (1.0 / cond_moments.M1_err[i, :]) / steps_us
                fit_params_D1 = np.polyfit(steps_us, m11_normalized, 1, w=weights_m11)
                D1[i] = fit_params_D1[1]  # use the intercept

                # --- Fit for D2 with correction ---
                m21_corrected = (cond_moments.M21[i, :] - (steps_us * D1[i]) ** 2) / (
                    2 * steps_us
                )
                weights_m21 = (1.0 / cond_moments.M2_err[i, :]) / steps_us
                fit_params_D2 = np.polyfit(steps_us, m21_corrected, 1, w=weights_m21)
                D2[i] = fit_params_D2[1]  # intercept

                # --- Fit for D2 without correction (for error estimation) ---
                m21_no_corr = (cond_moments.M21[i, :] / 2) / steps_us
                weights_m21_no_corr = (1.0 / cond_moments.M2_err[i, :]) / steps_us
                fit_params_D2_no_corr = np.polyfit(
                    steps_us, m21_no_corr, 1, w=weights_m21_no_corr
                )
                D2_no_correction[i] = fit_params_D2_no_corr[1]

                # --- Fit for D3 with correction ---
                m31_corrected = (
                    cond_moments.M31[i, :]
                    - 6 * steps_us**2 * fit_params_D1[0] * fit_params_D2[0]
                ) / 6
                fit_params_D3 = np.polyfit(steps_us, m31_corrected, 1)
                D3[i] = fit_params_D3[1]  # intercept

                # --- Fit for D4 with correction ---
                m41_corrected = (
                    cond_moments.M41[i, :]
                    - 12
                    * steps_us**2
                    * (2 * fit_params_D1[0] * fit_params_D3[0] + fit_params_D2[0] ** 2)
                ) / 24
                fit_params_D4 = np.polyfit(steps_us, m41_corrected, 1)
                D4[i] = fit_params_D4[1]  # intercept

                # --- Fit for D4 without correction (for error estimation) ---
                m41_no_corr = (cond_moments.M41[i, :] / 24) / steps_us
                weights_m41 = (1.0 / cond_moments.M2_err[i, :]) / steps_us
                fit_params_D4_no_corr = np.polyfit(
                    steps_us, m41_no_corr, 1, w=weights_m41
                )
                D4_no_corr_value = fit_params_D4_no_corr[1]

                # --- Compute error estimates ---
                max_count = np.max(density_funcs_group[idx].counts0)
                D1_err[i] = np.sqrt(
                    np.abs((2 * D2_no_correction[i] - D1[i] ** 2) / max_count)
                )
                D2_err[i] = np.sqrt(
                    np.abs(
                        (2 * D4_no_corr_value - D2_no_correction[i] ** 2) / max_count
                    )
                )

        # Adjust D2 values
        D2 = D2 + np.abs(np.min(D2))
        D2[D2 < 0] = 0
        D2 = np.abs(D2)
        D2_no_correction[D2_no_correction < 0] = 0

        # Create a KMCoeffsEstimation instance with the computed coefficients and errors
        km_instance = KMCoeffsEstimation(
            scale=cond_moments.scale,
            scale_us=cond_moments.scale_us,
            scale_short_us=cond_moments.scale_short_us,
            D1=D1,
            D1_err=D1_err,
            D2=np.abs(D2_no_correction),
            D2_err=D2_err,
            D3=np.abs(D3),
            D4=np.abs(D4),
            D1_opti=np.array([]),  # Optimized coefficients not computed here
            D2_opti=np.array([]),
        )
        km_coeffs_est_group.add(km_instance)

    return km_coeffs_est_group


def km_coeffs_estimation_plot(
    km_coeffs_est_group, density_funcs_group, nbins, taylor_scale, min_events
):
    n_scales_steps = len(km_coeffs_est_group)

    counts1 = density_funcs_group.unpack("counts1")
    u_incs = density_funcs_group.unpack("mean_per_bin0")
    D1 = km_coeffs_est_group.unpack("D1")
    D2 = km_coeffs_est_group.unpack("D2")
    D3 = km_coeffs_est_group.unpack("D3")
    D4 = km_coeffs_est_group.unpack("D4")

    scales_1d = km_coeffs_est_group.unpack("scale")/taylor_scale
    scales = np.ones_like(D1)*np.nan

    valid_idxs = (counts1[0] > min_events) & ~np.isnan(D1[0])
    u_incs[:, ~valid_idxs] = np.nan
    D1[:, ~valid_idxs] = np.nan
    D2[:, ~valid_idxs] = np.nan
    D3[:, ~valid_idxs] = np.nan
    D4[:, ~valid_idxs] = np.nan
    scales[:, valid_idxs] = scales_1d[:, None]

    def scatter_3d(x, y, z, x_label, y_label, z_label):
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        mask = ~np.isnan(x_flat) & ~np.isnan(y_flat) & ~np.isnan(z_flat)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            x_flat[mask],
            y_flat[mask],
            z_flat[mask],
            c="k",
            ec="white",
            s=30,
            lw=0.5,
            alpha=0.5,
        )
        ax.invert_yaxis()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)

    x_label = r"$u_r / \sigma_\infty$"
    y_label = r"$r / \lambda$"

    scatter_3d(
        u_incs,
        scales,
        (D2**2) / D4,
        x_label,
        y_label,
        r"$(D^{(2)})^2 / D^{(4)}$",
    )

    scatter_3d(
        u_incs,
        scales,
        D4 / (D2**2),
        x_label,
        y_label,
        r"$D^{(4)} / (D^{(2)})^2$",
    )

    scatter_3d(u_incs, scales, D4, x_label, y_label, r"$D^{(4)}$")

    scatter_3d(u_incs, scales, D3, x_label, y_label, r"$D^{(3)}$")

    scatter_3d(u_incs, scales, D2, x_label, y_label, r"$D^{(2)}$")

    scatter_3d(u_incs, scales, D1, x_label, y_label, r"$D^{(1)}$")

    plt.show()
