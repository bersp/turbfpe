import numpy as np
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

    kmcoeffs_est_group = KMCoeffsEstimationGroup()
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
        kmcoeffs_est_group.add(km_instance)

    return kmcoeffs_est_group


if __name__ == "__main__":
    cond_moments_group = ConditionalMomentsGroup.from_npz(
        "./processed_data/estimated_conditional_moments.npz"
    )
    density_funcs_group = DensityFunctionsGroup.from_npz(
        "./processed_data/density_functions.npz"
    )

    markov_scale_us = 22
    fs = 8000
    taylor_scale = 0.0067
    inc_bin = 93
    taylor_hyp_vel = 2.2499926832892565

    kmcoeffs_est_group = compute_km_coeffs_estimation(
        nbins=inc_bin,
        min_events=400,
        cond_moments_group=cond_moments_group,
        density_funcs_group=density_funcs_group,
        steps_con_moment_us=steps_con_moment_us,
        fs=fs,
        taylor_scale=taylor_scale,
        taylor_hyp_vel=taylor_hyp_vel,
    )
    kmcoeffs_est_group.write("./processed_data/KMCoeffsEstimation.npz")
