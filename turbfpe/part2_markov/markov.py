import matplotlib.pyplot as plt
import numpy as np

from ..utils.logger_setup import logger
from ..utils.mpl_utils import mpl_setup, save_fig
from ..utils.parameters_utils import Params
from ..utils.storing_clases import (
    ConditionalMomentsGroup,
    DensityFunctionsGroup,
    KMCoeffs,
    KMCoeffsEstimationGroup,
)
from .conditional_moments_estimation import compute_conditional_moments_estimation
from .km_coeffs_estimation import (
    compute_km_coeffs_estimation,
)
from .km_coeffs_estimation_stp_opti import compute_km_estimation_stp_opti
from .km_coeffs_fit import compute_km_coeffs_fit
from .plot_km_coeffs import (
    plot_km_coeffs_estimation,
    plot_km_coeffs_estimation_opti,
    plot_km_coeffs_fit,
)
from .wilcoxon_test import compute_wilcoxon_test, plot_wilcoxon_test


def exec_routine(params_file):
    params = Params(params_file)

    mpl_setup(params)

    data = params.load_data()

    for func_str in params.read("routine.part2_markov"):
        logger.info("-" * 80)
        logger.info(f"----- START {func_str} (PART 2)")

        func = globals()[f"{func_str}_params"]
        func(data, params)

        logger.info(f"----- END {func_str} (PART 2)")
        logger.info("-" * 80)


def compute_markov_autovalues_params(data, params):
    if params.is_auto("p2.general.markov_scale_us"):
        tmp = (
            params.read("general.taylor_scale")
            * params.read("general.fs")
            / params.read("general.taylor_hyp_vel")
        )
        params.write("p2.general.markov_scale_us", int(tmp))

    if params.is_auto("p2.general.nbins"):
        tmp = params.read("general.nbins")
        params.write("p2.general.nbins", tmp)

    if params.is_auto("p2.compute_wilcoxon_test.nbins"):
        tmp = params.read("p2.general.nbins")
        params.write("p2.compute_wilcoxon_test.nbins", tmp)

    if params.is_auto("p2.compute_wilcoxon_test.indep_scale"):
        to_us = params.read("general.fs") / params.read("general.taylor_hyp_vel")
        indep_scale = params.read("general.int_scale")
        indep_scale_us = round(indep_scale * to_us)
        n_interv = data.shape[1] // indep_scale_us - 1
        while n_interv < 5:
            indep_scale = 0.95 * indep_scale
            indep_scale_us = round(indep_scale * to_us)
            n_interv = data.shape[1] // indep_scale_us - 1
        tmp = indep_scale
        params.write("p2.compute_wilcoxon_test.indep_scale", tmp)

    if params.is_auto("p2.compute_wilcoxon_test.end_scale"):
        tmp = params.read("p2.compute_wilcoxon_test.indep_scale")
        params.write("p2.compute_wilcoxon_test.end_scale", tmp)


def compute_wilcoxon_test_params(data, params):
    fs = params.read("general.fs")
    taylor_hyp_vel = params.read("general.taylor_hyp_vel")
    nbins = params.read("p2.compute_wilcoxon_test.nbins")
    indep_scale = params.read("p2.compute_wilcoxon_test.indep_scale")
    end_scale = params.read("p2.compute_wilcoxon_test.end_scale")
    n_interv_sec = params.read("p2.compute_wilcoxon_test.n_interv_sec")

    delta_arr, wt_arr = compute_wilcoxon_test(
        data, fs, nbins, taylor_hyp_vel, indep_scale, end_scale, n_interv_sec
    )

    np.save(
        params.format_output_filename_for_data("wilcoxon_test.npy"),
        np.vstack([delta_arr, wt_arr]),
    )


def plot_wilcoxon_test_params(data, params):
    markov_scale_us = params.read("p2.general.markov_scale_us")

    delta_arr, *wt_arr = np.load(
        params.format_output_filename_for_data("wilcoxon_test.npy")
    )

    out = plot_wilcoxon_test(data, delta_arr, wt_arr, markov_scale_us)

    if params.read("config.mpl.save_figures"):
        save_fig(params.format_output_filename_for_figures("p2_wilcoxon_test.pdf"))
    if params.read("config.mpl.show_figures"):
        plt.show()
    else:
        plt.close()

    return out


def compute_conditional_moments_estimation_params(data, params):
    fs = params.read("general.fs")
    markov_scale_us = params.read("p2.general.markov_scale_us")
    highest_freq = params.read("general.highest_freq")
    nbins = params.read("p2.general.nbins")
    min_events = params.read("p2.general.min_events")
    int_scale = params.read("general.int_scale")
    taylor_scale = params.read("general.taylor_scale")
    n_scale_steps = params.read(
        "p2.compute_conditional_moments_estimation.n_scale_steps"
    )
    taylor_hyp_vel = params.read("general.taylor_hyp_vel")

    cond_moments_group, density_funcs_group = compute_conditional_moments_estimation(
        data,
        fs,
        highest_freq,
        markov_scale_us,
        nbins,
        min_events,
        int_scale,
        taylor_scale,
        n_scale_steps,
        taylor_hyp_vel,
    )

    cond_moments_group.write_npz(
        params.format_output_filename_for_data("conditional_moments_estimation.npz")
    )

    density_funcs_group.write_npz(
        params.format_output_filename_for_data("density_functions.npz")
    )


def compute_km_coeffs_estimation_params(_, params):
    cond_moments_group = ConditionalMomentsGroup.load_npz(
        params.format_output_filename_for_data("conditional_moments_estimation.npz")
    )

    density_funcs_group = DensityFunctionsGroup.load_npz(
        params.format_output_filename_for_data("density_functions.npz")
    )

    fs = params.read("general.fs")
    markov_scale_us = params.read("p2.general.markov_scale_us")
    nbins = params.read("p2.general.nbins")
    min_events = params.read("p2.general.min_events")
    taylor_scale = params.read("general.taylor_scale")
    taylor_hyp_vel = params.read("general.taylor_hyp_vel")

    km_coeffs_est_group = compute_km_coeffs_estimation(
        cond_moments_group,
        density_funcs_group,
        fs,
        markov_scale_us,
        nbins,
        min_events,
        taylor_scale,
        taylor_hyp_vel,
    )

    km_coeffs_est_group.write_npz(
        params.format_output_filename_for_data("km_coeffs_estimation.npz")
    )


def plot_km_coeffs_estimation_params(_, params):
    density_funcs_group = DensityFunctionsGroup.load_npz(
        params.format_output_filename_for_data("density_functions.npz")
    )

    km_coeffs_est_group = KMCoeffsEstimationGroup.load_npz(
        params.format_output_filename_for_data("km_coeffs_estimation.npz")
    )

    taylor_scale = params.read("general.taylor_scale")

    out = plot_km_coeffs_estimation(km_coeffs_est_group, density_funcs_group, taylor_scale)

    if params.read("config.mpl.save_figures"):
        save_fig(params.format_output_filename_for_figures("p2_km_coeffs_estimation.pdf"))
    if params.read("config.mpl.show_figures"):
        plt.show()
    else:
        plt.close()

    return out


def compute_km_coeffs_estimation_stp_opti_params(data, params):
    km_coeffs_est_group = KMCoeffsEstimationGroup.load_npz(
        params.format_output_filename_for_data("km_coeffs_estimation.npz")
    )

    fs = params.read("general.fs")
    nbins = params.read("p2.general.nbins")
    tol = params.read("p2.compute_km_coeffs_estimation_stp_opti.tol")
    taylor_scale = params.read("general.taylor_scale")
    taylor_hyp_vel = params.read("general.taylor_hyp_vel")

    km_coeffs_est_group = compute_km_estimation_stp_opti(
        data,
        km_coeffs_est_group,
        fs,
        nbins,
        tol,
        taylor_scale,
        taylor_hyp_vel,
    )

    km_coeffs_est_group.write_npz(
        params.format_output_filename_for_data("km_coeffs_estimation.npz")
    )


def plot_km_coeffs_estimation_opti_params(_, params):
    km_coeffs_est_group = KMCoeffsEstimationGroup.load_npz(
        params.format_output_filename_for_data("km_coeffs_estimation.npz")
    )

    density_funcs_group = DensityFunctionsGroup.load_npz(
        params.format_output_filename_for_data("density_functions.npz")
    )

    fs = params.read("general.fs")
    markov_scale_us = params.read("p2.general.markov_scale_us")
    nbins = params.read("p2.general.nbins")
    taylor_scale = params.read("general.taylor_scale")
    taylor_hyp_vel = params.read("general.taylor_hyp_vel")

    out = plot_km_coeffs_estimation_opti(
        km_coeffs_est_group,
        density_funcs_group,
        fs,
        markov_scale_us,
        nbins,
        taylor_scale,
        taylor_hyp_vel,
    )

    if params.read("config.mpl.save_figures"):
        save_fig(params.format_output_filename_for_figures("p2_km_coeffs_estimation_opti.pdf"))
    if params.read("config.mpl.show_figures"):
        plt.show()
    else:
        plt.close()

    return out


def compute_km_coeffs_fit_params(_, params):
    density_funcs_group = DensityFunctionsGroup.load_npz(
        params.format_output_filename_for_data("density_functions.npz")
    )

    km_coeffs_est_group = KMCoeffsEstimationGroup.load_npz(
        params.format_output_filename_for_data("km_coeffs_estimation.npz")
    )

    nbins = params.read("p2.general.nbins")
    taylor_scale = params.read("general.taylor_scale")

    km_coeffs, km_coeffs_no_opti = compute_km_coeffs_fit(
        density_funcs_group, km_coeffs_est_group, nbins, taylor_scale
    )

    km_coeffs.write_npz(
        params.format_output_filename_for_data("km_coeffs_stp_opti.npz")
    )
    km_coeffs_no_opti.write_npz(
        params.format_output_filename_for_data("km_coeffs_no_opti.npz")
    )


def plot_km_coeffs_fit_params(_, params):
    density_funcs_group = DensityFunctionsGroup.load_npz(
        params.format_output_filename_for_data("density_functions.npz")
    )

    km_coeffs_est_group = KMCoeffsEstimationGroup.load_npz(
        params.format_output_filename_for_data("km_coeffs_estimation.npz")
    )

    km_coeffs = KMCoeffs.load_npz(
        params.format_output_filename_for_data("km_coeffs_stp_opti.npz")
    )

    nbins = params.read("p2.general.nbins")
    taylor_scale = params.read("general.taylor_scale")

    out = plot_km_coeffs_fit(
        km_coeffs,
        density_funcs_group,
        km_coeffs_est_group,
        nbins,
        taylor_scale,
    )

    if params.read("config.mpl.save_figures"):
        save_fig(params.format_output_filename_for_figures("p2_km_coeffs_fit.pdf"))
    if params.read("config.mpl.show_figures"):
        plt.show()
    else:
        plt.close()

    return out
