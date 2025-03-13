import numpy as np

from ..utils.mpl_utils import mpl_setup
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


def exec_rutine(params_file):
    params = Params(params_file)

    mpl_setup(params)

    data = params.load_data()

    for func in params.read("rutine.part2_markov"):
        func = globals()[f"{func}_params"]
        func(data, params)


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


def compute_wilcoxon_test_params(data, params):
    fs = params.read("general.fs")
    taylor_hyp_vel = params.read("general.taylor_hyp_vel")
    nbins = params.read("p2.compute_wilcoxon_test.nbins")
    indep_scale = params.read("p2.compute_wilcoxon_test.indep_scale")
    n_interv_sec = params.read("p2.compute_wilcoxon_test.n_interv_sec")

    delta_arr, wt_arr = compute_wilcoxon_test(
        data, fs, nbins, taylor_hyp_vel, indep_scale, n_interv_sec
    )

    np.save(
        params.format_output_filename("wilcoxon_test.npy"),
        np.vstack([delta_arr, wt_arr]),
    )


def plot_wilcoxon_test_params(data, params):
    markov_scale_us = params.read("p2.general.markov_scale_us")

    delta_arr, *wt_arr = np.load(params.format_output_filename("wilcoxon_test.npy"))
    wt_arr = np.mean(wt_arr, axis=0)

    plot_wilcoxon_test(data, delta_arr, wt_arr, markov_scale_us)


def compute_conditional_moments_estimation_params(data, params):
    fs = params.read("general.fs")
    markov_scale_us = params.read("p2.general.markov_scale_us")
    highest_freq = params.read("general.highest_freq")
    nbins = params.read("p2.general.nbins")
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
        int_scale,
        taylor_scale,
        n_scale_steps,
        taylor_hyp_vel,
    )

    cond_moments_group.write_npz(
        params.format_output_filename("conditional_moments_estimation.npz")
    )

    density_funcs_group.write_npz(
        params.format_output_filename("density_functions.npz")
    )


def compute_km_coeffs_estimation_params(_, params):
    cond_moments_group = ConditionalMomentsGroup.load_npz(
        params.format_output_filename("conditional_moments_estimation.npz")
    )

    density_funcs_group = DensityFunctionsGroup.load_npz(
        params.format_output_filename("density_functions.npz")
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
        params.format_output_filename("km_coeffs_estimation.npz")  #
    )


def plot_km_coeffs_estimation_params(_, params):
    density_funcs_group = DensityFunctionsGroup.load_npz(
        params.format_output_filename("density_functions.npz")
    )

    km_coeffs_est_group = KMCoeffsEstimationGroup.load_npz(
        params.format_output_filename("km_coeffs_estimation.npz")
    )

    taylor_scale = params.read("general.taylor_scale")

    plot_km_coeffs_estimation(km_coeffs_est_group, density_funcs_group, taylor_scale)


def compute_km_coeffs_estimation_stp_opti_params(data, params):
    km_coeffs_est_group = KMCoeffsEstimationGroup.load_npz(
        params.format_output_filename("km_coeffs_estimation.npz")
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
        params.format_output_filename("km_coeffs_estimation.npz")  #
    )


def plot_km_coeffs_estimation_opti_params(_, params):
    km_coeffs_est_group = KMCoeffsEstimationGroup.load_npz(
        params.format_output_filename("km_coeffs_estimation.npz")
    )

    density_funcs_group = DensityFunctionsGroup.load_npz(
        params.format_output_filename("density_functions.npz")
    )

    fs = params.read("general.fs")
    markov_scale_us = params.read("p2.general.markov_scale_us")
    nbins = params.read("p2.general.nbins")
    taylor_scale = params.read("general.taylor_scale")
    taylor_hyp_vel = params.read("general.taylor_hyp_vel")

    plot_km_coeffs_estimation_opti(
        km_coeffs_est_group,
        density_funcs_group,
        fs,
        markov_scale_us,
        nbins,
        taylor_scale,
        taylor_hyp_vel,
    )


def compute_km_coeffs_fit_params(_, params):
    density_funcs_group = DensityFunctionsGroup.load_npz(
        params.format_output_filename("density_functions.npz")
    )

    km_coeffs_est_group = KMCoeffsEstimationGroup.load_npz(
        params.format_output_filename("km_coeffs_estimation.npz")
    )

    nbins = params.read("p2.general.nbins")
    taylor_scale = params.read("general.taylor_scale")

    km_coeffs, km_coeffs_no_opti = compute_km_coeffs_fit(
        density_funcs_group, km_coeffs_est_group, nbins, taylor_scale
    )

    km_coeffs.write_npz(params.format_output_filename("km_coeffs.npz"))
    km_coeffs_no_opti.write_npz(params.format_output_filename("km_coeffs_no_opti.npz"))


def plot_km_coeffs_fit_params(_, params):
    density_funcs_group = DensityFunctionsGroup.load_npz(
        params.format_output_filename("density_functions.npz")
    )

    km_coeffs_est_group = KMCoeffsEstimationGroup.load_npz(
        params.format_output_filename("km_coeffs_estimation.npz")
    )

    km_coeffs = KMCoeffs.load_npz(params.format_output_filename("km_coeffs.npz"))

    nbins = params.read("p2.general.nbins")
    taylor_scale = params.read("general.taylor_scale")

    plot_km_coeffs_fit(
        km_coeffs,
        density_funcs_group,
        km_coeffs_est_group,
        nbins,
        taylor_scale,
    )
