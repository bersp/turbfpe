import numpy as np

from turbfpe.utils.storing_clases import ConditionalMomentsGroup, DensityFunctionsGroup

from ..utils.mpl_utils import mpl_setup
from ..utils.parameters_utils import Params
from .conditional_moments_estimation import compute_conditional_moments_estimation
from .wilcoxon_test import compute_wilcoxon_test, plot_wilcoxon_test
from .km_coeffs_estimation import compute_km_coeffs_estimation


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
        tmp = params.read("general.nbins")
        params.write("p2.compute_wilcoxon_test.nbins", tmp)

    if params.is_auto("p2.compute_wilcoxon_test.indep_scale"):
        indep_scale = params.read("general.int_scale")
        fs = params.read("general.fs")
        taylor_hyp_vel = params.read("general.taylor_hyp_vel")
        indep_scale_in_samples = round(fs * indep_scale / taylor_hyp_vel)
        n_interv = data.shape[1] // indep_scale_in_samples - 1
        while n_interv < 5:
            indep_scale = 0.9 * indep_scale
            indep_scale_in_samples = round(fs * indep_scale / taylor_hyp_vel)
            n_interv = data.shape[1] // indep_scale_in_samples - 1
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

    output_filename = params.format_output_filename("wilcoxon_test.npy")
    np.save(output_filename, np.vstack([delta_arr, wt_arr]))


def plot_wilcoxon_test_params(_, params):
    markov_scale_us = params.read("p2.general.markov_scale_us")
    output_filename = params.format_output_filename("wilcoxon_test.npy")
    plot_wilcoxon_test(output_filename, markov_scale_us)


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
    taylor_scale = params.read("general.taylor_scale")
    taylor_hyp_vel = params.read("general.taylor_hyp_vel")
    nbins = params.read("p2.general.nbins")

    km_coeffs_estimation = compute_km_coeffs_estimation(
        cond_moments_group,
        density_funcs_group,
        fs,
        markov_scale_us,
        nbins,
        taylor_scale,
        taylor_hyp_vel,
    )

    km_coeffs_estimation.write_npz(
        params.format_output_filename("km_coeffs_estimation.npz")
    )
