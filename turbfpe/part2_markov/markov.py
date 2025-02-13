import os

from .markov_functions import compute_wilcoxon_test, plot_wilcoxon_test

from ..utils.mpl_utils import *
from ..utils.parameters_utils import Params


def exec_rutine(params_file):
    params = Params(params_file)
    data = params.load_data()

    for func in params.read("rutine.part2_markov"):
        func = globals()[f"{func}_params"]
        func(data, params)


def compute_markov_autovalues_params(data, params):
    if params.is_auto("p2.general.EM_scale"):
        tmp = (
            params.read("general.taylor_scale")
            * params.read("general.fs")
            / params.read("general.taylor_hyp_vel")
        )
        params.write("p2.general.EM_scale", int(tmp))

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

    output_filename = params.format_output_filename("wilcoxon_test.npy")

    compute_wilcoxon_test(
        data, fs, nbins, taylor_hyp_vel, indep_scale, n_interv_sec, output_filename
    )


def plot_wilcoxon_test_params(_, params):
    EM_scale = params.read("p2.general.EM_scale")
    output_filename = params.format_output_filename("wilcoxon_test.npy")
    plot_wilcoxon_test(output_filename, EM_scale)
