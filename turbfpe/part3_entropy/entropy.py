from ..utils.storing_clases import Entropies, KMCoeffs
from ..utils.mpl_utils import mpl_setup
from ..utils.parameters_utils import Params
from .entropy_computation import compute_entropy, plot_entropy_pdfs


def exec_rutine(params_file):
    params = Params(params_file)

    mpl_setup(params)

    data = params.load_data()

    for func in params.read("rutine.part3_entropy"):
        func = globals()[f"{func}_params"]
        func(data, params)


def compute_entropy_autovalues_params(_, params):
    if params.is_auto("p3.compute_entropy.smallest_scale"):
        tmp = params.read("general.taylor_scale")
        params.write("p3.compute_entropy.smallest_scale", tmp)

    if params.is_auto("p3.compute_entropy.largest_scale"):
        tmp = params.read("general.int_scale")
        params.write("p3.compute_entropy.largest_scale", tmp)

    if params.is_auto("p3.compute_entropy.scale_subsample_step_us"):
        tmp = params.read("p2.general.markov_scale_us")
        params.write("p3.compute_entropy.scale_subsample_step_us", tmp)


def compute_entropy_params(data, params):
    km_coeffs = KMCoeffs.load_npz(params.format_output_filename("km_coeffs.npz"))

    fs = params.read("general.fs")
    smallest_scale = params.read("p3.compute_entropy.smallest_scale")
    largest_scale = params.read("p3.compute_entropy.largest_scale")
    scale_subsample_step_us = params.read("p3.compute_entropy.scale_subsample_step_us")
    taylor_scale = params.read("general.taylor_scale")
    taylor_hyp_vel = params.read("general.taylor_hyp_vel")
    overlap_trajs_flag = params.read('p3.compute_entropy.overlap_trajs_flag')
    available_ram_gb = params.read('p3.compute_entropy.available_ram_gb')

    entropies, *_ = compute_entropy(
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

    entropies.write_npz(params.format_output_filename("entropies.npz"))


def plot_entropy_pdfs_params(_, params):
    entropies = Entropies.load_npz(
        params.format_output_filename("entropies.npz")  #
    )
    nbins = params.read("p3.plot_entropy_pdfs.nbins")
    plot_entropy_pdfs(entropies, nbins)
