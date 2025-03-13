from __future__ import print_function
import numpy as np

from ..utils.mpl_utils import mpl_setup
from ..utils.parameters_utils import Params
from ..utils.storing_clases import Entropies, KMCoeffs, KMCoeffsEstimationGroup
from .entropy_computation import (
    compute_entropy,
    compute_km_coeffs_ift_opti,
    plot_entropy_and_ift,
)


def exec_rutine(params_file):
    params = Params(params_file)

    mpl_setup(params)

    data = params.load_data()

    for func in params.read("rutine.part3_entropy"):
        func = globals()[f"{func}_params"]
        func(data, params)


def compute_entropy_autovalues_params(_, params):
    if params.is_auto("p3.general.smallest_scale"):
        tmp = params.read("general.taylor_scale")
        params.write("p3.general.smallest_scale", tmp)

    if params.is_auto("p3.general.largest_scale"):
        tmp = params.read("general.int_scale")
        params.write("p3.general.largest_scale", tmp)

    if params.is_auto("p3.general.scale_subsample_step_us"):
        tmp = params.read("p2.general.markov_scale_us")
        params.write("p3.general.scale_subsample_step_us", tmp)


def _compute_entropy_params(data, params, opti_type):
    km_coeffs = KMCoeffs.load_npz(
        params.format_output_filename(f"km_coeffs_{opti_type}_opti.npz")
    )

    fs = params.read("general.fs")
    smallest_scale = params.read("p3.general.smallest_scale")
    largest_scale = params.read("p3.general.largest_scale")
    scale_subsample_step_us = params.read("p3.general.scale_subsample_step_us")
    taylor_scale = params.read("general.taylor_scale")
    taylor_hyp_vel = params.read("general.taylor_hyp_vel")
    overlap_trajs_flag = params.read("p3.general.overlap_trajs_flag")
    available_ram_gb = params.read("p3.general.available_ram_gb")

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

    entropies.write_npz(
        params.format_output_filename(f"entropies_{opti_type}_opti.npz")
    )


def compute_entropy_stp_opti_params(data, params):
    _compute_entropy_params(data, params, opti_type="stp")


def compute_entropy_ift_opti_params(data, params):
    _compute_entropy_params(data, params, opti_type="ift")


def compute_km_coeffs_ift_opti_params(data, params):
    km_coeffs_stp_opti = KMCoeffs.load_npz(
        params.format_output_filename(f"km_coeffs_stp_opti.npz")
    )

    km_coeffs_est_group = KMCoeffsEstimationGroup.load_npz(
        params.format_output_filename(f"km_coeffs_estimation.npz")
    )
    scales_for_optimization = np.array([
        km_coeffs_est.scale for km_coeffs_est in km_coeffs_est_group
    ])

    tol_D1 = params.read("p3.compute_km_coeffs_ift_opti.tol_D1")
    tol_D2 = params.read("p3.compute_km_coeffs_ift_opti.tol_D2")
    iter_max = params.read("p3.compute_km_coeffs_ift_opti.iter_max")
    fs = params.read("general.fs")
    smallest_scale = params.read("p3.general.smallest_scale")
    largest_scale = params.read("p3.general.largest_scale")
    scale_subsample_step_us = params.read("p3.general.scale_subsample_step_us")
    taylor_scale = params.read("general.taylor_scale")
    taylor_hyp_vel = params.read("general.taylor_hyp_vel")
    overlap_trajs_flag = params.read("p3.general.overlap_trajs_flag")
    available_ram_gb = params.read("p3.general.available_ram_gb")

    km_coeffs_ift_opti, _ = compute_km_coeffs_ift_opti(
        data,
        km_coeffs_stp_opti,
        scales_for_optimization,
        tol_D1,
        tol_D2,
        iter_max,
        fs,
        smallest_scale,
        largest_scale,
        scale_subsample_step_us,
        taylor_scale,
        taylor_hyp_vel,
        overlap_trajs_flag,
        available_ram_gb,
    )

    km_coeffs_ift_opti.write_npz(params.format_output_filename("km_coeffs_ift_opti.npz"))


def _plot_entropy_and_ift_params(_, params, opti_type):
    entropies = Entropies.load_npz(
        params.format_output_filename(f"entropies_{opti_type}_opti.npz")  #
    )
    nbins = params.read("p3.plot_entropy_and_ift.nbins")
    plot_entropy_and_ift(entropies, nbins)


def plot_entropy_and_ift_for_stp_opti_params(_, params):
    _plot_entropy_and_ift_params(_, params, opti_type="stp")


def plot_entropy_and_ift_for_ift_opti_params(_, params):
    _plot_entropy_and_ift_params(_, params, opti_type="ift")
