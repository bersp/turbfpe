from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from ..utils.logger_setup import logger
from ..utils.mpl_utils import mpl_setup, save_fig
from ..utils.parameters_utils import Params, trim_data
from ..utils.storing_clases import Entropies, KMCoeffs, KMCoeffsEstimationGroup
from .entropy_computation import compute_entropy
from .ift_and_dft_optimizations import (
    compute_km_coeffs_ift_opti,
    compute_km_coeffs_dft_opti,
)
from .plot_functions import plot_entropy


def exec_routine(params_file):
    params = Params(params_file)

    mpl_setup(params)

    data = params.load_data()

    for func_str in params.read("routine.part3_entropy"):
        logger.info("-" * 80)
        logger.info(f"----- START {func_str} (PART 3)")

        func = globals()[f"{func_str}_params"]
        func(data, params)

        logger.info(f"----- END {func_str} (PART 3)")
        logger.info("-" * 80)


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


# Compute entropy

def compute_entropy_stp_opti_params(data, params):
    _compute_entropy_params(data, params, opti_type="stp")


def compute_entropy_ift_opti_params(data, params):
    _compute_entropy_params(data, params, opti_type="ift")


def compute_entropy_dft_opti_params(data, params):
    _compute_entropy_params(data, params, opti_type="dft")


def _compute_entropy_params(data, params, opti_type):
    km_coeffs = KMCoeffs.load_npz(
        params.format_output_filename_for_data(f"km_coeffs_{opti_type}_opti.npz")
    )

    fs = params.read("general.fs")
    smallest_scale = params.read("p3.general.smallest_scale")
    largest_scale = params.read("p3.general.largest_scale")
    scale_subsample_step_us = params.read("p3.general.scale_subsample_step_us")
    taylor_scale = params.read("general.taylor_scale")
    taylor_hyp_vel = params.read("general.taylor_hyp_vel")
    overlap_trajs_flag = params.read("p3.general.overlap_trajs_flag")
    available_ram_gb = params.read("p3.general.available_ram_gb")

    if overlap_trajs_flag:
        prop_to_use = params.read("p3.general.prop_to_use")
        data = trim_data(data, prop_to_use)
        if not np.all(np.array(prop_to_use) == 1.0):
            logger.info(f"Trimmed data. The new shape is {data.shape}")

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
        params.format_output_filename_for_data(f"entropies_{opti_type}_opti.npz")
    )


# KM coeffs optimization

def compute_km_coeffs_ift_opti_params(data, params):
    prop_to_use = params.read("p3.general.prop_to_use")
    data = trim_data(data, prop_to_use)
    if not np.all(np.array(prop_to_use) == 1.0):
        logger.info(f"Trimmed data. The new shape is {data.shape}")

    km_coeffs_stp_opti = KMCoeffs.load_npz(
        params.format_output_filename_for_data("km_coeffs_stp_opti.npz")
    )

    km_coeffs_est_group = KMCoeffsEstimationGroup.load_npz(
        params.format_output_filename_for_data("km_coeffs_estimation.npz")
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
        available_ram_gb,
    )

    km_coeffs_ift_opti.write_npz(
        params.format_output_filename_for_data("km_coeffs_ift_opti.npz")
    )


def compute_km_coeffs_dft_opti_params(data, params):
    prop_to_use = params.read("p3.general.prop_to_use")
    data = trim_data(data, prop_to_use)
    if not np.all(np.array(prop_to_use) == 1.0):
        logger.info(f"Trimmed data. The new shape is {data.shape}")

    km_coeffs_stp_opti = KMCoeffs.load_npz(
        params.format_output_filename_for_data("km_coeffs_stp_opti.npz")
    )

    km_coeffs_est_group = KMCoeffsEstimationGroup.load_npz(
        params.format_output_filename_for_data("km_coeffs_estimation.npz")
    )
    scales_for_optimization = np.array([
        km_coeffs_est.scale for km_coeffs_est in km_coeffs_est_group
    ])

    tol_D1 = params.read("p3.compute_km_coeffs_dft_opti.tol_D1")
    tol_D2 = params.read("p3.compute_km_coeffs_dft_opti.tol_D2")
    iter_max = params.read("p3.compute_km_coeffs_dft_opti.iter_max")
    fs = params.read("general.fs")
    smallest_scale = params.read("p3.general.smallest_scale")
    largest_scale = params.read("p3.general.largest_scale")
    scale_subsample_step_us = params.read("p3.general.scale_subsample_step_us")
    taylor_scale = params.read("general.taylor_scale")
    taylor_hyp_vel = params.read("general.taylor_hyp_vel")
    available_ram_gb = params.read("p3.general.available_ram_gb")

    km_coeffs_dft_opti, _ = compute_km_coeffs_dft_opti(
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
        available_ram_gb,
    )

    km_coeffs_dft_opti.write_npz(
        params.format_output_filename_for_data("km_coeffs_dft_opti.npz")
    )


# Plot functions

def plot_entropy_stp_opti_params(_, params):
    _plot_entropy_params(_, params, opti_type="stp")


def plot_entropy_ift_opti_params(_, params):
    _plot_entropy_params(_, params, opti_type="ift")


def plot_entropy_dft_opti_params(_, params):
    _plot_entropy_params(_, params, opti_type="dft")


def _plot_entropy_params(_, params, opti_type):
    entropies = Entropies.load_npz(
        params.format_output_filename_for_data(f"entropies_{opti_type}_opti.npz")  #
    )
    nbins = params.read(f"p3.plot_entropy_{opti_type}_opti.nbins")

    out = plot_entropy(entropies, nbins)

    if params.read("config.mpl.save_figures"):
        save_fig(
            params.format_output_filename_for_figures(
                f"p3_entropies_{opti_type}_opti.pdf"
            )
        )
    if params.read("config.mpl.show_figures"):
        plt.show()
    else:
        plt.close()

    return out
