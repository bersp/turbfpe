import matplotlib.pyplot as plt
import numpy as np

from ..utils.logger_setup import logger
from ..utils.mpl_utils import mpl_setup, save_fig
from ..utils.parameters_utils import Params
from .turbulence_analysis_functions import (
    compute_taylor_scale,
    plot_pdf,
    plot_spectrum,
    plot_stationary,
)


def exec_rutine(params_file):
    params = Params(params_file)

    mpl_setup(params)

    data = params.load_data(flat=True, ignore_opts=True)

    for func_str in params.read("rutine.part1_turbulence_analysis"):
        logger.info("-" * 80)
        logger.info(f"----- START {func_str} (PART 1)")

        func = globals()[f"{func_str}_params"]
        func(data, params)

        logger.info(f"----- END {func_str} (PART 1)")
        logger.info("-" * 80)


def compute_turbulence_analysis_autovalues_params(data, params):
    if params.is_auto("general.nbins"):
        data_range = data.max() - data.min()
        data_std = np.std(data)
        tmp = int(10 * data_range / data_std)
        params.write("general.nbins", tmp)

    if params.is_auto("p1.plot_pdf.nbins"):
        tmp = params.read("general.nbins")
        params.write("p1.plot_pdf.nbins", tmp)

    if params.is_auto("p1.plot_spectrum.moving_average_nbins"):
        tmp = 10 * params.read("general.nbins")
        params.write("p1.plot_spectrum.moving_average_nbins", tmp)


def plot_stationary_params(data, params):
    data_split_percent = params.read("p1.plot_stationary.data_split_percent")
    out = plot_stationary(data, data_split_percent)

    if params.read("config.misc.mpl.save_figures"):
        save_fig(params.format_output_filename_for_figures("p1_stationary.pdf"))
    if params.read("config.misc.mpl.show_figures"):
        plt.show()
    else:
        plt.close()

    return out


def plot_pdf_params(data, params):
    nbins = params.read("p1.plot_pdf.nbins")

    out = plot_pdf(data, nbins)

    if params.read("config.misc.mpl.save_figures"):
        save_fig(params.format_output_filename_for_figures("p1_pdf.pdf"))
    if params.read("config.misc.mpl.show_figures"):
        plt.show()
    else:
        plt.close()

    return out


def plot_spectrum_params(data, params):
    fs = params.read("general.fs")
    comp_exponent = params.read("p1.plot_spectrum.comp_exponent")
    int_scale = params.read("general.int_scale")
    taylor_scale = params.read("general.taylor_scale")
    ma_nbins = params.read("p1.plot_spectrum.moving_average_nbins")

    out = plot_spectrum(data, fs, int_scale, taylor_scale, comp_exponent, ma_nbins)

    if params.read("config.misc.mpl.save_figures"):
        save_fig(params.format_output_filename_for_figures("p1_spectrum.pdf"))
    if params.read("config.misc.mpl.show_figures"):
        plt.show()
    else:
        plt.close()

    return out


def compute_taylor_scale_params(data, params):
    fs = params.read("p1.general.fs")
    ma_nbins = params.read("p1.compute_taylor_scale.moving_average_nbins")
    return compute_taylor_scale(data, fs, ma_nbins)
