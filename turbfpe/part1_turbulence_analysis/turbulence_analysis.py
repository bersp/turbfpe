import numpy as np

from ..utils.parameters_utils import Params
from .turbulence_analysis_functions import (
    compute_taylor_scale,
    plot_pdf,
    plot_spectrum,
    plot_stationary,
)


def exec_rutine(params_file):
    params = Params(params_file)
    data = params.load_data(flat=True, ignore_opts=True)

    for func in params.read("rutine.part1_turbulence_analysis"):
        func = globals()[f"{func}_params"]
        func(data, params)


def compute_turbulence_analysis_autovalues_params(data, params):
    if params.is_auto("general.nbins"):
        data_range = data.max() - data.min()
        data_std = np.std(data)
        tmp = int(10 * data_range / data_std)
        params.write("general.nbins", tmp)

    if params.is_auto("p1.plot_pdf.nbins"):
        params.write("p1.plot_pdf.nbins", params.read("general.nbins"))

    if params.is_auto("p1.plot_spectrum.moving_average_nbins"):
        tmp = 10 * params.read("general.nbins")
        params.write("p1.plot_spectrum.moving_average_nbins", tmp)


def plot_stationary_params(data, params):
    data_split_percent = params.read("p1.plot_stationary.data_split_percent")
    return plot_stationary(data, data_split_percent)


def plot_pdf_params(data, params):
    nbins = params.read("p1.plot_pdf.nbins")
    return plot_pdf(data, nbins)


def plot_spectrum_params(data, params):
    fs = params.read("general.fs")
    ma_nbins = params.read("p1.plot_spectrum.moving_average_nbins")
    return plot_spectrum(data, fs, ma_nbins)


def compute_taylor_scale_params(data, params):
    fs = params.read("p1.general.fs")
    ma_nbins = params.read("p1.compute_taylor_scale.moving_average_nbins")
    return compute_taylor_scale(data, fs, ma_nbins)
