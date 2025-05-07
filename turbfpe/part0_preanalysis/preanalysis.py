import numpy as np
import scipy.stats as stats

from ..utils.logger_setup import logger
from ..utils.parameters_utils import Params
from .preanalysis_functions import compute_int_scale


def exec_rutine(params_file):
    params = Params(params_file)
    data = params.load_data(flat=False, ignore_opts=True)
    params.write("data.shape", data.shape)
    data = data.compressed()

    for func_str in params.read("rutine.part0_preanalysis"):
        logger.info("-" * 80)
        logger.info(f"----- START {func_str} (PART 0)")

        func = globals()[f"{func_str}_params"]
        func(data, params)

        logger.info(f"----- END {func_str} (PART 0)")
        logger.info("-" * 80)


def compute_int_scale_params(data, params: Params):
    fs = params.read("general.fs")
    taylor_hyp_vel = params.read("general.taylor_hyp_vel")
    return compute_int_scale(data, fs, taylor_hyp_vel)


def compute_and_write_data_stats_params(data, params: Params):
    data_mean, data_std = np.mean(data), np.std(data)
    data_rms = np.sqrt(np.mean(data**2))
    data_range = data.max() - data.min()
    data_skew = stats.skew(data)
    data_kurtosis = stats.kurtosis(data, fisher=False)

    params.write("data.stats.mean", data_mean)
    params.write("data.stats.rms", data_rms)
    params.write("data.stats.std", data_std)
    params.write("data.stats.skew", data_skew)
    params.write("data.stats.kurtosis", data_kurtosis)
    params.write("data.stats.range", data_range)


def compute_and_write_general_autovalues_params(data, params: Params):
    data_std = params.read("data.stats.std")
    data_range = params.read("data.stats.range")

    # Computing these parameters might introduce more noise than actual benefits, especially since they're not used.
    # if params.is_auto("general.epsilon"):
    #     tmp = (
    #         15
    #         * params.read("general.nu")
    #         * data_std**2
    #         / params.read("general.taylor_scale") ** 2
    #     )
    #     params.write("general.epsilon", tmp)

    # if params.is_auto("general.kolmogorov_length"):
    #     tmp = (params.read("general.nu") ** 3 / params.read("general.epsilon")) ** (
    #         1 / 4
    #     )
    #     params.write("general.kolmogorov_length", tmp)

    # if params.is_auto("general.kolmogorov_time"):
    #     tmp = (params.read("general.nu") / params.read("general.epsilon")) ** (1 / 2)
    #     params.write("general.kolmogorov_time", tmp)

    if params.is_auto("general.nbins"):
        tmp = int(10 * data_range / data_std)
        params.write("general.nbins", tmp)

    if params.is_auto("general.int_scale"):
        int_scale = compute_int_scale_params(data, params)
        params.write("general.int_scale", int_scale)
