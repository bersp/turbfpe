import time

import numpy as np
import scipy.stats as stats

from ..utils.logger_setup import (
    format_routine_done_log,
    format_routine_start_log,
    logger,
)
from ..utils.parameters_utils import Params
from .preanalysis_functions import compute_int_scale


def exec_routine(params_file):
    params = Params(filename=params_file)
    data = params.load_data(flat=False, ignore_opts=True)
    params.write("data.shape", data.shape)
    data = data.compressed()

    for func_str in params.read("routine.part0_preanalysis"):
        data_name = params.read("config.io.save_filenames_prefix")
        logger.info(
            format_routine_start_log(partn=0, func_str=func_str, data_name=data_name)
        )

        t0 = time.perf_counter()
        func = globals()[f"{func_str}_params"]
        func(data, params=params)
        elapsed_time = time.perf_counter() - t0

        logger.info(
            format_routine_done_log(
                partn=0,
                func_str=func_str,
                data_name=data_name,
                elapsed_time=elapsed_time,
            )
        )


def compute_int_scale_params(data, params: Params):
    fs = params.read("general.fs")
    taylor_hyp_vel = params.read("general.taylor_hyp_vel")
    return compute_int_scale(data=data, fs=fs, taylor_hyp_vel=taylor_hyp_vel)


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

    if params.is_auto("general.nbins"):
        tmp = int(10 * data_range / data_std)
        params.write("general.nbins", tmp)

    if params.is_auto("general.int_scale"):
        int_scale = compute_int_scale_params(data=data, params=params)
        params.write("general.int_scale", int_scale)
