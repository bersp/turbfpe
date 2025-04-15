import numpy as np
import scipy.signal as signal
from tqdm import tqdm 

def low_pass_filter(data, low_freq, fs):

    data_filter = np.zeros_like(data)
    for ii, data_row in tqdm(enumerate(data), total=data.shape[0], desc="Filtering data rows"):
        order = 20
        data_filter_row = np.ones_like(data_row)*np.nan

        row_valid_size = np.sum(~np.isnan(data_row))

        data_row = data_row[:row_valid_size]

        b, a = signal.butter(order, low_freq / (fs / 2), btype="lowpass")
        
        tmp = signal.filtfilt(b, a, data_row)

        data_var = np.var(data_row)
        while (
            np.sum(np.isnan(tmp)) > 0  # avoid nan in the filtered data
            or np.sum(tmp == 0) > 0  # avoid zeros in the filtered data
            or np.var(tmp) > data_var
        ):
            order -= 1
            if order == 0:
                tmp = data_row*np.nan
                break
            b, a = signal.butter(order, low_freq / (fs / 2), btype="lowpass")
            tmp = signal.filtfilt(b, a, data_row)

        data_filter_row[:row_valid_size] = tmp
        data_filter[ii] = data_filter_row

    return data_filter
