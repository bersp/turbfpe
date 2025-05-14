import os
import re
import pprint

import numpy as np
import tomlkit


class Params:
    def __init__(self, filename):
        self.filename = filename
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File not found: {self.filename}")

        with open(self.filename, "r") as f:
            self.params = tomlkit.parse(f.read())

    def _read_raw(self, prop):
        self._raise_error_if_is_not_a_known_prop(prop)
        keys = prop.split(".")
        data = self.params
        for k in keys:
            if isinstance(data, dict) and k in data:
                data = data[k]
            else:
                raise ValueError(f"'{prop}' not found in '{self.filename}'")
        return str(data), type(data)

    def _raise_error_if_is_not_a_known_prop(self, prop):
        """TODO: Implement."""
        ...

    def read(self, prop):
        data, data_type = self._read_raw(prop)

        prop_is_str = data_type is tomlkit.items.String

        prop_is_auto = False
        if prop_is_str:
            data, nsubs = re.subn(r"\s*!?auto$", "", data)
            prop_is_auto = bool(nsubs)

        if data == "":
            raise ValueError(f"'{prop}' is empty")
        if data == "none":
            return None

        if (not prop_is_str) or prop_is_auto:
            data = eval(data)

        return data

    def is_auto(self, prop):
        raw_data, _ = self._read_raw(prop)
        if "!auto" in raw_data:
            return True
        return False

    def write(self, prop, value):
        self._raise_error_if_is_not_a_known_prop(prop)

        # TOML files don't support tuples
        if isinstance(value, tuple):
            value = list(value)

        if self.is_auto(prop):
            value = str(value) + "!auto"

        keys = prop.split(".")
        sub_dict = self.params
        for key in keys[:-1]:
            sub_dict = sub_dict[key]
        sub_dict[keys[-1]] = value

        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(self.params))

    def load_data(self, *, flat=False, ignore_opts=False):
        data_path = self.read("data.io.path")
        data = np.load(data_path)

        data = np.ma.array(data, mask=np.isnan(data))

        if flat:
            data = data.compressed()
        elif data.ndim == 1:
            data = data[np.newaxis, :]
        else:
            pass

        prop_to_use = self.read("data.prop_to_use")
        data = trim_data(data, prop_to_use)

        if ignore_opts:
            return data

        if self.read("data.opts.flip_data") is True:
            if flat:
                data = np.flip(data)
            else:
                data = np.flip(data, axis=1)

        if self.read("data.opts.norm_data") is True:
            data_normalization = np.sqrt(2) * self.read("data.stats.std")
        else:
            data_normalization = 1

        data = data / data_normalization

        return data

    def format_output_filename_for_data(self, filename_raw):
        """
        return <config.io.processed_data_save_path>/<config.io.save_filenames_prefix>_<fielaneme_raw>
        """

        out = "".join([
            self.read("config.io.processed_data_save_path"),
            self.read("config.io.save_filenames_prefix"),
            "_",
            filename_raw,
        ])
        return out

    def format_output_filename_for_figures(self, filename_raw):
        """
        return <config.io.figures_save_path>/<config.io.save_filenames_prefix>_<fielaneme_raw>
        """

        out = "".join([
            self.read("config.io.figures_save_path"),
            self.read("config.io.save_filenames_prefix"),
            "_",
            filename_raw,
        ])
        return out

    def __repr__(self):
        return pprint.pformat(dict(self.params), sort_dicts=False)


def trim_data(data, prop_to_use):
    prop_to_use = np.array(prop_to_use)

    # remove axes of length 1 to have the right .ndim
    squeeze_data = np.squeeze(data)

    if squeeze_data.ndim != prop_to_use.size:
        raise ValueError(
            f"data.prop_to_use {repr(prop_to_use)} is not compatible with data.shape={repr(squeeze_data.shape)}"
        )

    if squeeze_data.ndim == 1:
        tot_size = data.size
        if data.ndim == 1:
            data = data[: int(tot_size * prop_to_use)]
        else:
            data = data[:, : int(tot_size * prop_to_use)]
    elif squeeze_data.ndim == 2:
        tot_size_1, tot_size_2 = data.shape
        prop_1, prop_2 = prop_to_use
        data = data[: int(tot_size_1 * prop_1), : int(tot_size_2 * prop_2)]
    else:
        raise ValueError("Data can't have more than two dimensions.")

    return data
