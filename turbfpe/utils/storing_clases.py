from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Generic, List, TypeVar

import numpy as np
import numpy.typing as npt


def _maybe_extract_scalar(
    value: npt.NDArray[np.float64] | float | int,
) -> float | int | npt.NDArray[np.float64]:
    """
    Returns a Python scalar if 'value' is a NumPy array of size 1.
    Otherwise, returns the value unchanged.
    """
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.item()
    return value


@dataclass
class ConditionalMoments:
    scale: float
    scale_us: float
    scale_short_us: float
    M11: npt.NDArray[np.float64]  # shape: (inc_bin, steps_con_moment.size)
    M21: npt.NDArray[np.float64]  # shape: (inc_bin, steps_con_moment.size)
    M31: npt.NDArray[np.float64]  # shape: (inc_bin, steps_con_moment.size)
    M41: npt.NDArray[np.float64]  # shape: (inc_bin, steps_con_moment.size)
    M1_err: npt.NDArray[np.float64]  # shape: (inc_bin, steps_con_moment.size)
    M2_err: npt.NDArray[np.float64]  # shape: (inc_bin, steps_con_moment.size)


@dataclass
class DensityFunctions:
    scale: float
    scale_us: float
    scale_short_us: float
    P_1: npt.NDArray[np.float64]
    P_0: npt.NDArray[np.float64]
    P_1I0: npt.NDArray[np.float64]
    P_0I1: npt.NDArray[np.float64]
    P_1n0: npt.NDArray[np.float64]
    bin1_edges: npt.NDArray[np.float64]
    bin0_edges: npt.NDArray[np.float64]
    bin1_width: float
    bin0_width: float
    counts1: npt.NDArray[np.float64]
    counts0: npt.NDArray[np.float64]
    mean_per_bin1: npt.NDArray[np.float64]
    mean_per_bin0: npt.NDArray[np.float64]
    inc1: npt.NDArray[np.float64]
    inc0: npt.NDArray[np.float64]


@dataclass
class KMCoeffsEstimation:
    scale: float
    scale_us: float
    scale_short_us: float
    D1: npt.NDArray[np.float64]
    D1_err: npt.NDArray[np.float64]
    D2: npt.NDArray[np.float64]
    D2_err: npt.NDArray[np.float64]
    D3: npt.NDArray[np.float64]
    D4: npt.NDArray[np.float64]
    D1_opti: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    D2_opti: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    valid_idxs: npt.NDArray[np.bool_] = field(
        default_factory=lambda: np.array([], dtype=np.bool_)
    )

    def set_D1_opti(self, value: npt.NDArray[np.float64]) -> None:
        self.D1_opti = value

    def set_D2_opti(self, value: npt.NDArray[np.float64]) -> None:
        self.D2_opti = value

    def set_valid_idxs(self, value: npt.NDArray[np.float64]) -> None:
        self.valid_idxs = value


@dataclass
class KMCoeffs:
    a11: float  # co["a"][0]
    b11: float  # co["ea"][0]
    c11: float  # co["a"][1]
    a20: float  # co["b"][0]
    b20: float  # co["eb"][0]
    c20: float  # co["b"][1]
    a21: float  # co["b"][2]
    b21: float  # co["eb"][1]
    c21: float  # co["b"][3]
    a22: float  # co["b"][4]
    b22: float  # co["eb"][2]
    c22: float  # co["b"][5]
    a11_conf: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    b11_conf: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    c11_conf: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    a20_conf: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    b20_conf: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    c20_conf: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    a21_conf: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    b21_conf: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    c21_conf: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    a22_conf: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    b22_conf: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    c22_conf: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    def eval_d11(self, r):
        return self.a11 * (r**self.b11) + self.c11

    def eval_d20(self, r):
        return self.a20 * (r**self.b20) + self.c20

    def eval_d21(self, r):
        return self.a21 * (r**self.b21) + self.c21

    def eval_d22(self, r):
        return self.a22 * (r**self.b22) + self.c22

    def eval_D1(self, u, r):
        """D1(u, r) = [a11 * r^b11 + c11] * u"""
        d11 = self.eval_d11(r)
        return d11 * u

    def eval_D2(self, u, r):
        """
        D2(u, r) = [a20 * r^b20 + c20]
                 + [a21 * r^b21 + c21] * u
                 + [a22 * r^b22 + c22] * u^2
        """
        d20 = self.eval_d20(r)
        d21 = self.eval_d21(r)
        d22 = self.eval_d22(r)
        return d20 + d21 * u + d22 * (u**2)

    def write_npz(self, filename: str) -> None:
        """
        Writes all dataclass items to a .npz file.
        """
        data_to_save: Dict[str, Any] = asdict(self)
        np.savez(filename, **data_to_save)

    @classmethod
    def load_npz(cls, filename: str) -> "KMCoeffs":
        """
        Loads a KMCoeffs instance from a .npz file.
        """
        data = np.load(filename)
        data_dict = {}
        for key in data.files:
            value = data[key]
            # Convert 0-dimensional arrays (scalars) to native Python types.
            if isinstance(value, np.ndarray) and value.ndim == 0:
                data_dict[key] = value.item()
            else:
                data_dict[key] = value
        data.close()
        return cls(**data_dict)


@dataclass
class Entropies:
    medium_entropy: npt.NDArray[np.float64]
    system_entropy: npt.NDArray[np.float64]
    total_entropy: npt.NDArray[np.float64]
    idx_track: npt.NDArray[np.float64] = None

    def write_npz(self, filename: str) -> None:
        """
        Writes all dataclass items to a .npz file.
        """
        data_to_save: Dict[str, Any] = asdict(self)
        np.savez(filename, **data_to_save)

    @classmethod
    def load_npz(cls, filename: str) -> "Entropies":
        """
        Loads a Entropies instance from a .npz file.
        """
        data_dict = dict(np.load(filename))
        return cls(**data_dict)


# --- Group Containers ---

T = TypeVar("T")


class DataClassGroup(Generic[T]):
    """
    A generic container for storing multiple dataclass instances.
    Provides methods for adding items, iteration, and saving/loading
    from a .npz file.
    """

    def __init__(self):
        self._items: List[T] = []

    def add(self, item: T) -> None:
        self._items.append(item)

    def unpack(self, attr_name: str) -> np.ndarray:
        """
        Extracts and returns a NumPy array containing the values of the
        specified attribute from all dataclass instances in the group,
        padding arrays with different sizes on the right.

        Args:
            attr_name (str): The name of the attribute to extract (e.g. "D1").

        Returns:
            np.ndarray: A NumPy array composed of the values of the specified attribute,
            where shorter arrays are padded with np.nan on the right.
        """

        is_scalar = np.asarray(getattr(self._items[0], attr_name)).ndim == 0
        if is_scalar:
            return np.array([getattr(item, attr_name) for item in self._items])

        arrays = []
        for item in self._items:
            arr = np.asarray(getattr(item, attr_name))
            if arr.ndim == 0:
                arr = np.array([arr])
            arrays.append(arr)

        # Determine the maximum size along the first dimension.
        max_len = max(arr.shape[0] for arr in arrays)
        # Get the remaining dimensions, if any.
        result = np.full((len(arrays), max_len), np.nan, dtype=arrays[0].dtype)
        for i, arr in enumerate(arrays):
            length = arr.shape[0]
            result[i, :length, ...] = arr
        return result

    def write_npz(self, filename: str) -> None:
        """
        Writes all dataclass items to a .npz file.
        Each item is enumerated with keys like "0_field", "1_field", etc.
        """
        data_to_save: Dict[str, Any] = {}
        for idx, item in enumerate(self._items):
            item_dict = asdict(item)
            for field_name, value in item_dict.items():
                data_to_save[f"{idx}_{field_name}"] = value
        np.savez(filename, **data_to_save)

    def copy(self) -> "DataClassGroup[T]":
        return deepcopy(self)

    @classmethod
    def load_npz(
        cls, filename: str, constructor: Callable[[Dict[str, Any]], T]
    ) -> "DataClassGroup[T]":
        """
        Generic method to create a DataClassGroup from an .npz file.
        The provided 'constructor' converts a dict of fields into an instance of T.
        """
        group = cls()
        raw_groups = cls._load_raw_data(filename)
        for idx in sorted(raw_groups.keys()):
            fields = raw_groups[idx]
            instance = constructor(fields)
            group.add(instance)
        return group

    @classmethod
    def _load_raw_data(cls, filename: str) -> Dict[int, Dict[str, Any]]:
        """
        Reads an .npz file and groups data by item index.
        Returns:
            A dict mapping each item index to its field dictionary.
        """
        raw_groups: Dict[int, Dict[str, Any]] = {}
        with np.load(filename, allow_pickle=False) as data:
            for key in data.keys():
                idx_str, field_name = key.split("_", 1)
                idx = int(idx_str)
                raw_groups.setdefault(idx, {})[field_name] = data[key]
        return raw_groups

    def __getitem__(self, index: int) -> T:
        return self._items[index]

    def __setitem__(self, index: int, value: T) -> None:
        self._items[index] = value

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


class ConditionalMomentsGroup(DataClassGroup[ConditionalMoments]):
    """
    Group for storing ConditionalMoments instances.
    """

    @classmethod
    def load_npz(cls, filename: str) -> "ConditionalMomentsGroup":
        def constructor(fields: Dict[str, Any]) -> ConditionalMoments:
            # Convert scalar fields
            for key in ("scale", "scale_us", "scale_short_us"):
                fields[key] = _maybe_extract_scalar(fields[key])
            return ConditionalMoments(**fields)

        return super().load_npz(filename, constructor)


class DensityFunctionsGroup(DataClassGroup[DensityFunctions]):
    """
    Group for storing DensityFunctions instances.
    """

    @classmethod
    def load_npz(cls, filename: str) -> "DensityFunctionsGroup":
        def constructor(fields: Dict[str, Any]) -> DensityFunctions:
            for key in (
                "scale",
                "scale_us",
                "scale_short_us",
                "bin1_width",
                "bin0_width",
            ):
                fields[key] = _maybe_extract_scalar(fields[key])
            return DensityFunctions(**fields)

        return super().load_npz(filename, constructor)


class KMCoeffsEstimationGroup(DataClassGroup[KMCoeffsEstimation]):
    """
    Group for storing KMCoeffsEstimation instances.
    """

    @classmethod
    def load_npz(cls, filename: str) -> "KMCoeffsEstimationGroup":
        def constructor(fields: Dict[str, Any]) -> KMCoeffsEstimation:
            for key in ("scale", "scale_us", "scale_short_us"):
                fields[key] = _maybe_extract_scalar(fields[key])
            return KMCoeffsEstimation(**fields)

        return super().load_npz(filename, constructor)
