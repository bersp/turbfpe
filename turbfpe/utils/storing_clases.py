from __future__ import annotations

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
    D1_opti: npt.NDArray[np.float64]
    D2_opti: npt.NDArray[np.float64]
    valid_idxs: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))


@dataclass
class KMCoeffs:
    a11: float
    b11: float
    c11: float
    a20: float
    b20: float
    c20: float
    a21: float
    b21: float
    c21: float
    a22: float
    b22: float
    c22: float
    a11_conf: npt.NDArray[np.float64]
    b11_conf: npt.NDArray[np.float64]
    c11_conf: npt.NDArray[np.float64]
    a20_conf: npt.NDArray[np.float64]
    b20_conf: npt.NDArray[np.float64]
    c20_conf: npt.NDArray[np.float64]
    a21_conf: npt.NDArray[np.float64]
    b21_conf: npt.NDArray[np.float64]
    c21_conf: npt.NDArray[np.float64]
    a22_conf: npt.NDArray[np.float64]
    b22_conf: npt.NDArray[np.float64]
    c22_conf: npt.NDArray[np.float64]

    def eval_D1(self, u, r):
        """D1(u, r) = [a11 * r^b11 + c11] * u"""
        d11 = self.a11 * (r**self.b11) + self.c11
        return d11 * u

    def eval_D2(self, u, r):
        """
        D2(u, r) = [a20 * r^b20 + c20]
                 + [a21 * r^b21 + c21] * u
                 + [a22 * r^b22 + c22] * u^2
        """
        d20 = self.a20 * (r**self.b20) + self.c20
        d21 = self.a21 * (r**self.b21) + self.c21
        d22 = self.a22 * (r**self.b22) + self.c22
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


# --- Generic Group Container ---

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

    def __getitem__(self, index: int) -> T:
        return self._items[index]

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

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
