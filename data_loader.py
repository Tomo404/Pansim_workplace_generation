# data_loader.py
# ------------------------------
# Loads KSH-derived organization counts from:
# - tensor_main.npy (shape: settlements x size_bins x teaor)
# - tensor_main_dimensions.json (names for each axis)
#
# Output format:
#   Dict[CellKey, int]  where CellKey = (settlement, teaor, bin_name)

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from generator import CellKey


# Mapping from Hungarian KSH bin labels -> generator bin keys
KSH_BIN_TO_INTERNAL = {
    "1-4 fő": "1-4",
    "5-9 fő": "5-9",
    "10-19 fő": "10-19",
    "20-49 fő": "20-49",
    "50-249 fő": "50-249",
    "250 fő felett": "250-1000",  # capped in generator.py
}


@dataclass(frozen=True)
class TensorDims:
    settlements: List[str]
    bins: List[str]
    teaor: List[str]


def load_tensor_dims(dimensions_json_path: str) -> TensorDims:
    """
    Reads tensor dimension JSON.

    Supported old format:
      - teruletek
      - letszamkategoriak
      - teaor_alagak

    Supported new format:
      - teruletek
      - letszamkategoriak
      - teaor_kulcsok
      - teaor_feloldas
    """
    with open(dimensions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    settlements = list(data["teruletek"])
    bins = list(data["letszamkategoriak"])

    if "teaor_alagak" in data:
        teaor = list(data["teaor_alagak"])

    elif "teaor_feloldas" in data and "teaor_kulcsok" in data:
        # Új formátum: a tensor tengelye teaor_kulcsok sorrendben van,
        # de a generátorhoz inkább a numerikus alágkód kell.
        mapping = {
            row["teaor_kulcs"]: str(row["TEÁOR alág kód"])
            for row in data["teaor_feloldas"]
        }
        teaor = [mapping[k] for k in data["teaor_kulcsok"]]

    elif "teaor_kulcsok" in data:
        # Fallback: ha nincs feloldás, akkor legalább a kulcsokat használjuk
        teaor = list(data["teaor_kulcsok"])

    else:
        raise KeyError(
            f"Nem található megfelelő TEÁOR kulcs a dimenziófájlban. "
            f"Elérhető kulcsok: {list(data.keys())}"
        )

    return TensorDims(
        settlements=settlements,
        bins=bins,
        teaor=teaor,
    )


def load_tensor(tensor_npy_path: str) -> np.ndarray:
    """
    Loads tensor_main.npy.
    Expected shape: (n_settlements, n_bins, n_teaor)
    """
    arr = np.load(tensor_npy_path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {arr.shape}")
    return arr


def build_counts_from_tensor(
    tensor: np.ndarray,
    dims: TensorDims,
    drop_zeros: bool = True,
) -> Dict[CellKey, int]:
    """
    Convert the 3D tensor into Dict[CellKey, int] counts.

    - settlement name comes from dims.settlements[i]
    - TEÁOR code comes from dims.teaor[k] (kept as string)
    - bin_name is mapped through KSH_BIN_TO_INTERNAL
    """
    n_sett, n_bins, n_teaor = tensor.shape

    if n_sett != len(dims.settlements):
        raise ValueError("Mismatch: tensor settlements axis vs dimensions file")
    if n_bins != len(dims.bins):
        raise ValueError("Mismatch: tensor bins axis vs dimensions file")
    if n_teaor != len(dims.teaor):
        raise ValueError("Mismatch: tensor teaor axis vs dimensions file")

    counts: Dict[CellKey, int] = {}

    for i_sett, sett_name in enumerate(dims.settlements):
        for i_bin, ksh_bin in enumerate(dims.bins):
            if ksh_bin not in KSH_BIN_TO_INTERNAL:
                raise KeyError(
                    f"Unknown KSH bin label: {ksh_bin}. "
                    f"Known: {list(KSH_BIN_TO_INTERNAL.keys())}"
                )
            bin_name = KSH_BIN_TO_INTERNAL[ksh_bin]

            for i_tea, teaor_code in enumerate(dims.teaor):
                value = int(tensor[i_sett, i_bin, i_tea])  # these are integers in practice

                if drop_zeros and value == 0:
                    continue

                key = CellKey(settlement=sett_name, teaor=str(teaor_code), bin_name=bin_name)
                counts[key] = value

    return counts


def load_counts_from_tensor_files(
    tensor_npy_path: str,
    dimensions_json_path: str,
    drop_zeros: bool = True,
) -> Dict[CellKey, int]:
    """
    Convenience wrapper used by main.py.
    """
    dims = load_tensor_dims(dimensions_json_path)
    tensor = load_tensor(tensor_npy_path)
    return build_counts_from_tensor(tensor, dims, drop_zeros=drop_zeros)
