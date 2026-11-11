"""Loading CAMELS-US basins and computing potential evapotranspiration."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

CFS_TO_M3S = 0.0283168466


def hargreaves_pet(tmin: np.ndarray, tmax: np.ndarray, doy: np.ndarray, lat_deg: float) -> np.ndarray:
    """Hargreaves (1985) potential evapotranspiration in mm/day.

    Uses extraterrestrial radiation from FAO-56 (Allen et al., 1998), eq. 21.
    """
    lat = math.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(decl), -1, 1))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.sin(ws))  # MJ/m2/day
    tmean = (tmin + tmax) / 2
    pet = 0.0023 * 0.408 * ra * (tmean + 17.8) * np.sqrt(np.clip(tmax - tmin, 0, None))
    return np.clip(pet, 0, None)


def _find(root: Path, pattern: str) -> Path:
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matching {pattern!r} under {root}")
    return hits[0]


def load_camels_basin(camels_root: str | Path, basin: str, forcing: str = "daymet") -> pd.DataFrame:
    """Load one CAMELS-US basin as a daily DataFrame.

    ``camels_root`` is the folder that contains ``basin_mean_forcing`` and
    ``usgs_streamflow`` (from the "CAMELS time series" download on
    https://ral.ucar.edu/solutions/products/camels).

    Returns columns: precip (mm/day), tmin, tmax, tmean (°C), pet (mm/day), qobs (mm/day).
    Attributes ``lat`` and ``area_km2`` are stored in ``df.attrs``.
    """
    root = Path(camels_root)
    f_forc = _find(root / "basin_mean_forcing" / forcing, f"{basin}_lump_*forcing*.txt")
    f_q = _find(root / "usgs_streamflow", f"{basin}_streamflow_qc.txt")

    with open(f_forc) as fh:
        lat = float(fh.readline())
        _elev = float(fh.readline())
        area_m2 = float(fh.readline())

    forc = pd.read_csv(f_forc, sep=r"\s+", skiprows=3)
    forc.index = pd.to_datetime(dict(year=forc["Year"], month=forc["Mnth"], day=forc["Day"]))
    forc = forc.rename(columns=lambda c: c.split("(")[0].strip().lower())

    q = pd.read_csv(f_q, sep=r"\s+", header=None,
                    names=["gauge", "year", "month", "day", "q_cfs", "flag"])
    q.index = pd.to_datetime(q[["year", "month", "day"]])
    q_cfs = q["q_cfs"].where(q["q_cfs"] >= 0)            # -999 = missing
    qobs = q_cfs * CFS_TO_M3S * 86400 / area_m2 * 1000      # m3/s -> mm/day

    df = pd.DataFrame({
        "precip": forc["prcp"],
        "tmin": forc["tmin"],
        "tmax": forc["tmax"],
    })
    df["tmean"] = (df["tmin"] + df["tmax"]) / 2
    df["pet"] = hargreaves_pet(df["tmin"].values, df["tmax"].values, df.index.dayofyear.values, lat)
    df["qobs"] = qobs.reindex(df.index)
    df.attrs.update(lat=lat, area_km2=area_m2 / 1e6, basin=basin)
    return df
