"""Differentiable HBV-96 hydrological model in PyTorch.

The model follows the classic HBV structure (Bergström, 1976; Lindström et al., 1997;
Seibert & Vis, 2012): a degree-day snow routine, a soil-moisture routine, a two-box
response routine and a triangular routing function.

Every operation is written in PyTorch, so gradients of any loss (e.g. 1 - NSE) flow
back to the 15 parameters. The model runs on a batch of basins at once, which makes
it suitable for large-sample (CAMELS / Caravan) calibration.

Mass balance is tracked explicitly: over any simulation,

    sum(P_eff) = sum(ET) + sum(Q_generated) + delta(storages)

holds to floating-point precision (see ``HBVOutput.water_balance_residual``).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# name: (lower bound, upper bound, unit, description)
PARAMETER_BOUNDS: dict[str, tuple[float, float, str, str]] = {
    "TT":     (-2.5, 2.5,  "°C",          "threshold temperature for snow/rain"),
    "CFMAX":  (0.5, 10.0,  "mm/°C/day",   "degree-day melt factor"),
    "SFCF":   (0.4, 1.4,   "-",           "snowfall correction factor"),
    "CFR":    (0.0, 0.1,   "-",           "refreezing coefficient"),
    "CWH":    (0.0, 0.2,   "-",           "water holding capacity of snowpack"),
    "FC":     (50.0, 1000.0, "mm",        "maximum soil moisture storage"),
    "LP":     (0.2, 1.0,   "-",           "fraction of FC above which ET is potential"),
    "BETA":   (1.0, 6.0,   "-",           "shape coefficient of soil recharge"),
    "PERC":   (0.0, 10.0,  "mm/day",      "maximum percolation to lower zone"),
    "UZL":    (0.0, 100.0, "mm",          "threshold for quick flow from upper zone"),
    "K0":     (0.05, 0.9,  "1/day",       "quick-flow recession coefficient"),
    "K1":     (0.01, 0.5,  "1/day",       "upper-zone recession coefficient"),
    "K2":     (0.001, 0.2, "1/day",       "lower-zone (baseflow) recession coefficient"),
    "MAXBAS": (1.0, 7.0,   "day",         "length of triangular routing function"),
    "PCORR":  (0.8, 1.2,   "-",           "rainfall correction factor"),
}
PARAMETER_NAMES = list(PARAMETER_BOUNDS)
N_PARAMS = len(PARAMETER_NAMES)
MAX_ROUTING_DAYS = 7
EPS = 1e-6


@dataclass
class HBVOutput:
    """Simulation results; all tensors have shape [batch, time] unless noted."""

    q_routed: torch.Tensor      # streamflow at the outlet (mm/day)
    q_generated: torch.Tensor   # runoff before routing (mm/day)
    et: torch.Tensor            # actual evapotranspiration (mm/day)
    p_eff: torch.Tensor         # corrected precipitation entering the model (mm/day)
    snowpack: torch.Tensor      # frozen snow storage (mm)
    soil_moisture: torch.Tensor  # soil storage (mm)
    suz: torch.Tensor           # upper-zone storage (mm)
    slz: torch.Tensor           # lower-zone storage (mm)
    initial_storage: torch.Tensor  # total storage at t=0, shape [batch]
    final_storage: torch.Tensor    # total storage at the end, shape [batch]
    routing_tail: torch.Tensor     # runoff still inside the routing buffer at the end, [batch]

    def water_balance_residual(self) -> torch.Tensor:
        """Inputs - outputs - storage change for each basin (should be ~0), shape [batch]."""
        inputs = self.p_eff.sum(dim=1)
        outputs = self.et.sum(dim=1) + self.q_generated.sum(dim=1)
        return inputs - outputs - (self.final_storage - self.initial_storage)

    def routing_residual(self) -> torch.Tensor:
        """Generated runoff - routed runoff - runoff left in the routing buffer, shape [batch]."""
        return self.q_generated.sum(dim=1) - self.q_routed.sum(dim=1) - self.routing_tail


def scale_parameters(raw: torch.Tensor) -> torch.Tensor:
    """Map unconstrained values (any real number) to physical parameter ranges with a sigmoid."""
    lower = torch.tensor([b[0] for b in PARAMETER_BOUNDS.values()], dtype=raw.dtype, device=raw.device)
    upper = torch.tensor([b[1] for b in PARAMETER_BOUNDS.values()], dtype=raw.dtype, device=raw.device)
    return lower + (upper - lower) * torch.sigmoid(raw)


def unscale_parameters(params: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`scale_parameters` (for initialising from known values)."""
    lower = torch.tensor([b[0] for b in PARAMETER_BOUNDS.values()], dtype=params.dtype, device=params.device)
    upper = torch.tensor([b[1] for b in PARAMETER_BOUNDS.values()], dtype=params.dtype, device=params.device)
    frac = ((params - lower) / (upper - lower)).clamp(1e-6, 1 - 1e-6)
    return torch.log(frac / (1 - frac))


def triangular_weights(maxbas: torch.Tensor, length: int = MAX_ROUTING_DAYS) -> torch.Tensor:
    """Differentiable triangular unit hydrograph with base ``maxbas`` (days).

    Returns weights of shape [batch, length] that sum to 1 for every basin.
    """
    t = torch.arange(length, dtype=maxbas.dtype, device=maxbas.device) + 0.5  # bin centres
    half = (maxbas / 2).unsqueeze(1)
    tri = torch.relu(1 - torch.abs(t.unsqueeze(0) - half) / half)
    tri = tri + EPS  # keeps gradients alive for very short MAXBAS
    return tri / tri.sum(dim=1, keepdim=True)


class HBV(nn.Module):
    """Batched differentiable HBV model.

    Parameters are stored as unconstrained values and scaled to physical ranges in
    ``forward``, so any gradient-based optimiser can be used directly.
    """

    def __init__(self, n_basins: int = 1, init_raw: torch.Tensor | None = None):
        super().__init__()
        if init_raw is None:
            init_raw = torch.zeros(n_basins, N_PARAMS)  # = middle of every range
        self.raw = nn.Parameter(init_raw.clone().float())

    @property
    def params(self) -> torch.Tensor:
        return scale_parameters(self.raw)

    def named_params(self, basin: int = 0) -> dict[str, float]:
        values = self.params[basin].detach().cpu().tolist()
        return dict(zip(PARAMETER_NAMES, values))

    def forward(self, precip: torch.Tensor, temp: torch.Tensor, pet: torch.Tensor,
                params: torch.Tensor | None = None, initial_states: dict | None = None) -> HBVOutput:
        """Run the model.

        Args:
            precip, temp, pet: forcing tensors of shape [batch, time]
                (mm/day, °C, mm/day).
            params: optional physical parameters [batch, 15]; defaults to this module's.
            initial_states: optional dict with keys snow, liquid, sm, suz, slz ([batch]).
        """
        p = self.params if params is None else params
        p = p.to(precip.dtype)
        b, n_t = precip.shape
        (tt, cfmax, sfcf, cfr, cwh, fc, lp, beta, perc,
         uzl, k0, k1, k2, maxbas, pcorr) = [p[:, i] for i in range(N_PARAMS)]

        zeros = torch.zeros(b, dtype=precip.dtype, device=precip.device)
        s = initial_states or {}
        snow = s.get("snow", zeros.clone())
        liquid = s.get("liquid", zeros.clone())
        sm = s.get("sm", 0.5 * fc)
        suz = s.get("suz", zeros.clone())
        slz = s.get("slz", zeros.clone())
        initial_storage = snow + liquid + sm + suz + slz

        weights = triangular_weights(maxbas)          # [batch, L]
        buffer = torch.zeros(b, MAX_ROUTING_DAYS, dtype=precip.dtype, device=precip.device)

        out = {k: [] for k in ("q_routed", "q_generated", "et", "p_eff", "snow", "sm", "suz", "slz")}

        for t in range(n_t):
            P, T, E = precip[:, t], temp[:, t], pet[:, t]

            # --- snow routine (degree-day) ---
            is_snow = (T < tt).to(P.dtype)
            rain = (1 - is_snow) * P * pcorr
            snowfall = is_snow * P * sfcf
            p_eff = rain + snowfall
            snow = snow + snowfall
            melt = torch.minimum(cfmax * torch.relu(T - tt), snow)
            snow = snow - melt
            refreeze = torch.minimum(cfr * cfmax * torch.relu(tt - T), liquid)
            snow = snow + refreeze
            liquid = liquid + melt - refreeze + rain
            release = torch.relu(liquid - cwh * snow)   # water the snowpack cannot hold
            liquid = liquid - release

            # --- soil routine ---
            ratio = torch.clamp(sm / fc, EPS, 1.0)
            recharge = release * ratio ** beta
            sm = sm + release - recharge
            excess = torch.relu(sm - fc)
            sm = sm - excess
            recharge = recharge + excess
            et = torch.minimum(E * torch.clamp(sm / (lp * fc), 0.0, 1.0), sm)
            sm = sm - et

            # --- response routine ---
            suz = suz + recharge
            percolation = torch.minimum(perc, suz)
            suz = suz - percolation
            slz = slz + percolation
            q0 = k0 * torch.relu(suz - uzl)
            suz = suz - q0
            q1 = k1 * suz
            suz = suz - q1
            q2 = k2 * slz
            slz = slz - q2
            q_gen = q0 + q1 + q2

            # --- routing (triangular unit hydrograph) ---
            buffer = buffer + q_gen.unsqueeze(1) * weights
            q_out = buffer[:, 0]
            buffer = torch.cat([buffer[:, 1:], torch.zeros_like(buffer[:, :1])], dim=1)

            out["q_routed"].append(q_out)
            out["q_generated"].append(q_gen)
            out["et"].append(et)
            out["p_eff"].append(p_eff)
            out["snow"].append(snow + liquid)
            out["sm"].append(sm)
            out["suz"].append(suz)
            out["slz"].append(slz)

        stack = {k: torch.stack(v, dim=1) for k, v in out.items()}
        return HBVOutput(
            q_routed=stack["q_routed"], q_generated=stack["q_generated"], et=stack["et"],
            p_eff=stack["p_eff"], snowpack=stack["snow"], soil_moisture=stack["sm"],
            suz=stack["suz"], slz=stack["slz"],
            initial_storage=initial_storage,
            final_storage=snow + liquid + sm + suz + slz,
            routing_tail=buffer.sum(dim=1),
        )
