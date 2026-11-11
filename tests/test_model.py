import math

import torch

from hbv_torch import HBV, PARAMETER_NAMES, nse, nse_loss, scale_parameters, unscale_parameters
from hbv_torch.calibrate import fit
from hbv_torch.model import triangular_weights


def synthetic_forcing(n_days=730, n_basins=2, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n_days, dtype=dtype)
    season = torch.sin(2 * math.pi * (t - 100) / 365)
    temp = (8 + 12 * season).repeat(n_basins, 1) + torch.randn(n_basins, n_days, generator=g, dtype=dtype) * 2
    wet = (torch.rand(n_basins, n_days, generator=g, dtype=dtype) < 0.35).to(dtype)
    precip = wet * torch.distributions.Exponential(torch.tensor(0.15, dtype=dtype)).sample((n_basins, n_days))
    pet = torch.clamp(1.5 + 2.0 * season, min=0.1).repeat(n_basins, 1)
    return precip, temp, pet


def test_water_balance_closes():
    precip, temp, pet = synthetic_forcing()
    model = HBV(n_basins=2, init_raw=torch.randn(2, len(PARAMETER_NAMES)))
    out = model(precip, temp, pet, params=model.params.double())
    assert torch.all(out.water_balance_residual().abs() < 1e-9)
    assert torch.all(out.routing_residual().abs() < 1e-9)


def test_states_non_negative():
    precip, temp, pet = synthetic_forcing()
    model = HBV(n_basins=2, init_raw=torch.randn(2, len(PARAMETER_NAMES)))
    out = model(precip, temp, pet, params=model.params.double())
    for x in (out.q_routed, out.et, out.snowpack, out.soil_moisture, out.suz, out.slz):
        assert torch.all(x >= -1e-12)


def test_routing_weights_sum_to_one():
    w = triangular_weights(torch.tensor([1.0, 2.5, 7.0]))
    assert torch.allclose(w.sum(dim=1), torch.ones(3))


def test_every_parameter_gets_a_gradient():
    precip, temp, pet = synthetic_forcing(n_days=400, n_basins=1, dtype=torch.float32)
    model = HBV(n_basins=1)
    out = model(precip, temp, pet)
    out.q_routed.sum().backward()
    assert torch.all(torch.isfinite(model.raw.grad))
    assert (model.raw.grad.abs() > 0).sum() >= len(PARAMETER_NAMES) - 2  # snow params can be inactive


def test_scaling_round_trip():
    raw = torch.randn(3, len(PARAMETER_NAMES), dtype=torch.float64)
    assert torch.allclose(unscale_parameters(scale_parameters(raw)), raw, atol=1e-6)


def test_nse_metrics():
    obs = torch.tensor([1.0, 2.0, 3.0, float("nan"), 5.0])
    assert nse(obs, obs) == 1
    assert abs(float(nse(obs, torch.full_like(obs, float(torch.nanmean(obs)))))) < 1e-6
    assert abs(float(nse_loss(obs.unsqueeze(0), obs.unsqueeze(0)))) < 1e-9


def test_calibration_recovers_synthetic_truth():
    """Generate flow with known parameters, start elsewhere, and calibrate back."""
    torch.manual_seed(1)
    precip, temp, pet = synthetic_forcing(n_days=1095, n_basins=1, dtype=torch.float32)
    truth = HBV(n_basins=1, init_raw=torch.randn(1, len(PARAMETER_NAMES)) * 0.8)
    with torch.no_grad():
        q_true = truth(precip, temp, pet).q_routed
    model = HBV(n_basins=1)
    fit(model, precip, temp, pet, q_true, warmup=365, epochs=150, lr=0.05, verbose=False)
    with torch.no_grad():
        q_sim = model(precip, temp, pet).q_routed
    assert nse(q_true[0, 365:], q_sim[0, 365:]) > 0.9
