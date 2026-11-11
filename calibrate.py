"""Gradient-based calibration of HBV on a CAMELS-US basin.

Example:
    python -m hbv_torch.calibrate --camels-root data/CAMELS --basin 01022500 --epochs 300
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_camels_basin
from .metrics import kge, nse, nse_loss, pbias
from .model import HBV, PARAMETER_BOUNDS


def to_tensor(x: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(np.array(x, dtype=np.float64), dtype=dtype).unsqueeze(0)


def fit(model: HBV, precip, temp, pet, qobs, warmup: int, epochs: int = 300, lr: float = 0.05,
        verbose: bool = True) -> list[float]:
    """Adam on 1 - NSE, ignoring the first ``warmup`` days. Tensors: [batch, time]."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    target = qobs[:, warmup:]
    history = []
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        out = model(precip, temp, pet)
        loss = nse_loss(target, out.q_routed[:, warmup:])
        loss.backward()
        opt.step()
        history.append(loss.item())
        if verbose and (epoch == 1 or epoch % 25 == 0):
            print(f"epoch {epoch:4d}  NSE {1 - loss.item():.3f}")
    return history


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camels-root", required=True, help="folder containing basin_mean_forcing/ and usgs_streamflow/")
    ap.add_argument("--basin", default="01022500", help="8-digit USGS gauge id")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--warmup-start", default="1998-10-01")
    ap.add_argument("--cal-start", default="1999-10-01")
    ap.add_argument("--cal-end", default="2008-09-30")
    ap.add_argument("--val-end", default="2014-09-30")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    df = load_camels_basin(args.camels_root, args.basin)
    cal = df.loc[args.warmup_start:args.cal_end]
    full = df.loc[args.warmup_start:args.val_end]
    warmup = len(df.loc[args.warmup_start:args.cal_start]) - 1

    model = HBV(n_basins=1)
    t0 = time.time()
    history = fit(model, to_tensor(cal.precip), to_tensor(cal.tmean), to_tensor(cal.pet),
                  to_tensor(cal.qobs), warmup=warmup, epochs=args.epochs, lr=args.lr)
    print(f"calibration took {time.time() - t0:.0f} s")

    with torch.no_grad():
        out = model(to_tensor(full.precip, torch.float64), to_tensor(full.tmean, torch.float64),
                    to_tensor(full.pet, torch.float64), params=model.params.double())
    sim = out.q_routed[0].numpy()
    obs = full.qobs.values
    n_cal = len(df.loc[args.warmup_start:args.cal_end])
    periods = {"calibration": slice(warmup, n_cal), "validation": slice(n_cal, len(full))}
    metrics = {
        name: {"NSE": float(nse(torch.tensor(obs[s]), torch.tensor(sim[s]))),
               "KGE": float(kge(torch.tensor(obs[s]), torch.tensor(sim[s]))),
               "PBIAS_%": float(pbias(torch.tensor(obs[s]), torch.tensor(sim[s])))}
        for name, s in periods.items()
    }
    metrics["water_balance_residual_mm"] = float(out.water_balance_residual()[0])
    metrics["routing_residual_mm"] = float(out.routing_residual()[0])

    out_dir = Path(args.out) / args.basin
    out_dir.mkdir(parents=True, exist_ok=True)
    params = {k: round(v, 4) for k, v in model.named_params().items()}
    (out_dir / "params.json").write_text(json.dumps(params, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    full.assign(qsim=sim).to_csv(out_dir / "simulation.csv")
    _plot(full.index, obs, sim, history, n_cal, out_dir, args.basin, metrics)

    print(json.dumps(metrics, indent=2))
    print(f"results written to {out_dir}/")


def _plot(index, obs, sim, history, n_cal, out_dir, basin, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(index, obs, color="black", lw=0.8, label="observed")
    ax.plot(index, sim, color="tab:blue", lw=0.8, alpha=0.85, label="HBV (differentiable)")
    ax.axvline(index[n_cal], color="grey", ls="--", lw=0.8)
    ax.set_ylabel("Streamflow (mm/day)")
    ax.set_title(f"Basin {basin} — calibration NSE {metrics['calibration']['NSE']:.2f}, "
                 f"validation NSE {metrics['validation']['NSE']:.2f}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "hydrograph.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot([1 - h for h in history])
    ax.set_xlabel("epoch")
    ax.set_ylabel("calibration NSE")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=150)


if __name__ == "__main__":
    main()
