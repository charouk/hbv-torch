"""LSTM baseline on the same basin and periods as the HBV calibration, plus a side-by-side comparison.

Run the HBV calibration first (it writes results/<basin>/simulation.csv), then:

    python -m hbv_torch.lstm_baseline --camels-root data/CAMELS --basin 01022500

The LSTM sees only meteorological forcing (precipitation, temperature, PET) over the
past 365 days, never past discharge, so it is a fair comparison with HBV.
Water years 2000–2006 train the network, 2007–2008 are used for early stopping
(still inside the HBV calibration period), and 2009–2014 is the shared test period.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import load_camels_basin

FEATURES = ["precip", "tmean", "pet", "tmin", "tmax"]


class StreamflowLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1]))


def make_windows(x: np.ndarray, y: np.ndarray, seq_len: int, idx: np.ndarray):
    """Windows ending at each index in ``idx`` with a valid target."""
    keep = [i for i in idx if i >= seq_len - 1 and not np.isnan(y[i])]
    xs = np.stack([x[i - seq_len + 1:i + 1] for i in keep])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(y[keep], dtype=torch.float32).unsqueeze(1), keep


def nse_np(o, s):
    v = ~np.isnan(o)
    o, s = o[v], s[v]
    return float(1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2))


def kge_np(o, s):
    v = ~np.isnan(o)
    o, s = o[v], s[v]
    r = np.corrcoef(o, s)[0, 1]
    return float(1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2))


def flow_biases(o, s):
    """Volume bias (%) on the highest 2% and lowest 30% of observed flows (sim - obs, positive = overestimate)."""
    v = ~np.isnan(o)
    o, s = o[v], s[v]
    hi = o >= np.quantile(o, 0.98)
    lo = o <= np.quantile(o, 0.30)
    return (float(100 * (s[hi].sum() - o[hi].sum()) / o[hi].sum()),
            float(100 * (s[lo].sum() - o[lo].sum()) / o[lo].sum()))


def scores(o, s):
    hi, lo = flow_biases(o, s)
    return {"NSE": nse_np(o, s), "KGE": kge_np(o, s),
            "PBIAS_%": float(100 * np.nansum(o - s) / np.nansum(o)),
            "peak_flow_bias_%": hi, "low_flow_bias_%": lo}


def train_one(xtr, ytr, xva, yva, n_features, seed, epochs, lr, hidden, verbose):
    torch.manual_seed(seed)
    model = StreamflowLSTM(n_features, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best, best_state, patience = -np.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        for b in range(0, len(xtr), 256):
            i = perm[b:b + 256]
            opt.zero_grad()
            loss = loss_fn(model(xtr[i]), ytr[i])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_nse = nse_np(yva.numpy().ravel(), model(xva).numpy().ravel())
        if verbose:
            print(f"  seed {seed} epoch {epoch:3d}  validation NSE {val_nse:.3f}")
        if val_nse > best:
            best, best_state, patience = val_nse, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 8:
                break
    model.load_state_dict(best_state)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camels-root", required=True)
    ap.add_argument("--basin", default="01022500")
    ap.add_argument("--seq-len", type=int, default=365)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=3, help="ensemble size (mean of seeds is reported)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    df = load_camels_basin(args.camels_root, args.basin).loc["1998-10-01":"2014-09-30"]
    dates = df.index
    train_idx = np.where((dates >= "1999-10-01") & (dates <= "2006-09-30"))[0]
    val_idx = np.where((dates >= "2006-10-01") & (dates <= "2008-09-30"))[0]
    test_idx = np.where(dates >= "2008-10-01")[0]

    x = df[FEATURES].to_numpy(dtype=float)
    y = df["qobs"].to_numpy(dtype=float)
    mu, sd = x[train_idx].mean(0), x[train_idx].std(0) + 1e-6    # training statistics only
    q_mu, q_sd = np.nanmean(y[train_idx]), np.nanstd(y[train_idx])
    xn, yn = (x - mu) / sd, (y - q_mu) / q_sd

    xtr, ytr, _ = make_windows(xn, yn, args.seq_len, train_idx)
    xva, yva, _ = make_windows(xn, yn, args.seq_len, val_idx)
    xte, _, keep_te = make_windows(xn, yn, args.seq_len, test_idx)

    preds = []
    for seed in range(args.seeds):
        print(f"training LSTM seed {seed + 1}/{args.seeds}")
        model = train_one(xtr, ytr, xva, yva, len(FEATURES), seed, args.epochs, args.lr, args.hidden, verbose=True)
        with torch.no_grad():
            preds.append(np.clip(model(xte).numpy().ravel() * q_sd + q_mu, 0, None))
    q_lstm = pd.Series(np.mean(preds, axis=0), index=dates[keep_te])

    out_dir = Path(args.out) / args.basin
    out_dir.mkdir(parents=True, exist_ok=True)
    obs = df["qobs"].loc[q_lstm.index].to_numpy()
    summary = {"test_period": f"{q_lstm.index[0].date()} to {q_lstm.index[-1].date()}",
               "LSTM": scores(obs, q_lstm.to_numpy()),
               "LSTM_per_seed_NSE": [nse_np(obs, p) for p in preds]}

    hbv_file = out_dir / "simulation.csv"
    q_hbv = None
    if hbv_file.exists():
        sim = pd.read_csv(hbv_file, index_col=0, parse_dates=True)
        q_hbv = sim["qsim"].reindex(q_lstm.index)
        summary["HBV"] = scores(obs, q_hbv.to_numpy())
    else:
        print(f"note: {hbv_file} not found — run `python -m hbv_torch.calibrate` first to compare with HBV")

    (out_dir / "comparison.json").write_text(json.dumps(summary, indent=2))
    _plot(q_lstm.index, obs, q_lstm.to_numpy(), None if q_hbv is None else q_hbv.to_numpy(), out_dir, args.basin, summary)
    print(json.dumps(summary, indent=2))
    print(f"written to {out_dir}/comparison.json and comparison.png")


def _plot(index, obs, lstm, hbv, out_dir, basin, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), gridspec_kw={"height_ratios": [1.3, 1]})
    title = f"Basin {basin}, test period — LSTM NSE {summary['LSTM']['NSE']:.2f}"
    if hbv is not None:
        title += f", HBV NSE {summary['HBV']['NSE']:.2f}"
    ax = axes[0]
    ax.plot(index, obs, color="black", lw=0.8, label="observed")
    if hbv is not None:
        ax.plot(index, hbv, color="tab:blue", lw=0.8, alpha=0.85, label="HBV (differentiable)")
    ax.plot(index, lstm, color="tab:orange", lw=0.8, alpha=0.85, label="LSTM")
    ax.set_ylabel("Streamflow (mm/day)")
    ax.set_title(title)
    ax.legend(loc="upper right")

    # zoom on the wettest water year of the test period
    s = pd.Series(obs, index=index)
    wy = (s.index + pd.offsets.DateOffset(months=3)).year
    wettest = s.groupby(wy).sum().idxmax()
    zoom = (wy == wettest)
    ax = axes[1]
    ax.plot(index[zoom], obs[zoom], color="black", lw=1, label="observed")
    if hbv is not None:
        ax.plot(index[zoom], hbv[zoom], color="tab:blue", lw=1, label="HBV")
    ax.plot(index[zoom], lstm[zoom], color="tab:orange", lw=1, label="LSTM")
    ax.set_ylabel("mm/day")
    ax.set_title(f"Wettest test water year ({wettest})")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=150)


if __name__ == "__main__":
    main()
