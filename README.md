# hbv-torch

**A differentiable HBV hydrological model in PyTorch, with exact mass conservation and gradient-based calibration on CAMELS basins.**

HBV is one of the most widely used conceptual rainfall–runoff models. `hbv-torch` writes the full HBV-96 structure in PyTorch so that parameters can be learned by backpropagation instead of random search. This opens the door to hybrid models where a neural network predicts HBV parameters from catchment attributes, an approach known as differentiable parameter learning (Tsai et al., 2021; Feng et al., 2022).

- **Differentiable:** every step of the model is a PyTorch operation, so gradients of any loss reach all 15 parameters.
- **Mass-conserving:** precipitation = evapotranspiration + runoff + storage change, closed to about 10⁻¹² mm over 16 years, and checked in the tests.
- **Batched:** simulates many basins at once, ready for large-sample hydrology (CAMELS, Caravan).
- **Physically bounded:** parameters are learned in an unconstrained space and mapped to published ranges with a sigmoid.

## Model structure

```
 precipitation, temperature                     PET
          │                                       │
   ┌──────▼──────┐  melt / rain   ┌──────────────▼─────┐  recharge  ┌────────────────────┐
   │ Snow routine│───────────────▶│   Soil moisture    │───────────▶│ Upper zone (SUZ)   │──▶ Q0 (above UZL), Q1
   │ degree-day  │                │ (FC, LP, BETA)      │            └─────────┬──────────┘
   └─────────────┘                └────────────────────┘                     │ PERC
                                                                    ┌─────────▼──────────┐
                                                                    │ Lower zone (SLZ)   │──▶ Q2 (baseflow)
                                                                    └────────────────────┘
                                         Q0 + Q1 + Q2 ──▶ triangular routing (MAXBAS) ──▶ Q outlet
```

| Routine | Parameters | Equations |
|---|---|---|
| Snow | TT, CFMAX, SFCF, CFR, CWH | snow if T < TT; melt = CFMAX·(T − TT); refreeze = CFR·CFMAX·(TT − T); snowpack holds CWH·snow of liquid water |
| Soil | FC, LP, BETA | recharge = input·(SM/FC)^BETA; ET = PET·min(SM/(LP·FC), 1) |
| Response | PERC, UZL, K0, K1, K2 | Q0 = K0·max(SUZ − UZL, 0); Q1 = K1·SUZ; Q2 = K2·SLZ; percolation ≤ PERC |
| Routing | MAXBAS | triangular unit hydrograph with base MAXBAS days |
| Input | PCORR | rainfall correction factor |

Parameter ranges are listed in [`hbv_torch/model.py`](hbv_torch/model.py). Potential evapotranspiration is computed with the Hargreaves method (FAO-56 extraterrestrial radiation).

## Quick start

```bash
git clone https://github.com/charouk/hbv-torch.git
cd hbv-torch
pip install -e .
pytest                       # mass balance, gradients, synthetic parameter recovery
```

Download the **CAMELS time series** from [NCAR](https://ral.ucar.edu/solutions/products/camels) and unzip it so that `basin_mean_forcing/` and `usgs_streamflow/` sit in one folder. Then calibrate one basin:

```bash
python -m hbv_torch.calibrate --camels-root path/to/CAMELS --basin 01022500 --epochs 300
```

This trains on water years 2000–2008 (after a one-year warm-up), validates on 2009–2014, and writes `params.json`, `metrics.json`, `simulation.csv`, `hydrograph.png` and `training_curve.png` to `results/<basin>/`. One basin takes about 15 minutes on a laptop CPU.

Use it from Python:

```python
import torch
from hbv_torch import HBV, nse_loss

model = HBV(n_basins=1)
out = model(precip, temp, pet)          # tensors of shape [basins, days]
loss = nse_loss(qobs, out.q_routed)
loss.backward()                         # gradients for all 15 parameters
print(out.water_balance_residual())     # ~0
```

## Results

Calibration on water years 2000–2008 (after a one-year warm-up), validation on 2009–2014, 300 epochs of Adam on 1 − NSE.

| Basin | Period | NSE | KGE | PBIAS | Water-balance residual |
|---|---|---|---|---|---|
| 01022500 (Maine) | Calibration | 0.81 | 0.85 | 0.3% | — |
| 01022500 (Maine) | Validation | **0.80** | **0.78** | 1.5% | 6 × 10⁻¹² mm over 16 years |

![hydrograph](results/01022500/hydrograph.png)

Validation performance is close to calibration performance, so the gradient-based calibration generalises to unseen years. The model tends to underestimate the largest flood peaks and slightly overestimate summer low flows, two known limits of HBV's lumped soil and groundwater structure with daily forcing. Next steps are a multi-basin batch run and differentiable parameter learning (see Roadmap).

## Comparison with an LSTM

How does a conceptual model with 15 interpretable parameters compare with a data-driven LSTM on the same basin? The LSTM sees the past 365 days of forcing (precipitation, temperature, PET), **never past discharge**, so both models solve the same simulation problem. It trains on 2000–2006, uses 2007–2008 for early stopping (inside the HBV calibration period), and both models are scored on the same test period, 2009–2014. The LSTM result is the mean of 3 seeds.

```bash
python -m hbv_torch.calibrate    --camels-root path/to/CAMELS --basin 01022500
python -m hbv_torch.lstm_baseline --camels-root path/to/CAMELS --basin 01022500
```

| Test 2009–2014 | NSE | KGE | PBIAS | Peak-flow bias (top 2%) | Low-flow bias (bottom 30%) | Water balance closed? |
|---|---|---|---|---|---|---|
| Differentiable HBV | **0.80** | **0.78** | 1.5% | −30% | +59% | yes (10⁻¹¹ mm) |
| LSTM (3 seeds) | 0.72 (0.70–0.74) | 0.71 | 14.1% | −37% | −6% | no constraint |

PBIAS > 0 means the model underestimates total volume; peak- and low-flow biases are simulated minus observed volume.

<img width="1650" height="975" alt="comparison" src="https://github.com/user-attachments/assets/0503b56e-66b1-410e-9c8f-81edd4910edf" />

On this single basin with 7–9 years of training data, the differentiable HBV outperforms the LSTM (NSE 0.80 vs 0.72, above the best LSTM seed) and closes the water balance, while the unconstrained LSTM underestimates total volume by 14%. Both models underestimate the largest floods. The LSTM, however, reproduces low flows far better (−6% vs +59%): HBV's lower groundwater store drains too slowly in dry summers. Each model fails where the other succeeds, which motivates hybrid models that keep HBV's mass-conserving structure while learning the processes it represents poorly, such as baseflow. LSTMs trained on hundreds of basins are known to outperform single-basin models (Kratzert et al., 2019), so this comparison reflects a data-limited, single-basin setting.

## Tests

| Test | What it checks |
|---|---|
| `test_water_balance_closes` | inputs − outputs − Δstorage < 10⁻⁹ mm for random parameters |
| `test_states_non_negative` | no negative storages or fluxes |
| `test_every_parameter_gets_a_gradient` | backpropagation reaches the parameters |
| `test_calibration_recovers_synthetic_truth` | flow generated with known parameters is re-fitted with NSE > 0.9 |

## Roadmap

- [ ] Calibrate the 671 CAMELS basins in one batch and map parameters against catchment attributes
- [ ] Differentiable parameter learning: a neural network predicting HBV parameters from attributes
- [x] Comparison with a single-basin LSTM baseline (see above)
- [ ] Multi-basin LSTM vs. HBV across CAMELS (see [hydro-ai-foundations](https://github.com/charouk/hydro-ai-foundations))
- [ ] Uncertainty: ensembles of calibrated parameter sets

## References

- Bergström, S. (1976). *Development and application of a conceptual runoff model for Scandinavian catchments*. SMHI Report RHO 7.
- Lindström, G., et al. (1997). Development and test of the distributed HBV-96 hydrological model. *Journal of Hydrology*, 201, 272–288.
- Seibert, J., & Vis, M. J. P. (2012). Teaching hydrological modeling with a user-friendly catchment-runoff-model software package. *HESS*, 16, 3315–3325.
- Newman, A. J., et al. (2015). Development of a large-sample watershed-scale hydrometeorological data set for the contiguous USA. *HESS*, 19, 209–223.
- Tsai, W.-P., et al. (2021). From calibration to parameter learning. *Nature Communications*, 12, 5988.
- Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable, regionalized process-based models with multiphysical outputs can approach state-of-the-art hydrologic prediction accuracy. *Water Resources Research*.

## Author

**Ichrak Boubakri** — hydrological forecasting and physics-informed machine learning · [GitHub](https://github.com/charouk) · ichrak.enstab@gmail.com

This is an independent open implementation built from published equations and public data. It contains no data or code from any client project.

MIT License.
