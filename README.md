# Fuzzy-AFT

Fuzzy-AFT is a neural accelerated failure time model for right-censored
survival data. It learns not only when an event is expected to occur, but also
how imprecise that prediction is. The model is introduced in **Robust Survival
Analysis via Fuzzy Interval Learning** (CIKM 2026).

Survival models can remain confidently precise even when they encounter
unfamiliar patients or have only limited training data. Fuzzy-AFT addresses
this by representing each prediction as a possibility distribution whose
width is learned directly from censored observations.

For an input `x`, Fuzzy-AFT predicts a location `mu(x)` and a positive scale
`tau(x)` on log-time. Together they define the Gaussian-shaped possibility
distribution

```text
pi(s | x) = exp(-(s - mu(x))^2 / (2 * tau(x)^2)),  s = log(time).
```

The location gives the central survival-time prediction, while `tau` describes
possibilistic imprecision. A gamma-cut converts the possibility distribution
into an interval of plausible log-times. Larger intervals indicate less
precise predictions and can therefore provide a useful signal under covariate
shift or when training data are scarce.

Fuzzy-AFT learns these quantities jointly rather than attaching uncertainty
after training. Its objective combines a right-censoring-aware negative
log-likelihood with a censoring-aware interval score:

```text
loss = censored NLL + lambda * censored interval score.
```

This repository contains the reusable PyTorch model, losses, training loop,
interval construction, evaluation utilities, and the complete experiments
from the paper.

## Installation

Python 3.11 is required.

```bash
uv sync --locked --extra dev
source .venv/bin/activate
```

This installs the exact dependency versions recorded in `uv.lock`. For an
editable installation without `uv`, use `python3.11 -m venv .venv`, activate
it, and run `pip install -e '.[dev]'`; that fallback uses the versions pinned
in `pyproject.toml` but does not lock transitive dependencies.

METABRIC is downloaded by `pycox`; WHAS500 is provided by
`scikit-survival`. Raw data are not redistributed here.

## Use Fuzzy-AFT on your data

The model expects numerical covariates, positive observed durations, and an
event indicator (`1` for an observed event and `0` for right censoring).
Features should be preprocessed using statistics fitted on the training set.
The training targets are log-durations.

The following minimal example assumes that `X_train`, `time_train`,
`event_train`, and their validation counterparts are NumPy arrays:

```python
import numpy as np
import torch
from torch.utils.data import DataLoader

from fuzzy_aft.losses import fuzzy_gamma_cut_bounds
from fuzzy_aft.models import FuzzyAFTMLP
from fuzzy_aft.training import TrainingConfig, fit
from fuzzy_aft.utils.data import SurvivalDataset

train_data = SurvivalDataset(X_train, np.log(time_train), event_train)
val_data = SurvivalDataset(X_val, np.log(time_val), event_val)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
val_loader = DataLoader(val_data, batch_size=256, shuffle=False)

model = FuzzyAFTMLP(input_dim=X_train.shape[1])
result = fit(
    model,
    train_loader,
    val_loader,
    training_cfg=TrainingConfig(device="cpu"),
    eta=1.0,
    alpha=0.1,
    gamma=0.1,
    lambda_interval=0.01,
    interval_variant="fuzzy_gammacut",
)
model = result["model"].eval()

# X_new must use the same preprocessing as X_train.
with torch.no_grad():
    prediction = model(torch.as_tensor(X_new, dtype=torch.float32))
    lower, upper = fuzzy_gamma_cut_bounds(
        prediction["mu"], prediction["tau"], gamma=0.1
    )

point_time = torch.exp(prediction["mu"])
time_interval = (torch.exp(lower), torch.exp(upper))
imprecision = prediction["tau"]
```

Here, `gamma` selects the possibility cut, `alpha` controls the interval-score
penalty, `eta` controls the power-normalized likelihood surrogate, and
`lambda_interval` balances predictive fit against interval quality. In an
application, select `lambda_interval` on validation data rather than treating
the example value as a universal default.

## Reproduce the paper

The repository reproduces Tables 1--7 in the paper and also evaluates every
reported experiment on both METABRIC and WHAS500.

```bash
python reproduce.py all
python make_tables.py --compare-reference
```

The comparison command reports whether each generated paper table matches the
published values in `results/reference/`. Do not use `--quick` for this check:
quick mode intentionally uses one seed and two epochs and therefore validates
the pipeline without reproducing the published numbers.

Individual experiment groups can also be run:

```bash
python reproduce.py in-domain   # Tables 1--2
python reproduce.py ood         # Tables 3--4
python reproduce.py scarcity    # Both datasets; paper Table 5 uses WHAS500
python reproduce.py components  # Both datasets; paper Table 6 uses METABRIC
python reproduce.py ablations   # Both datasets; paper Table 7 uses METABRIC
```

All reported experiments use seeds 0--4. For an installation check without
reproducing paper values, add `--quick` (one seed, two epochs).

New per-seed results are written to the git-ignored `results/runs/`. Every run
filename ends in `_metabric` or `_whas500`, so results for one dataset never
overwrite the other. The checked-in per-seed paper-experiment runs are archived
separately in `results/paper_runs/` and are never overwritten by `reproduce.py`.

`make_tables.py` uses new runs from `results/runs/` by default. To generate
tables directly from the archived paper runs without rerunning the experiments,
use:

```bash
python make_tables.py --runs-dir results/paper_runs --compare-reference
```

The command writes the original numbered paper tables to `results/tables/` and
the complete dataset-specific tables to `results/tables/complete/`. The
published aggregate values are included in `results/reference/` for comparison.

## Table map

| Table | Experiment | Data |
|---|---|---|
| 1 | In-domain prediction | METABRIC, WHAS500 |
| 2 | In-domain prediction | WHAS500, METABRIC |
| 3 | Combined OOD detection | METABRIC, WHAS500 |
| 4 | OOD detection by mode | METABRIC, WHAS500 |
| 5 | Data scarcity (20%, 50%) | WHAS500, METABRIC |
| 6 | Component ablation | METABRIC, WHAS500 |
| 7 | Lambda and eta ablation | METABRIC, WHAS500 |

The repository also creates each experiment table for both datasets, including
dataset--experiment combinations that were not shown in the paper.

Every CSV table has a matching LaTeX file.

## Reproduction details

- Five censoring-stratified 70/15/15 splits; preprocessing is fit on training
  data only.
- Three hidden layers with 128 ReLU units, dropout 0.1, Adam at 1e-3, batch
  size 256, at most 200 epochs, and early stopping with patience 20.
- The main Fuzzy-AFT experiments use `gamma=0.1`, `alpha=0.1`, and `eta=1`.
- Lambda is selected per seed by validation cNLL from the candidates in
  `configs/base.yaml`.
- MC Dropout uses 30 stochastic passes.

### Table 6 provenance note

The published Table 6 values were produced with a level-matched 90% fuzzy cut,
`gamma=exp(-Phi^-1(0.95)^2/2)=0.25852271228708085`.

## Repository layout

```text
configs/          model and experiment settings
src/fuzzy_aft/    models, losses, training, metrics, and experiments
results/paper_runs archived per-seed paper-experiment runs
results/reference published aggregate values
results/runs       git-ignored workspace for newly generated runs
tests/            mathematical and pipeline smoke tests
reproduce.py      experiment entry point
make_tables.py    paper and complete tables from per-seed results
```

## Citation

If you use this repository, please cite:

Lennart Lamberts, Andreas Bender, Bernd Bischl, and Mina Rezaei. 2026.
**Robust Survival Analysis via Fuzzy Interval Learning.** In *Proceedings of
the 35th ACM International Conference on Information and Knowledge Management
(CIKM '26)*. Association for Computing Machinery, New York, NY, USA.
7405--7409.
<https://doi.org/10.1145/3799682.3839863>

```bibtex
@inproceedings{lamberts2026robust,
  author    = {Lennart Lamberts and Andreas Bender and Bernd Bischl and Mina Rezaei},
  title     = {Robust Survival Analysis via Fuzzy Interval Learning},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  pages     = {7405--7409},
  year      = {2026},
  doi       = {10.1145/3799682.3839863}
}
```

## License

The source code in this repository is available under the [MIT
License](LICENSE).

The CIKM 2026 paper is published through ACM's Open Access model under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)
license](https://creativecommons.org/licenses/by/4.0/). That publication
license applies to the paper; the MIT License applies to the repository's
source code.
