import math
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr

from fuzzy_aft.experiments.common import build_effective_config
from fuzzy_aft.experiments.ood import perturb, stable_rng, tail_masks
from fuzzy_aft.losses import CensoredNLLLoss, fuzzy_gamma_cut_bounds, gaussian_interval_bounds
from fuzzy_aft.metrics import test_cnll_from_arrays as array_cnll
from fuzzy_aft.models import FuzzyAFTMLP


def test_level_matched_cut_equals_central_90_percent_gaussian_interval() -> None:
    mu = torch.tensor([0.0, 1.0])
    tau = torch.tensor([1.0, 2.0])
    z95 = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.95)).item()
    gamma = math.exp(-(z95**2) / 2.0)
    fuzzy = fuzzy_gamma_cut_bounds(mu, tau, gamma=gamma)
    gaussian = gaussian_interval_bounds(mu, tau, alpha=0.1, eta=1.0)
    assert torch.allclose(fuzzy[0], gaussian[0], atol=1e-6)
    assert torch.allclose(fuzzy[1], gaussian[1], atol=1e-6)


def test_model_output_and_censored_loss() -> None:
    model = FuzzyAFTMLP(input_dim=3, hidden_sizes=(8,), dropout=0.0)
    output = model(torch.randn(5, 3))
    assert set(output) == {"mu", "tau"}
    assert output["mu"].shape == (5,)
    assert torch.all(output["tau"] > 0)
    loss = CensoredNLLLoss()(
        output["mu"], output["tau"], torch.ones(5), torch.tensor([1, 0, 1, 0, 1])
    )
    assert torch.isfinite(loss)


def test_array_cnll_is_finite_for_extreme_float32_predictions() -> None:
    value = array_cnll(
        s=np.array([100.0, 100.0], dtype=np.float32),
        delta=np.array([1.0, 0.0], dtype=np.float32),
        mu=np.zeros(2, dtype=np.float32),
        tau=np.ones(2, dtype=np.float32),
    )
    assert np.isfinite(value)


def test_array_cnll_matches_original_formula_for_finite_values() -> None:
    s = np.array([-0.5, 0.25, 1.0])
    delta = np.array([1.0, 0.0, 1.0])
    mu = np.array([-0.25, 0.0, 0.75])
    tau = np.array([0.8, 1.1, 0.9])
    z = (s - mu) / tau
    log_pdf = -0.5 * (math.log(2.0 * math.pi) + 2.0 * np.log(tau) + z * z)
    cdf = np.clip(ndtr(z), 1e-12, 1.0 - 1e-12)
    log_surv = np.log1p(-cdf)
    expected = float(-np.mean(delta * log_pdf + (1.0 - delta) * log_surv))
    assert array_cnll(s, delta, mu, tau) == expected


def test_config_contains_paper_defaults() -> None:
    root = Path(__file__).parents[1]
    cfg = build_effective_config(root / "configs" / "metabric.yaml")
    assert cfg["experiment"]["seeds"] == [0, 1, 2, 3, 4]
    assert cfg["possibility"] == {"alpha": 0.1, "gamma": 0.1, "eta": 1.0}
    assert cfg["dataset"]["split"] == {"train": 0.70, "val": 0.15, "test": 0.15}


def test_ood_protocol_is_deterministic() -> None:
    x = np.arange(40, dtype=float).reshape(10, 4)
    non_tail, tail, _ = tail_masks(x)
    assert non_tail.sum() + tail.sum() == len(x)
    first = perturb(x, stable_rng("metabric", 2, "combined"))
    second = perturb(x, stable_rng("metabric", 2, "combined"))
    assert np.array_equal(first, second)
