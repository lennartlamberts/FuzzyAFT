from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from fuzzy_aft.losses import GaussianCensoredNLLLoss, censored_interval_score
from fuzzy_aft.models import (
    FuzzyAFTMLP,
    GaussianAFTMLP,
)
from fuzzy_aft.training import fit_from_config, set_global_seed


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge_dict(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def build_effective_config(config_path: str, base_config_path: str | None = None) -> dict[str, Any]:
    """Load a YAML config and resolve its optional ``extends`` parent."""
    path = Path(config_path)
    cfg = load_yaml(path)
    parent = base_config_path or cfg.pop("extends", None)
    if parent is None:
        return cfg
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    base = build_effective_config(str(parent_path))
    return deep_merge_dict(base, cfg)


def lambda_candidates(cfg: dict[str, Any]) -> list[float]:
    objective_cfg = cfg.get("objective", {})
    if "lambda_candidates" in objective_cfg:
        vals = [float(v) for v in objective_cfg["lambda_candidates"]]
    else:
        vals = [0.0, 0.1, 1.0, 10.0]
    if not vals:
        raise ValueError("lambda candidate list must not be empty.")
    return vals


def build_model(cfg: dict[str, Any], input_dim: int, model_variant: str = "fuzzy_aft") -> FuzzyAFTMLP:
    variant = str(model_variant).strip().lower()
    if variant != "fuzzy_aft":
        raise ValueError(f"Unknown fuzzy model variant '{model_variant}'.")

    return FuzzyAFTMLP(
        input_dim=input_dim,
        hidden_sizes=tuple(cfg.get("model", {}).get("hidden_sizes", [128, 128, 128])),
        dropout=float(cfg.get("model", {}).get("dropout", 0.1)),
    )


def build_gaussian_model(cfg: dict[str, Any], input_dim: int) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    gauss_cfg = cfg.get("gaussian", {})
    dropout = float(gauss_cfg.get("dropout", model_cfg.get("dropout", 0.1)))
    hidden_sizes = tuple(model_cfg.get("hidden_sizes", [128, 128, 128]))
    sigma_param = str(gauss_cfg.get("sigma_param", "log"))
    return GaussianAFTMLP(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        sigma_param=sigma_param,
    )


def train_select_lambda(
    cfg_seed: dict[str, Any],
    train_loader,
    val_loader,
    seed: int,
    *,
    interval_variant: str = "gaussian_surrogate",
    model_variant: str = "fuzzy_aft",
) -> tuple[dict[str, Any], float, float]:
    first_batch = next(iter(train_loader))
    x0 = first_batch[0]
    input_dim = int(x0.shape[1])
    best_fit: dict[str, Any] | None = None
    best_lambda: float | None = None
    best_val: float | None = None

    for lambda_val in lambda_candidates(cfg_seed):
        cfg_try = copy.deepcopy(cfg_seed)
        cfg_try["objective"]["lambda"] = float(lambda_val)
        cfg_try["objective"]["interval_variant"] = str(interval_variant)

        set_global_seed(int(seed))
        model = build_model(cfg_try, input_dim=input_dim, model_variant=model_variant)
        fit_out = fit_from_config(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg_try,
            seed=seed,
        )
        current_val = float(fit_out["best_value"])
        if best_val is None or current_val < best_val:
            best_fit = fit_out
            best_lambda = float(lambda_val)
            best_val = current_val

    if best_fit is None or best_lambda is None or best_val is None:
        raise RuntimeError("Lambda selection failed: no candidate produced a result.")
    return best_fit, best_lambda, best_val


def train_fixed_lambda(
    cfg_seed: dict[str, Any],
    train_loader,
    val_loader,
    *,
    seed: int,
    lambda_val: float,
    interval_variant: str = "gaussian_surrogate",
    model_variant: str = "fuzzy_aft",
) -> tuple[dict[str, Any], float]:
    cfg_try = copy.deepcopy(cfg_seed)
    cfg_try.setdefault("objective", {})
    cfg_try["objective"]["lambda"] = float(lambda_val)
    cfg_try["objective"]["interval_variant"] = str(interval_variant)

    first_batch = next(iter(train_loader))
    x0 = first_batch[0]
    input_dim = int(x0.shape[1])

    set_global_seed(int(seed))
    model = build_model(cfg_try, input_dim=input_dim, model_variant=model_variant)
    fit_out = fit_from_config(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg_try,
        seed=seed,
    )
    return fit_out, float(fit_out["best_value"])


@torch.no_grad()
def predict_from_loader(model: torch.nn.Module, loader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    dev = torch.device(device)
    s_all: list[np.ndarray] = []
    d_all: list[np.ndarray] = []
    mu_all: list[np.ndarray] = []
    tau_all: list[np.ndarray] = []

    for x, s, delta in loader:
        x = x.to(dev)
        out = model(x)
        mu_all.append(out["mu"].detach().cpu().numpy())
        tau_all.append(out["tau"].detach().cpu().numpy())
        s_all.append(s.numpy())
        d_all.append(delta.numpy())

    return (
        np.concatenate(s_all, axis=0),
        np.concatenate(d_all, axis=0),
        np.concatenate(mu_all, axis=0),
        np.concatenate(tau_all, axis=0),
    )


@torch.no_grad()
def predict_from_features(
    model: torch.nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: str,
    *,
    force_eval: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if force_eval:
        model.eval()
    dev = torch.device(device)
    x_t = torch.tensor(x, dtype=torch.float32)

    mu_all: list[np.ndarray] = []
    tau_all: list[np.ndarray] = []
    for start in range(0, x_t.shape[0], batch_size):
        xb = x_t[start : start + batch_size].to(dev)
        out = model(xb)
        mu_all.append(out["mu"].detach().cpu().numpy())
        tau_all.append(out["tau"].detach().cpu().numpy())
    return np.concatenate(mu_all, axis=0), np.concatenate(tau_all, axis=0)


def collect_targets_from_loader(loader) -> tuple[np.ndarray, np.ndarray]:
    s_all: list[np.ndarray] = []
    d_all: list[np.ndarray] = []
    for _, s, delta in loader:
        s_all.append(s.numpy())
        d_all.append(delta.numpy())
    return np.concatenate(s_all, axis=0), np.concatenate(d_all, axis=0)


def collect_arrays_from_loader(loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_all: list[np.ndarray] = []
    s_all: list[np.ndarray] = []
    d_all: list[np.ndarray] = []
    for x, s, delta in loader:
        x_all.append(x.numpy())
        s_all.append(s.numpy())
        d_all.append(delta.numpy())
    return np.concatenate(x_all, axis=0), np.concatenate(s_all, axis=0), np.concatenate(d_all, axis=0)


def train_gaussian_model(
    cfg_seed: dict[str, Any],
    train_loader,
    val_loader,
    seed: int,
) -> tuple[torch.nn.Module, str, int, float]:
    input_dim = int(next(iter(train_loader))[0].shape[1])
    set_global_seed(int(seed))
    model = build_gaussian_model(cfg_seed, input_dim=input_dim)
    device = torch.device(str(cfg_seed.get("training", {}).get("device", "cpu")))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg_seed["training"].get("learning_rate", 1e-3)))
    loss_fn = GaussianCensoredNLLLoss()

    es_cfg = cfg_seed.get("training", {}).get("early_stopping", {})
    patience = int(es_cfg.get("patience", 20))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    max_epochs = int(cfg_seed.get("training", {}).get("max_epochs", 200))

    best_state: dict[str, torch.Tensor] | None = None
    best_val: float | None = None
    best_epoch = -1
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, s, delta in train_loader:
            x = x.to(device)
            s = s.to(device)
            delta = delta.to(device)
            out = model(x)
            loss = loss_fn(mu=out["mu"], sigma=out["sigma"], s=s, delta=delta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        vals: list[float] = []
        with torch.no_grad():
            for x, s, delta in val_loader:
                x = x.to(device)
                s = s.to(device)
                delta = delta.to(device)
                out = model(x)
                vals.append(float(loss_fn(mu=out["mu"], sigma=out["sigma"], s=s, delta=delta).item()))
        val_loss = float(np.mean(vals)) if vals else float("inf")
        if best_val is None or val_loss < (best_val - min_delta):
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, str(device), best_epoch, float(best_val if best_val is not None else float("nan"))


def train_gaussian_hybrid_model(
    cfg_seed: dict[str, Any],
    train_loader,
    val_loader,
    seed: int,
) -> tuple[torch.nn.Module, str, int, float, float]:
    """
    Train the Gaussian AFT baseline with an interval-score regularizer.

    This is intentionally separate from train_gaussian_model so the standard
    heteroscedastic Gaussian baseline remains cNLL-only.
    """
    input_dim = int(next(iter(train_loader))[0].shape[1])
    set_global_seed(int(seed))
    model = build_gaussian_model(cfg_seed, input_dim=input_dim)
    device = torch.device(str(cfg_seed.get("training", {}).get("device", "cpu")))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg_seed["training"].get("learning_rate", 1e-3)))
    cnll_fn = GaussianCensoredNLLLoss()

    objective_cfg = cfg_seed.get("objective", {})
    possibility_cfg = cfg_seed.get("possibility", {})
    lambda_interval = float(objective_cfg.get("gaussian_hybrid_lambda", objective_cfg.get("lambda", 1.0)))
    alpha = float(possibility_cfg.get("alpha", 0.1))

    es_cfg = cfg_seed.get("training", {}).get("early_stopping", {})
    patience = int(es_cfg.get("patience", 20))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    max_epochs = int(cfg_seed.get("training", {}).get("max_epochs", 200))

    best_state: dict[str, torch.Tensor] | None = None
    best_val: float | None = None
    best_epoch = -1
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, s, delta in train_loader:
            x = x.to(device)
            s = s.to(device)
            delta = delta.to(device)
            out = model(x)
            cnll = cnll_fn(mu=out["mu"], sigma=out["sigma"], s=s, delta=delta)
            interval = censored_interval_score(
                s=s,
                delta=delta,
                mu=out["mu"],
                tau=out["sigma"],
                alpha=alpha,
                eta=1.0,
            )
            loss = cnll + lambda_interval * interval
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        vals: list[float] = []
        with torch.no_grad():
            for x, s, delta in val_loader:
                x = x.to(device)
                s = s.to(device)
                delta = delta.to(device)
                out = model(x)
                vals.append(float(cnll_fn(mu=out["mu"], sigma=out["sigma"], s=s, delta=delta).item()))
        val_cnll = float(np.mean(vals)) if vals else float("inf")
        if best_val is None or val_cnll < (best_val - min_delta):
            best_val = val_cnll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, str(device), best_epoch, float(best_val if best_val is not None else float("nan")), lambda_interval


@torch.no_grad()
def predict_gaussian_from_features(
    model: torch.nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: str,
    *,
    force_eval: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if force_eval:
        model.eval()
    dev = torch.device(device)
    xt = torch.tensor(x, dtype=torch.float32)
    mu_all: list[np.ndarray] = []
    sigma_all: list[np.ndarray] = []
    for start in range(0, xt.shape[0], batch_size):
        xb = xt[start : start + batch_size].to(dev)
        out = model(xb)
        mu_all.append(out["mu"].detach().cpu().numpy())
        sigma_all.append(out["sigma"].detach().cpu().numpy())
    return np.concatenate(mu_all, axis=0), np.concatenate(sigma_all, axis=0)


@torch.no_grad()
def predict_gaussian_trainmode_from_features(
    model: torch.nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.train()
    # Important: do not override train() with eval() inside prediction, otherwise MC-dropout is disabled.
    return predict_gaussian_from_features(model=model, x=x, batch_size=batch_size, device=device, force_eval=False)


def mc_dropout_gaussian_scores(
    model: torch.nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: str,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_samples < 2:
        raise ValueError("mc_dropout.n_samples must be >= 2.")
    mu_members: list[np.ndarray] = []
    sigma_members: list[np.ndarray] = []
    for _ in range(n_samples):
        mu_i, sigma_i = predict_gaussian_trainmode_from_features(model=model, x=x, batch_size=batch_size, device=device)
        mu_members.append(mu_i)
        sigma_members.append(sigma_i)
    mu_stack = np.stack(mu_members, axis=0)
    sigma_stack = np.stack(sigma_members, axis=0)
    mu_mean = np.mean(mu_stack, axis=0)
    v_mu = np.var(mu_stack, axis=0)
    v_sigma = np.mean(sigma_stack ** 2, axis=0)
    v_total = v_mu + v_sigma
    return mu_mean, v_mu, v_sigma, v_total


def save_seed_and_summary_outputs(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    method: str,
    task: str,
    base_dir: str = "outputs",
) -> tuple[Path, Path]:
    out_dir = Path(base_dir) / dataset.lower() / method.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_path = out_dir / f"{task}_seed_results.csv"
    summary_path = out_dir / f"{task}_summary.csv"

    df = pd.DataFrame(rows)
    if seed_path.exists():
        seed_path.unlink()
    df.to_csv(seed_path, index=False)

    numeric_cols = [c for c in df.columns if c != "seed" and pd.api.types.is_numeric_dtype(df[c])]
    summary = pd.DataFrame(
        {
            "metric": numeric_cols,
            "mean": [df[c].mean() for c in numeric_cols],
            "std": [df[c].std(ddof=1) for c in numeric_cols],
        }
    )
    if summary_path.exists():
        summary_path.unlink()
    summary.to_csv(summary_path, index=False)
    return seed_path, summary_path


def save_grouped_summary_outputs(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    method: str,
    task: str,
    group_key: str,
    base_dir: str = "outputs",
) -> Path:
    out_dir = Path(base_dir) / dataset.lower() / method.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped_path = out_dir / f"{task}_grouped_summary.csv"

    df = pd.DataFrame(rows)
    if group_key not in df.columns:
        raise ValueError(f"group_key '{group_key}' missing from rows.")

    numeric_cols = [c for c in df.columns if c not in {"seed", group_key} and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby([group_key], as_index=False)[numeric_cols].agg(["mean", "std"])
    g.columns = ["_".join([c for c in col if c]).strip("_") for col in g.columns.to_flat_index()]
    if grouped_path.exists():
        grouped_path.unlink()
    g.to_csv(grouped_path, index=False)
    return grouped_path
