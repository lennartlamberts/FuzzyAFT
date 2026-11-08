from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from fuzzy_aft.losses import CensoredNLLLoss, HybridFuzzyCutLoss, HybridLoss


@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool = True
    monitor: str = "val_cNLL"
    mode: str = "min"
    patience: int = 20
    min_delta: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    max_epochs: int = 200
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    device: str = "cpu"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _select_device(device: str | None = None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested device 'cuda' but CUDA is not available.")
    return torch.device(device)


def _to_device(batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], device: torch.device):
    x, s, delta = batch
    return x.to(device), s.to(device), delta.to(device)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    hybrid_loss: torch.nn.Module,
    cnll_loss: CensoredNLLLoss,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_cnll = 0.0
    n_samples = 0

    for batch in loader:
        x, s, delta = _to_device(batch, device)
        out = model(x)
        mu, tau = out["mu"], out["tau"]

        loss = hybrid_loss(mu=mu, tau=tau, s=s, delta=delta)
        cnll = cnll_loss(mu=mu, tau=tau, s=s, delta=delta)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_cnll += float(cnll.item()) * bs
        n_samples += bs

    return {
        "train_loss": total_loss / max(n_samples, 1),
        "train_cNLL": total_cnll / max(n_samples, 1),
    }


@torch.no_grad()
def evaluate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    hybrid_loss: torch.nn.Module,
    cnll_loss: CensoredNLLLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_cnll = 0.0
    n_samples = 0

    for batch in loader:
        x, s, delta = _to_device(batch, device)
        out = model(x)
        mu, tau = out["mu"], out["tau"]

        loss = hybrid_loss(mu=mu, tau=tau, s=s, delta=delta)
        cnll = cnll_loss(mu=mu, tau=tau, s=s, delta=delta)

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_cnll += float(cnll.item()) * bs
        n_samples += bs

    return {
        "val_loss": total_loss / max(n_samples, 1),
        "val_cNLL": total_cnll / max(n_samples, 1),
    }


def _is_improved(current: float, best: float | None, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < (best - min_delta)
    if mode == "max":
        return current > (best + min_delta)
    raise ValueError(f"Unsupported early stopping mode '{mode}'.")


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    training_cfg: TrainingConfig,
    eta: float,
    alpha: float,
    gamma: float,
    lambda_interval: float,
    interval_variant: str = "gaussian_surrogate",
) -> dict[str, Any]:
    device = _select_device(training_cfg.device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
    variant = str(interval_variant).strip().lower()
    if variant == "gaussian_surrogate":
        hybrid_loss = HybridLoss(lambda_interval=lambda_interval, eta=eta, alpha=alpha)
    elif variant in {"fuzzy_gammacut", "gammacut", "fuzzy_alphacut"}:
        hybrid_loss = HybridFuzzyCutLoss(lambda_interval=lambda_interval, eta=eta, alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unknown interval variant '{interval_variant}'.")
    cnll_loss = CensoredNLLLoss(eta=eta)

    history: list[dict[str, float]] = []
    best_value: float | None = None
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0

    es = training_cfg.early_stopping

    for epoch in range(1, training_cfg.max_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            hybrid_loss=hybrid_loss,
            cnll_loss=cnll_loss,
            device=device,
        )
        val_metrics = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            hybrid_loss=hybrid_loss,
            cnll_loss=cnll_loss,
            device=device,
        )

        metrics = {"epoch": float(epoch), **train_metrics, **val_metrics}
        history.append(metrics)

        monitored = float(metrics[es.monitor])
        if _is_improved(monitored, best_value, es.mode, es.min_delta):
            best_value = monitored
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if es.enabled and wait >= es.patience:
            break

    if best_state is not None and es.enabled:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_value": best_value,
        "device": str(device),
    }


def _parse_training_config(cfg: dict[str, Any]) -> TrainingConfig:
    tr = cfg["training"]
    es_cfg = tr.get("early_stopping", {})
    es = EarlyStoppingConfig(
        enabled=bool(es_cfg.get("enabled", True)),
        monitor=str(es_cfg.get("monitor", "val_cNLL")),
        mode=str(es_cfg.get("mode", "min")),
        patience=int(es_cfg.get("patience", 20)),
        min_delta=float(es_cfg.get("min_delta", 0.0)),
    )
    return TrainingConfig(
        learning_rate=float(tr.get("learning_rate", 1e-3)),
        max_epochs=int(tr.get("max_epochs", 200)),
        early_stopping=es,
        device=str(tr.get("device", "cpu")),
    )


def fit_from_config(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict[str, Any],
    seed: int | None = None,
) -> dict[str, Any]:
    exp_cfg = cfg.get("experiment", {})
    selected_seed = int(seed if seed is not None else exp_cfg.get("seed", exp_cfg.get("seeds", [0])[0]))
    set_global_seed(selected_seed)

    training_cfg = _parse_training_config(cfg)
    possibility_cfg = cfg["possibility"]
    objective_cfg = cfg["objective"]

    out = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_cfg=training_cfg,
        eta=float(possibility_cfg.get("eta", 1.0)),
        alpha=float(possibility_cfg.get("alpha", 0.1)),
        gamma=float(possibility_cfg.get("gamma", possibility_cfg.get("alpha", 0.1))),
        lambda_interval=float(objective_cfg.get("lambda", 1.0)),
        interval_variant=str(objective_cfg.get("interval_variant", "gaussian_surrogate")),
    )
    out["seed"] = selected_seed
    return out
