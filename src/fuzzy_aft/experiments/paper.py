from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fuzzy_aft.experiments.common import (
    build_effective_config,
    collect_arrays_from_loader,
    predict_from_features,
    train_fixed_lambda,
)
from fuzzy_aft.experiments.methods import _build_method_runner, _build_ood_scorer
from fuzzy_aft.experiments.ood import ood_metrics, perturb, stable_rng, tail_masks
from fuzzy_aft.metrics import interval_metrics, survival_metrics
from fuzzy_aft.utils.data import SurvivalDataset, get_dataloaders


PAPER_MODELS = (
    "fuzzy_aft",
    "gaussian_hetero",
    "mcdropout",
    "coxph",
    "deep_surv",
    "deep_hit",
    "logistic_hazard",
    "rsf",
)
SCARCITY_MODELS = tuple(model for model in PAPER_MODELS if model != "coxph")
COMPONENT_MODELS = (
    "gaussian_hetero",
    "gaussian_hetero_hybrid",
    "fuzzy_aft_cnll",
    "fuzzy_aft_fixed",
)
LEVEL_MATCHED_GAMMA = 0.25852271228708085


def _loader(
    x: np.ndarray,
    s: np.ndarray,
    delta: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        SurvivalDataset(x, s, delta),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _subset(delta: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Select a censoring-stratified training subset."""
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for event in (0, 1):
        candidates = np.flatnonzero((delta > 0.5) == bool(event))
        count = max(1, round(fraction * len(candidates)))
        chosen.append(rng.choice(candidates, size=count, replace=False))
    indices = np.concatenate(chosen)
    rng.shuffle(indices)
    return indices


def _config(
    dataset: str,
    config_dir: Path,
    seeds: Iterable[int] | None,
    quick: bool,
) -> dict:
    cfg = build_effective_config(config_dir / f"{dataset}.yaml")
    if seeds is not None:
        cfg["experiment"]["seeds"] = [int(seed) for seed in seeds]
    if quick:
        cfg["experiment"]["seeds"] = [0]
        cfg["training"]["max_epochs"] = 2
        cfg["training"]["early_stopping"]["patience"] = 1
        cfg["objective"]["lambda_candidates"] = [0.01]
        for key in ("deep_surv", "deep_hit", "logistic_hazard"):
            cfg[key]["max_epochs"] = 2
            cfg[key]["patience"] = 1
        cfg["rsf"]["n_estimators"] = 10
    return cfg


def _save(rows: list[dict], output_dir: Path, experiment: str) -> Path:
    if not rows:
        raise RuntimeError(f"No rows produced for {experiment}.")
    path = output_dir / "runs" / f"{experiment}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote {path}")
    return path


def _result_row(
    dataset: str,
    model: str,
    experiment: str,
    seed: int,
    best_epoch: float,
    best_val: float,
    best_lambda: float,
    metrics: dict,
    **parameters: object,
) -> dict:
    return {
        "dataset": dataset,
        "model": model,
        "experiment": experiment,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_cNLL": best_val,
        "best_lambda": best_lambda,
        **parameters,
        **metrics,
    }


def run_in_domain(
    dataset: str,
    cfg: dict,
    models: Iterable[str],
    output_dir: Path,
) -> Path:
    rows: list[dict] = []
    for seed in cfg["experiment"]["seeds"]:
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed["experiment"]["seed"] = seed
        train, val, test, _ = get_dataloaders(cfg_seed)
        _, s_train, d_train = collect_arrays_from_loader(train)
        x_test, s_test, d_test = collect_arrays_from_loader(test)
        for model in models:
            evaluate, epoch, val_loss, selected_lambda = _build_method_runner(
                cfg_seed, model, seed, train, val, s_train, d_train, s_test, d_test,
                mc_samples=cfg["mc_dropout"]["n_samples"],
            )
            survival, intervals, extra = evaluate(x_test)
            rows.append(
                _result_row(
                    dataset, model, "in_domain", seed, epoch, val_loss,
                    selected_lambda, {**survival, **intervals, **extra}
                )
            )
    return _save(rows, output_dir, f"in_domain_{dataset}")


def run_scarcity(
    dataset: str,
    cfg: dict,
    models: Iterable[str],
    fractions: Iterable[float],
    output_dir: Path,
    *,
    experiment: str = "data_scarcity",
    level_matched_fuzzy: bool = False,
) -> Path:
    rows: list[dict] = []
    for seed in cfg["experiment"]["seeds"]:
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed["experiment"]["seed"] = seed
        train_full, val, test, _ = get_dataloaders(cfg_seed)
        x_train, s_train, d_train = collect_arrays_from_loader(train_full)
        x_val, s_val, d_val = collect_arrays_from_loader(val)
        x_test, s_test, d_test = collect_arrays_from_loader(test)
        batch_size = cfg_seed["training"]["batch_size"]
        for fraction in fractions:
            indices = _subset(d_train, fraction, seed)
            train = _loader(
                x_train[indices], s_train[indices], d_train[indices], batch_size,
                shuffle=True, seed=seed,
            )
            validation = _loader(
                x_val, s_val, d_val, batch_size, shuffle=False, seed=seed
            )
            for model in models:
                model_cfg = copy.deepcopy(cfg_seed)
                if level_matched_fuzzy and model.startswith("fuzzy_aft"):
                    model_cfg["possibility"]["gamma"] = LEVEL_MATCHED_GAMMA
                evaluate, epoch, val_loss, selected_lambda = _build_method_runner(
                    model_cfg, model, seed, train, validation,
                    s_train[indices], d_train[indices], s_test, d_test,
                    mc_samples=cfg["mc_dropout"]["n_samples"],
                )
                survival, intervals, extra = evaluate(x_test)
                rows.append(
                    _result_row(
                        dataset, model, experiment, seed, epoch, val_loss,
                        selected_lambda, {**survival, **intervals, **extra},
                        train_fraction=fraction, n_train=len(indices),
                    )
                )
    return _save(rows, output_dir, f"{experiment}_{dataset}")


def run_ood(
    dataset: str,
    cfg: dict,
    models: Iterable[str],
    modes: Iterable[str],
    output_dir: Path,
    *,
    experiment: str = "ood",
) -> Path:
    rows: list[dict] = []
    settings = cfg["ood"]
    for seed in cfg["experiment"]["seeds"]:
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed["experiment"]["seed"] = seed
        train, val, test, _ = get_dataloaders(cfg_seed)
        x_train, s_train, d_train = collect_arrays_from_loader(train)
        x_test, _, _ = collect_arrays_from_loader(test)
        non_tail, tail, threshold = tail_masks(
            x_test, settings["covariate"], settings["quantile"]
        )
        for model in models:
            score, epoch, val_loss, selected_lambda, score_meta = _build_ood_scorer(
                cfg_seed=cfg_seed,
                ood_settings={"scoring": settings["scoring"]},
                method=model,
                seed=seed,
                train_loader=train,
                val_loader=val,
                s_train=s_train,
                d_train=d_train,
                mc_samples=cfg["mc_dropout"]["n_samples"],
            )
            for mode in modes:
                if mode == "tail_holdout":
                    x_id, x_ood = x_test[non_tail], x_test[tail]
                elif mode == "perturb_only":
                    x_id = x_test
                    x_ood = perturb(x_test, stable_rng(dataset, seed, mode))
                elif mode == "combined":
                    x_id = x_test[non_tail]
                    x_ood = perturb(x_test[tail], stable_rng(dataset, seed, mode))
                else:
                    raise ValueError(f"Unknown OOD mode: {mode}")
                auroc, auprc = ood_metrics(score(x_id), score(x_ood))
                rows.append(
                    _result_row(
                        dataset, model, experiment, seed, epoch, val_loss,
                        selected_lambda,
                        {"ood_auroc": auroc, "ood_auprc": auprc},
                        evaluation_mode=mode,
                        holdout_threshold=threshold,
                        n_id=len(x_id), n_ood=len(x_ood),
                        **score_meta,
                    )
                )
    return _save(rows, output_dir, f"{experiment}_{dataset}")


def run_hyperparameter_ablation(
    dataset: str, cfg: dict, output_dir: Path
) -> Path:
    """Run the lambda and eta sweeps for one dataset."""
    rows: list[dict] = []
    settings = (
        ("lambda", (0.0, 0.1, 1.0), 1.0),
        ("eta", (0.5, 1.0, 2.0), 1.0),
    )
    for seed in cfg["experiment"]["seeds"]:
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed["experiment"]["seed"] = seed
        train, val, test, _ = get_dataloaders(cfg_seed)
        _, s_train, d_train = collect_arrays_from_loader(train)
        x_test, s_test, d_test = collect_arrays_from_loader(test)
        for parameter, values, fixed_lambda in settings:
            for value in values:
                trial = copy.deepcopy(cfg_seed)
                trial["possibility"]["eta"] = value if parameter == "eta" else 1.0
                lambda_value = value if parameter == "lambda" else fixed_lambda
                fit, best_val = train_fixed_lambda(
                    trial,
                    train,
                    val,
                    seed=seed,
                    lambda_val=lambda_value,
                    interval_variant="fuzzy_gammacut",
                    model_variant="fuzzy_aft",
                )
                mu, tau = predict_from_features(
                    fit["model"], x_test, trial["training"]["batch_size"], fit["device"]
                )
                survival = survival_metrics(
                    s_train=s_train,
                    delta_train=d_train,
                    s_test=s_test,
                    delta_test=d_test,
                    mu_test=mu,
                    tau_test=tau,
                    eta=trial["possibility"]["eta"],
                )
                intervals = interval_metrics(
                    s_test=s_test,
                    delta_test=d_test,
                    mu_test=mu,
                    tau_test=tau,
                    alpha=trial["possibility"]["alpha"],
                    eta=trial["possibility"]["eta"],
                    interval_variant="fuzzy_gammacut",
                    gamma=trial["possibility"]["gamma"],
                )
                rows.append(
                    _result_row(
                        dataset, "fuzzy_aft", "ablation", seed,
                        fit["best_epoch"], best_val, lambda_value,
                        {**survival, **intervals},
                        parameter=parameter, value=value,
                    )
                )
    return _save(rows, output_dir, f"ablation_{dataset}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the paper experiments on METABRIC and WHAS500."
    )
    parser.add_argument(
        "experiment",
        choices=("in-domain", "ood", "scarcity", "components", "ablations", "all"),
    )
    parser.add_argument("--quick", action="store_true", help="One seed and two epochs.")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configs = {
        dataset: _config(dataset, args.config_dir, args.seeds, args.quick)
        for dataset in ("metabric", "whas500")
    }
    selected = args.experiment
    if selected in {"in-domain", "all"}:
        for dataset in configs:
            run_in_domain(dataset, configs[dataset], PAPER_MODELS, args.output_dir)
    if selected in {"ood", "all"}:
        for dataset, cfg in configs.items():
            run_ood(
                dataset, cfg, PAPER_MODELS,
                ("perturb_only", "tail_holdout", "combined"), args.output_dir,
            )
    if selected in {"scarcity", "all"}:
        for dataset, cfg in configs.items():
            run_scarcity(
                dataset, cfg, SCARCITY_MODELS, (0.2, 0.5), args.output_dir,
            )
    if selected in {"components", "all"}:
        for dataset, cfg in configs.items():
            component_cfg = copy.deepcopy(cfg)
            component_cfg["objective"]["fuzzy_fixed_lambda"] = 0.01
            component_cfg["objective"]["gaussian_hybrid_lambda"] = 0.01
            run_scarcity(
                dataset, component_cfg, COMPONENT_MODELS, (0.2,), args.output_dir,
                experiment="component_scarcity", level_matched_fuzzy=True,
            )
            run_ood(
                dataset, component_cfg, COMPONENT_MODELS, ("combined",),
                args.output_dir, experiment="component_ood",
            )
    if selected in {"ablations", "all"}:
        for dataset, cfg in configs.items():
            run_hyperparameter_ablation(dataset, cfg, args.output_dir)


if __name__ == "__main__":
    main()
