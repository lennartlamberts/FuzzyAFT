from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score
import torch
from torch.utils.data import DataLoader

from fuzzy_aft.experiments.common import (
    collect_arrays_from_loader,
    mc_dropout_gaussian_scores,
    predict_from_features,
    predict_gaussian_from_features,
    train_fixed_lambda,
    train_gaussian_hybrid_model,
    train_gaussian_model,
    train_select_lambda,
)
from fuzzy_aft.losses import fuzzy_gamma_cut_bounds, gaussian_interval_bounds
from fuzzy_aft.metrics import interval_metrics, interval_metrics_from_bounds, survival_metrics
from fuzzy_aft.models import (
    CoxPHModel,
    DeepHitModel,
    DeepSurvModel,
    LogisticHazardModel,
    RandomSurvivalForestModel,
)

FUZZY_METHODS = {
    "fuzzy_aft",
    "fuzzy_aft_cnll",
    "fuzzy_aft_fixed",
}
_FLOAT_CHARS = set("0123456789+-.eE")


def _safe_float(value: Any, *, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ValueError("Expected float, got None.")
        return float(default)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    cleaned = "".join(ch for ch in str(value).strip() if ch in _FLOAT_CHARS)
    try:
        return float(cleaned)
    except ValueError:
        if default is None:
            raise ValueError(f"Could not parse float from {value!r}") from None
        return float(default)


def _survival_time_grid_from_train(
    s_train: np.ndarray, d_train: np.ndarray, *, n: int = 400, eps: float = 1e-12
) -> np.ndarray:
    t_train = np.exp(np.asarray(s_train, dtype=float).reshape(-1))
    d_train = np.asarray(d_train, dtype=int).reshape(-1)
    obs = t_train[d_train.astype(bool)]
    if obs.size < 5:
        t_min, t_max = 1.0, 2.0
    else:
        t_min = float(max(np.quantile(obs, 0.01), eps))
        t_max = float(np.quantile(obs, 0.99))
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            t_min, t_max = 1.0, 2.0
    return np.linspace(t_min, t_max, int(n)).astype(float)

def _logtime_interval_width_from_survival(surv: np.ndarray, times: np.ndarray, *, alpha: float, eps: float = 1e-12) -> np.ndarray:
    """
    Compute log-time prediction interval width via survival quantiles:
      t_q := inf{ t : F(t) >= q } with F(t)=1-S(t).
      width := log(t_{1-a/2}) - log(t_{a/2})

    Quantiles outside the evaluation grid are clamped to the grid endpoints so
    survival-curve baselines still yield comparable finite width scores.
    """
    surv = np.asarray(surv, dtype=float)
    times = np.asarray(times, dtype=float).reshape(-1)
    if surv.ndim != 2 or times.ndim != 1 or surv.shape[1] != times.shape[0]:
        raise ValueError("surv must be (n, T) and times must be (T,).")

    q_lo = float(alpha) / 2.0
    q_hi = 1.0 - float(alpha) / 2.0
    widths = np.full((surv.shape[0],), np.nan, dtype=float)

    cdf = 1.0 - np.clip(surv, eps, 1.0)
    cdf = np.maximum.accumulate(cdf, axis=1)

    for i in range(cdf.shape[0]):
        c = cdf[i]
        if not np.isfinite(c).all():
            continue
        t_lo = float(times[0]) if q_lo < float(c[0]) else float(np.interp(q_lo, c, times))
        t_hi = float(times[-1]) if q_hi > float(c[-1]) else float(np.interp(q_hi, c, times))
        if not (np.isfinite(t_lo) and np.isfinite(t_hi) and t_lo > 0.0 and t_hi > 0.0 and t_hi >= t_lo):
            continue
        widths[i] = float(np.log(max(t_hi, eps)) - np.log(max(t_lo, eps)))
    return widths

def _ood_score_width_from_mu_tau(
    *,
    mu: np.ndarray,
    tau: np.ndarray,
    alpha: float,
    eta: float,
    interval_variant: str,
    gamma: float | None = None,
) -> np.ndarray:
    mu_t = torch.from_numpy(np.asarray(mu, dtype=np.float64))
    tau_t = torch.from_numpy(np.asarray(tau, dtype=np.float64))
    var = str(interval_variant).strip().lower()
    if var == "gaussian_surrogate":
        lower, upper = gaussian_interval_bounds(mu=mu_t, tau=tau_t, alpha=float(alpha), eta=float(eta))
    elif var in {"fuzzy_gammacut", "gammacut", "fuzzy_alphacut"}:
        lower, upper = fuzzy_gamma_cut_bounds(mu=mu_t, tau=tau_t, gamma=float(alpha if gamma is None else gamma))
    else:
        raise ValueError(f"Unknown interval_variant '{interval_variant}'.")
    return (upper - lower).detach().cpu().numpy().reshape(-1)

def _to_structured_y(t: np.ndarray, delta: np.ndarray) -> np.ndarray:
    y = np.zeros(t.shape[0], dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = delta.astype(bool)
    y["time"] = t.astype(float)
    return y

def _ibs_from_surv_probs(
    s_train: np.ndarray,
    d_train: np.ndarray,
    s_test: np.ndarray,
    d_test: np.ndarray,
    surv_test: np.ndarray,
    eps: float = 1e-12,
) -> float:
    t_train = np.exp(s_train)
    t_test = np.exp(s_test)
    obs_train = t_train[d_train.astype(bool)]
    if obs_train.size < 5:
        return float("nan")
    t_min = float(max(np.quantile(obs_train, 0.05), eps))
    t_max = float(np.quantile(obs_train, 0.95))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return float("nan")
    times = np.linspace(t_min, t_max, surv_test.shape[1])
    y_train = _to_structured_y(t=t_train, delta=d_train)
    y_test = _to_structured_y(t=t_test, delta=d_test)
    try:
        return float(integrated_brier_score(y_train, y_test, surv_test, times))
    except Exception:
        return float("nan")

def _efficiency_flags(cfg_seed: dict[str, Any]) -> tuple[bool, bool]:
    eff = cfg_seed.get("evaluation", {}).get("efficiency", {})
    report_time = bool(eff.get("report_inference_time_per_1000", False))
    report_passes = bool(eff.get("report_forward_passes", False))
    return report_time, report_passes

def _time_per_1000(dt_s: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return float(dt_s) * 1000.0 / float(n)

def _evaluate_mu_tau(
    *,
    s_train: np.ndarray,
    d_train: np.ndarray,
    s_test: np.ndarray,
    d_test: np.ndarray,
    mu: np.ndarray,
    tau: np.ndarray,
    eta: float,
    alpha: float,
    gamma: float | None = None,
    interval_variant: str = "gaussian_surrogate",
) -> tuple[dict[str, Any], dict[str, Any]]:
    surv = survival_metrics(
        s_train=s_train,
        delta_train=d_train,
        s_test=s_test,
        delta_test=d_test,
        mu_test=mu,
        tau_test=tau,
        eta=eta,
    )
    inter = interval_metrics(
        s_test=s_test,
        delta_test=d_test,
        mu_test=mu,
        tau_test=tau,
        alpha=alpha,
        eta=eta,
        interval_variant=interval_variant,
        gamma=gamma,
    )
    return surv, inter

def _fit_fuzzy_method(
    cfg: dict[str, Any],
    method: str,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> tuple[dict[str, Any], float, float, str]:
    """Fit any Fuzzy-AFT variant and return one normalized result tuple."""
    interval_variant = "fuzzy_gammacut"
    model_variant = "fuzzy_aft"

    if method == "fuzzy_aft_cnll":
        best_lambda = 0.0
    elif method == "fuzzy_aft_fixed":
        best_lambda = float(cfg.get("objective", {}).get("fuzzy_fixed_lambda", 0.01))
    else:
        fit, best_lambda, best_val = train_select_lambda(
            cfg,
            train_loader,
            val_loader,
            seed=seed,
            interval_variant=interval_variant,
            model_variant=model_variant,
        )
        return fit, float(best_lambda), float(best_val), interval_variant

    fit, best_val = train_fixed_lambda(
        cfg,
        train_loader,
        val_loader,
        seed=seed,
        lambda_val=best_lambda,
        interval_variant=interval_variant,
        model_variant=model_variant,
    )
    return fit, best_lambda, float(best_val), interval_variant

def _build_method_runner(
    cfg_seed: dict[str, Any],
    method: str,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    s_train: np.ndarray,
    d_train: np.ndarray,
    s_test: np.ndarray,
    d_test: np.ndarray,
    mc_samples: int,
):
    batch_size = int(cfg_seed["training"]["batch_size"])
    alpha = float(cfg_seed.get("possibility", {}).get("alpha", 0.1))
    gamma = float(cfg_seed.get("possibility", {}).get("gamma", alpha))
    best_epoch = float("nan")
    best_val = float("nan")
    best_lambda = float("nan")

    if method in FUZZY_METHODS:
        fit_out, best_lambda, best_val, interval_variant = _fit_fuzzy_method(
            cfg_seed, method, seed, train_loader, val_loader
        )
        model = fit_out["model"]
        device = str(fit_out["device"])
        eta = float(cfg_seed.get("possibility", {}).get("eta", 1.0))
        best_epoch = int(fit_out["best_epoch"])
        report_time, report_passes = _efficiency_flags(cfg_seed)

        def eval_fn(x_eval: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            t0 = time.perf_counter()
            mu, tau = predict_from_features(model, x=x_eval, batch_size=batch_size, device=device)
            dt = time.perf_counter() - t0
            surv, inter = _evaluate_mu_tau(
                s_train=s_train,
                d_train=d_train,
                s_test=s_test,
                d_test=d_test,
                mu=mu,
                tau=tau,
                eta=eta,
                alpha=alpha,
                gamma=gamma if interval_variant == "fuzzy_gammacut" else None,
                interval_variant=interval_variant,
            )
            extra: dict[str, Any] = {}
            if report_time:
                extra["inference_time_per_1000"] = _time_per_1000(dt, int(x_eval.shape[0]))
            if report_passes:
                extra["forward_passes"] = 1.0
            return surv, inter, extra

        return eval_fn, best_epoch, best_val, best_lambda

    if method == "gaussian_hetero":
        model, device, best_epoch, best_val = train_gaussian_model(cfg_seed, train_loader, val_loader, seed=seed)
        report_time, report_passes = _efficiency_flags(cfg_seed)

        def eval_fn(x_eval: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            t0 = time.perf_counter()
            mu, sigma = predict_gaussian_from_features(model, x=x_eval, batch_size=batch_size, device=device)
            dt = time.perf_counter() - t0
            surv, inter = _evaluate_mu_tau(
                s_train=s_train, d_train=d_train, s_test=s_test, d_test=d_test, mu=mu, tau=sigma, eta=1.0, alpha=alpha
            )
            extra: dict[str, Any] = {}
            if report_time:
                extra["inference_time_per_1000"] = _time_per_1000(dt, int(x_eval.shape[0]))
            if report_passes:
                extra["forward_passes"] = 1.0
            return surv, inter, extra

        return eval_fn, best_epoch, best_val, best_lambda

    if method == "gaussian_hetero_hybrid":
        model, device, best_epoch, best_val, best_lambda = train_gaussian_hybrid_model(
            cfg_seed, train_loader, val_loader, seed=seed
        )
        report_time, report_passes = _efficiency_flags(cfg_seed)

        def eval_fn(x_eval: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            t0 = time.perf_counter()
            mu, sigma = predict_gaussian_from_features(model, x=x_eval, batch_size=batch_size, device=device)
            dt = time.perf_counter() - t0
            surv, inter = _evaluate_mu_tau(
                s_train=s_train, d_train=d_train, s_test=s_test, d_test=d_test, mu=mu, tau=sigma, eta=1.0, alpha=alpha
            )
            extra: dict[str, Any] = {}
            if report_time:
                extra["inference_time_per_1000"] = _time_per_1000(dt, int(x_eval.shape[0]))
            if report_passes:
                extra["forward_passes"] = 1.0
            return surv, inter, extra

        return eval_fn, best_epoch, best_val, best_lambda

    if method == "mcdropout":
        model, device, best_epoch, best_val = train_gaussian_model(cfg_seed, train_loader, val_loader, seed=seed)
        report_time, report_passes = _efficiency_flags(cfg_seed)

        def eval_fn(x_eval: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            t0 = time.perf_counter()
            mu, v_mu, v_sigma, v_total = mc_dropout_gaussian_scores(
                model=model, x=x_eval, batch_size=batch_size, device=device, n_samples=mc_samples
            )
            tau = np.sqrt(v_total)
            dt = time.perf_counter() - t0
            surv, inter = _evaluate_mu_tau(
                s_train=s_train, d_train=d_train, s_test=s_test, d_test=d_test, mu=mu, tau=tau, eta=1.0, alpha=alpha
            )
            extra = {
                "mean_v_mu": float(np.mean(v_mu)),
                "mean_v_sigma": float(np.mean(v_sigma)),
                "mean_v_total": float(np.mean(v_total)),
            }
            if report_time:
                extra["inference_time_per_1000"] = _time_per_1000(dt, int(x_eval.shape[0]))
            if report_passes:
                extra["forward_passes"] = float(mc_samples)
            return surv, inter, extra

        return eval_fn, best_epoch, best_val, best_lambda

    if method == "coxph":
        x_train, _, _ = collect_arrays_from_loader(train_loader)
        t_train = np.exp(s_train)
        cph = CoxPHModel()
        cph.fit(x_train=x_train, t_train=t_train, event_train=d_train)

        obs_train = t_train[d_train.astype(bool)]
        times = None
        if obs_train.size >= 5:
            times = np.linspace(float(np.quantile(obs_train, 0.05)), float(np.quantile(obs_train, 0.95)), 100)
        report_time, report_passes = _efficiency_flags(cfg_seed)

        def eval_fn(x_eval: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            t0 = time.perf_counter()
            risk = cph.predict_risk(x_eval)
            cidx = float(concordance_index(event_times=s_test, predicted_scores=-risk, event_observed=d_test))
            if times is not None:
                surv_mat = np.clip(cph.predict_survival(x_eval, times=times), 1e-12, 1.0)
                ibs = _ibs_from_surv_probs(s_train, d_train, s_test, d_test, surv_mat)
            else:
                ibs = float("nan")
            dt = time.perf_counter() - t0
            surv = {
                "cindex": cidx,
                "ibs": ibs,
                "test_cNLL": float("nan"),
                "coverage_uncensored": float("nan"),
                "mean_interval_width": float("nan"),
                "censored_interval_score": float("nan"),
                "lower_bound_violation_rate": float("nan"),
            }
            inter = {
                "coverage_uncensored": float("nan"),
                "mean_interval_width": float("nan"),
                "censored_interval_score": float("nan"),
                "lower_bound_violation_rate": float("nan"),
            }
            extra: dict[str, Any] = {"proxy_loglik": float(cph.log_likelihood_)}
            if report_time:
                extra["inference_time_per_1000"] = _time_per_1000(dt, int(x_eval.shape[0]))
            if report_passes:
                extra["forward_passes"] = 1.0
            return surv, inter, extra

        return eval_fn, best_epoch, best_val, best_lambda

    if method == "deep_surv":
        x_train, _, _ = collect_arrays_from_loader(train_loader)
        x_val, s_val, d_val = collect_arrays_from_loader(val_loader)
        t_train = np.exp(s_train)
        t_val = np.exp(s_val)

        ds_cfg = cfg_seed.get("deep_surv", {})
        tr_cfg = cfg_seed.get("training", {})
        es_cfg = tr_cfg.get("early_stopping", {})
        model = DeepSurvModel(
            hidden_sizes=tuple(cfg_seed.get("model", {}).get("hidden_sizes", [128, 128, 128])),
            dropout=float(ds_cfg.get("dropout", cfg_seed.get("model", {}).get("dropout", 0.1))),
            learning_rate=float(ds_cfg.get("learning_rate", tr_cfg.get("learning_rate", 1e-3))),
            batch_size=int(ds_cfg.get("batch_size", tr_cfg.get("batch_size", 256))),
            max_epochs=int(ds_cfg.get("max_epochs", tr_cfg.get("max_epochs", 200))),
            patience=int(ds_cfg.get("patience", es_cfg.get("patience", 20))),
            min_delta=float(ds_cfg.get("min_delta", es_cfg.get("min_delta", 0.0))),
            device=str(tr_cfg.get("device", "cpu")),
            random_state=int(seed),
        )
        model.fit(
            x_train=x_train,
            t_train=t_train,
            event_train=d_train,
            x_val=x_val,
            t_val=t_val,
            event_val=d_val,
        )

        if model.fit_info_ is not None:
            best_epoch = int(model.fit_info_.best_epoch)
            best_val = float(model.fit_info_.best_val_loss)

        obs_train = t_train[d_train.astype(bool)]
        times = None
        if obs_train.size >= 5:
            times = np.linspace(float(np.quantile(obs_train, 0.05)), float(np.quantile(obs_train, 0.95)), 100)
        report_time, report_passes = _efficiency_flags(cfg_seed)

        def eval_fn(x_eval: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            t0 = time.perf_counter()
            risk = model.predict_risk(x_eval)
            cidx = float(concordance_index(event_times=s_test, predicted_scores=-risk, event_observed=d_test))
            if times is not None:
                surv_mat = model.predict_survival(x_eval, times=times)
                ibs = _ibs_from_surv_probs(s_train, d_train, s_test, d_test, surv_mat)
            else:
                ibs = float("nan")
            dt = time.perf_counter() - t0
            surv = {
                "cindex": cidx,
                "ibs": ibs,
                "test_cNLL": float("nan"),
                "coverage_uncensored": float("nan"),
                "mean_interval_width": float("nan"),
                "censored_interval_score": float("nan"),
                "lower_bound_violation_rate": float("nan"),
            }
            inter = {
                "coverage_uncensored": float("nan"),
                "mean_interval_width": float("nan"),
                "censored_interval_score": float("nan"),
                "lower_bound_violation_rate": float("nan"),
            }
            extra: dict[str, Any] = {}
            if report_time:
                extra["inference_time_per_1000"] = _time_per_1000(dt, int(x_eval.shape[0]))
            if report_passes:
                extra["forward_passes"] = 1.0
            return surv, inter, extra

        return eval_fn, best_epoch, best_val, best_lambda

    if method == "deep_hit":
        x_train, _, _ = collect_arrays_from_loader(train_loader)
        x_val, s_val, d_val = collect_arrays_from_loader(val_loader)
        t_train = np.exp(s_train)
        t_val = np.exp(s_val)

        dh_cfg = cfg_seed.get("deep_hit", {})
        tr_cfg = cfg_seed.get("training", {})
        es_cfg = tr_cfg.get("early_stopping", {})
        model = DeepHitModel(
            num_time_bins=int(dh_cfg.get("num_time_bins", 32)),
            hidden_sizes=tuple(cfg_seed.get("model", {}).get("hidden_sizes", [128, 128, 128])),
            dropout=float(dh_cfg.get("dropout", cfg_seed.get("model", {}).get("dropout", 0.1))),
            learning_rate=float(dh_cfg.get("learning_rate", tr_cfg.get("learning_rate", 1e-3))),
            batch_size=int(dh_cfg.get("batch_size", tr_cfg.get("batch_size", 256))),
            max_epochs=int(dh_cfg.get("max_epochs", tr_cfg.get("max_epochs", 200))),
            patience=int(dh_cfg.get("patience", es_cfg.get("patience", 20))),
            min_delta=float(dh_cfg.get("min_delta", es_cfg.get("min_delta", 0.0))),
            device=str(tr_cfg.get("device", "cpu")),
            random_state=int(seed),
        )
        model.fit(
            x_train=x_train,
            t_train=t_train,
            event_train=d_train,
            x_val=x_val,
            t_val=t_val,
            event_val=d_val,
        )
        if model.fit_info_ is not None:
            best_epoch = int(model.fit_info_.best_epoch)
            best_val = float(model.fit_info_.best_val_loss)

        obs_train = t_train[d_train.astype(bool)]
        times = None
        if obs_train.size >= 5:
            times = np.linspace(float(np.quantile(obs_train, 0.05)), float(np.quantile(obs_train, 0.95)), 100)
        report_time, report_passes = _efficiency_flags(cfg_seed)

        def eval_fn(x_eval: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            t0 = time.perf_counter()
            pmf = model.predict_pmf(x_eval)
            if model.bin_midpoints_ is None:
                raise RuntimeError("DeepHit bins are unavailable after fitting.")
            mean_time = np.sum(pmf * model.bin_midpoints_.reshape(1, -1), axis=1)
            risk = -mean_time
            cidx = float(concordance_index(event_times=s_test, predicted_scores=-risk, event_observed=d_test))
            if times is not None:
                surv_mat = model.predict_survival(x_eval, times=times)
                ibs = _ibs_from_surv_probs(s_train, d_train, s_test, d_test, surv_mat)
            else:
                ibs = float("nan")
            dt = time.perf_counter() - t0
            surv = {
                "cindex": cidx,
                "ibs": ibs,
                "test_cNLL": float("nan"),
                "coverage_uncensored": float("nan"),
                "mean_interval_width": float("nan"),
                "censored_interval_score": float("nan"),
                "lower_bound_violation_rate": float("nan"),
            }
            inter = {
                "coverage_uncensored": float("nan"),
                "mean_interval_width": float("nan"),
                "censored_interval_score": float("nan"),
                "lower_bound_violation_rate": float("nan"),
            }
            extra: dict[str, Any] = {
                "deep_hit_num_time_bins": float(model.num_time_bins),
            }
            if report_time:
                extra["inference_time_per_1000"] = _time_per_1000(dt, int(x_eval.shape[0]))
            if report_passes:
                extra["forward_passes"] = 1.0
            return surv, inter, extra

        return eval_fn, best_epoch, best_val, best_lambda

    if method == "logistic_hazard":
        x_train, _, _ = collect_arrays_from_loader(train_loader)
        x_val, s_val, d_val = collect_arrays_from_loader(val_loader)
        t_train = np.exp(s_train)
        t_val = np.exp(s_val)

        lh_cfg = cfg_seed.get("logistic_hazard", {})
        tr_cfg = cfg_seed.get("training", {})
        es_cfg = tr_cfg.get("early_stopping", {})
        n_bins = int(lh_cfg.get("num_durations", lh_cfg.get("num_time_bins", 32)))
        disc_scheme = str(lh_cfg.get("discretization_scheme", "quantiles"))
        model = LogisticHazardModel(
            num_time_bins=n_bins,
            discretization_scheme=disc_scheme,
            hidden_sizes=tuple(cfg_seed.get("model", {}).get("hidden_sizes", [128, 128, 128])),
            dropout=float(lh_cfg.get("dropout", cfg_seed.get("model", {}).get("dropout", 0.1))),
            learning_rate=float(lh_cfg.get("learning_rate", tr_cfg.get("learning_rate", 1e-3))),
            batch_size=int(lh_cfg.get("batch_size", tr_cfg.get("batch_size", 256))),
            max_epochs=int(lh_cfg.get("max_epochs", tr_cfg.get("max_epochs", 200))),
            patience=int(lh_cfg.get("patience", es_cfg.get("patience", 20))),
            min_delta=float(lh_cfg.get("min_delta", es_cfg.get("min_delta", 0.0))),
            device=str(tr_cfg.get("device", "cpu")),
            random_state=int(seed),
        )
        model.fit(
            x_train=x_train,
            t_train=t_train,
            event_train=d_train,
            x_val=x_val,
            t_val=t_val,
            event_val=d_val,
        )
        if model.fit_info_ is not None:
            best_epoch = int(model.fit_info_.best_epoch)
            best_val = float(model.fit_info_.best_val_loss)

        obs_train = t_train[d_train.astype(bool)]
        times = None
        if obs_train.size >= 5:
            times = np.linspace(float(np.quantile(obs_train, 0.05)), float(np.quantile(obs_train, 0.95)), 100)
        report_time, report_passes = _efficiency_flags(cfg_seed)

        def eval_fn(x_eval: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            t0 = time.perf_counter()
            risk = model.predict_risk(x_eval)
            cidx = float(concordance_index(event_times=s_test, predicted_scores=-risk, event_observed=d_test))
            if times is not None:
                surv_mat = model.predict_survival(x_eval, times=times)
                ibs = _ibs_from_surv_probs(s_train, d_train, s_test, d_test, surv_mat)
            else:
                ibs = float("nan")
            dt = time.perf_counter() - t0
            surv = {
                "cindex": cidx,
                "ibs": ibs,
                "test_cNLL": float("nan"),
                "coverage_uncensored": float("nan"),
                "mean_interval_width": float("nan"),
                "censored_interval_score": float("nan"),
                "lower_bound_violation_rate": float("nan"),
            }
            inter = {
                "coverage_uncensored": float("nan"),
                "mean_interval_width": float("nan"),
                "censored_interval_score": float("nan"),
                "lower_bound_violation_rate": float("nan"),
            }
            extra: dict[str, Any] = {
                "logistic_hazard_num_time_bins": float(model.num_time_bins),
                "logistic_hazard_discretization_scheme": str(model.discretization_scheme),
            }
            if report_time:
                extra["inference_time_per_1000"] = _time_per_1000(dt, int(x_eval.shape[0]))
            if report_passes:
                extra["forward_passes"] = 1.0
            return surv, inter, extra

        return eval_fn, best_epoch, best_val, best_lambda

    if method == "rsf":
        x_train, _, _ = collect_arrays_from_loader(train_loader)
        t_train = np.exp(s_train)

        rsf_cfg = cfg_seed.get("rsf", {})
        rsf = RandomSurvivalForestModel(
            n_estimators=int(rsf_cfg.get("n_estimators", 200)),
            max_depth=rsf_cfg.get("max_depth", None),
            max_features=rsf_cfg.get("max_features", "sqrt"),
            min_samples_split=int(rsf_cfg.get("min_samples_split", 10)),
            min_samples_leaf=int(rsf_cfg.get("min_samples_leaf", 15)),
            n_jobs=rsf_cfg.get("n_jobs", -1),
            random_state=int(seed),
        )
        rsf.fit(x_train=x_train, t_train=t_train, event_train=d_train)

        obs_train = t_train[d_train.astype(bool)]
        times = None
        if obs_train.size >= 5:
            times = np.linspace(float(np.quantile(obs_train, 0.05)), float(np.quantile(obs_train, 0.95)), 100)

        report_time, report_passes = _efficiency_flags(cfg_seed)

        def eval_fn(x_eval: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            t0 = time.perf_counter()
            risk = rsf.predict_risk(x_eval)
            cidx = float(concordance_index(event_times=s_test, predicted_scores=-risk, event_observed=d_test))
            if times is not None:
                surv_mat = rsf.predict_survival(x_eval, times=times)
                ibs = _ibs_from_surv_probs(s_train, d_train, s_test, d_test, surv_mat)
            else:
                ibs = float("nan")
            dt = time.perf_counter() - t0

            surv = {
                "cindex": cidx,
                "ibs": ibs,
                "test_cNLL": float("nan"),
                "coverage_uncensored": float("nan"),
                "mean_interval_width": float("nan"),
                "censored_interval_score": float("nan"),
                "lower_bound_violation_rate": float("nan"),
            }
            inter = {
                "coverage_uncensored": float("nan"),
                "mean_interval_width": float("nan"),
                "censored_interval_score": float("nan"),
                "lower_bound_violation_rate": float("nan"),
            }
            extra: dict[str, Any] = {}
            if rsf.fit_info_ is not None:
                extra.update(
                    {
                        "rsf_n_estimators": float(rsf.fit_info_.n_estimators),
                        "rsf_min_samples_leaf": float(rsf.fit_info_.min_samples_leaf),
                        "rsf_min_samples_split": float(rsf.fit_info_.min_samples_split),
                    }
                )
            if report_time:
                extra["inference_time_per_1000"] = _time_per_1000(dt, int(x_eval.shape[0]))
            if report_passes:
                extra["forward_passes"] = 1.0
            return surv, inter, extra

        return eval_fn, best_epoch, best_val, best_lambda

    raise ValueError(f"Unknown method '{method}'")

def _build_ood_scorer(
    *,
    cfg_seed: dict[str, Any],
    ood_settings: dict[str, Any],
    method: str,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    s_train: np.ndarray,
    d_train: np.ndarray,
    mc_samples: int,
) -> tuple[callable, float, float, float, dict[str, Any]]:
    """
    Returns:
      score_fn(x_eval)->np.ndarray (higher means more OOD),
      best_epoch, best_val, best_lambda,
      meta (score_type, alpha, ...)
    """
    batch_size = int(cfg_seed["training"]["batch_size"])
    poss_cfg = cfg_seed.get("possibility", {})
    alpha = _safe_float(ood_settings.get("scoring", {}).get("alpha", poss_cfg.get("alpha", 0.1)))
    gamma = _safe_float(ood_settings.get("scoring", {}).get("gamma", poss_cfg.get("gamma", alpha)))
    score_type = str(ood_settings.get("scoring", {}).get("score_type", "width_tau")).strip().lower()
    if score_type not in {"width_tau"}:
        raise ValueError(f"Unsupported ood.scoring.score_type '{score_type}'.")

    best_epoch = float("nan")
    best_val = float("nan")
    best_lambda = float("nan")
    meta: dict[str, Any] = {"ood_score_type": score_type, "ood_alpha": float(alpha), "ood_gamma": float(gamma)}

    if method in FUZZY_METHODS:
        fit_out, best_lambda, best_val, interval_variant = _fit_fuzzy_method(
            cfg_seed, method, seed, train_loader, val_loader
        )
        model = fit_out["model"]
        device = str(fit_out["device"])
        best_epoch = float(fit_out["best_epoch"])
        eta = _safe_float(poss_cfg.get("eta", 1.0))

        def score_fn(x_eval: np.ndarray) -> np.ndarray:
            mu, tau = predict_from_features(model, x=x_eval, batch_size=batch_size, device=device)
            return _ood_score_width_from_mu_tau(
                mu=mu,
                tau=tau,
                alpha=alpha,
                eta=eta,
                interval_variant=interval_variant,
                gamma=gamma if interval_variant == "fuzzy_gammacut" else None,
            )

        return score_fn, float(best_epoch), float(best_val), float(best_lambda), meta

    if method == "gaussian_hetero":
        model, device, best_epoch, best_val = train_gaussian_model(cfg_seed, train_loader, val_loader, seed=seed)

        def score_fn(x_eval: np.ndarray) -> np.ndarray:
            mu, sigma = predict_gaussian_from_features(model, x=x_eval, batch_size=batch_size, device=device)
            return _ood_score_width_from_mu_tau(mu=mu, tau=sigma, alpha=alpha, eta=1.0, interval_variant="gaussian_surrogate")

        return score_fn, float(best_epoch), float(best_val), float(best_lambda), meta

    if method == "gaussian_hetero_hybrid":
        model, device, best_epoch, best_val, best_lambda = train_gaussian_hybrid_model(
            cfg_seed, train_loader, val_loader, seed=seed
        )

        def score_fn(x_eval: np.ndarray) -> np.ndarray:
            mu, sigma = predict_gaussian_from_features(model, x=x_eval, batch_size=batch_size, device=device)
            return _ood_score_width_from_mu_tau(mu=mu, tau=sigma, alpha=alpha, eta=1.0, interval_variant="gaussian_surrogate")

        return score_fn, float(best_epoch), float(best_val), float(best_lambda), meta

    if method == "mcdropout":
        model, device, best_epoch, best_val = train_gaussian_model(cfg_seed, train_loader, val_loader, seed=seed)

        def score_fn(x_eval: np.ndarray) -> np.ndarray:
            mu, _, _, v_total = mc_dropout_gaussian_scores(
                model=model, x=x_eval, batch_size=batch_size, device=device, n_samples=mc_samples
            )
            tau = np.sqrt(v_total)
            return _ood_score_width_from_mu_tau(mu=mu, tau=tau, alpha=alpha, eta=1.0, interval_variant="gaussian_surrogate")

        return score_fn, float(best_epoch), float(best_val), float(best_lambda), meta

    times = _survival_time_grid_from_train(s_train=s_train, d_train=d_train)

    if method == "coxph":
        x_train, _, _ = collect_arrays_from_loader(train_loader)
        t_train = np.exp(np.asarray(s_train, dtype=float).reshape(-1))
        cph = CoxPHModel()
        cph.fit(x_train=x_train, t_train=t_train, event_train=d_train)
        risk_train = cph.predict_risk(x_train)
        center = float(np.mean(risk_train))
        scale = float(np.std(risk_train) + 1e-12)
        meta["ood_score_type"] = "risk_z"
        meta["ood_risk_center"] = float(center)
        meta["ood_risk_scale"] = float(scale)

        def score_fn(x_eval: np.ndarray) -> np.ndarray:
            risk = cph.predict_risk(x_eval)
            return np.abs((risk - center) / scale).astype(float).reshape(-1)

        return score_fn, float(best_epoch), float(best_val), float(best_lambda), meta

    if method == "deep_surv":
        x_train, _, _ = collect_arrays_from_loader(train_loader)
        x_val, s_val, d_val = collect_arrays_from_loader(val_loader)
        t_train = np.exp(np.asarray(s_train, dtype=float).reshape(-1))
        t_val = np.exp(np.asarray(s_val, dtype=float).reshape(-1))

        ds_cfg = cfg_seed.get("deep_surv", {})
        tr_cfg = cfg_seed.get("training", {})
        es_cfg = tr_cfg.get("early_stopping", {})
        model = DeepSurvModel(
            hidden_sizes=tuple(cfg_seed.get("model", {}).get("hidden_sizes", [128, 128, 128])),
            dropout=float(ds_cfg.get("dropout", cfg_seed.get("model", {}).get("dropout", 0.1))),
            learning_rate=float(ds_cfg.get("learning_rate", tr_cfg.get("learning_rate", 1e-3))),
            batch_size=int(ds_cfg.get("batch_size", tr_cfg.get("batch_size", 256))),
            max_epochs=int(ds_cfg.get("max_epochs", tr_cfg.get("max_epochs", 200))),
            patience=int(ds_cfg.get("patience", es_cfg.get("patience", 20))),
            min_delta=float(ds_cfg.get("min_delta", es_cfg.get("min_delta", 0.0))),
            device=str(tr_cfg.get("device", "cpu")),
            random_state=int(seed),
        )
        model.fit(
            x_train=x_train,
            t_train=t_train,
            event_train=d_train,
            x_val=x_val,
            t_val=t_val,
            event_val=d_val,
        )
        if model.fit_info_ is not None:
            best_epoch = float(model.fit_info_.best_epoch)
            best_val = float(model.fit_info_.best_val_loss)
        risk_train = model.predict_risk(x_train)
        center = float(np.mean(risk_train))
        scale = float(np.std(risk_train) + 1e-12)
        meta["ood_score_type"] = "risk_z"
        meta["ood_risk_center"] = float(center)
        meta["ood_risk_scale"] = float(scale)

        def score_fn(x_eval: np.ndarray) -> np.ndarray:
            risk = model.predict_risk(x_eval)
            return np.abs((risk - center) / scale).astype(float).reshape(-1)

        return score_fn, float(best_epoch), float(best_val), float(best_lambda), meta

    if method == "deep_hit":
        x_train, _, _ = collect_arrays_from_loader(train_loader)
        x_val, s_val, d_val = collect_arrays_from_loader(val_loader)
        t_train = np.exp(np.asarray(s_train, dtype=float).reshape(-1))
        t_val = np.exp(np.asarray(s_val, dtype=float).reshape(-1))

        dh_cfg = cfg_seed.get("deep_hit", {})
        tr_cfg = cfg_seed.get("training", {})
        es_cfg = tr_cfg.get("early_stopping", {})
        model = DeepHitModel(
            num_time_bins=int(dh_cfg.get("num_time_bins", 32)),
            hidden_sizes=tuple(cfg_seed.get("model", {}).get("hidden_sizes", [128, 128, 128])),
            dropout=float(dh_cfg.get("dropout", cfg_seed.get("model", {}).get("dropout", 0.1))),
            learning_rate=float(dh_cfg.get("learning_rate", tr_cfg.get("learning_rate", 1e-3))),
            batch_size=int(dh_cfg.get("batch_size", tr_cfg.get("batch_size", 256))),
            max_epochs=int(dh_cfg.get("max_epochs", tr_cfg.get("max_epochs", 200))),
            patience=int(dh_cfg.get("patience", es_cfg.get("patience", 20))),
            min_delta=float(dh_cfg.get("min_delta", es_cfg.get("min_delta", 0.0))),
            device=str(tr_cfg.get("device", "cpu")),
            random_state=int(seed),
        )
        model.fit(
            x_train=x_train,
            t_train=t_train,
            event_train=d_train,
            x_val=x_val,
            t_val=t_val,
            event_val=d_val,
        )
        if model.fit_info_ is not None:
            best_epoch = float(model.fit_info_.best_epoch)
            best_val = float(model.fit_info_.best_val_loss)
        meta["ood_score_survival_grid_n"] = float(times.size)
        meta["deep_hit_num_time_bins"] = float(model.num_time_bins)

        def score_fn(x_eval: np.ndarray) -> np.ndarray:
            surv = model.predict_survival(x_eval, times=times)
            return _logtime_interval_width_from_survival(surv, times, alpha=alpha)

        return score_fn, float(best_epoch), float(best_val), float(best_lambda), meta

    if method == "logistic_hazard":
        x_train, _, _ = collect_arrays_from_loader(train_loader)
        x_val, s_val, d_val = collect_arrays_from_loader(val_loader)
        t_train = np.exp(np.asarray(s_train, dtype=float).reshape(-1))
        t_val = np.exp(np.asarray(s_val, dtype=float).reshape(-1))

        lh_cfg = cfg_seed.get("logistic_hazard", {})
        tr_cfg = cfg_seed.get("training", {})
        es_cfg = tr_cfg.get("early_stopping", {})
        n_bins = int(lh_cfg.get("num_durations", lh_cfg.get("num_time_bins", 32)))
        disc_scheme = str(lh_cfg.get("discretization_scheme", "quantiles"))
        model = LogisticHazardModel(
            num_time_bins=n_bins,
            discretization_scheme=disc_scheme,
            hidden_sizes=tuple(cfg_seed.get("model", {}).get("hidden_sizes", [128, 128, 128])),
            dropout=float(lh_cfg.get("dropout", cfg_seed.get("model", {}).get("dropout", 0.1))),
            learning_rate=float(lh_cfg.get("learning_rate", tr_cfg.get("learning_rate", 1e-3))),
            batch_size=int(lh_cfg.get("batch_size", tr_cfg.get("batch_size", 256))),
            max_epochs=int(lh_cfg.get("max_epochs", tr_cfg.get("max_epochs", 200))),
            patience=int(lh_cfg.get("patience", es_cfg.get("patience", 20))),
            min_delta=float(lh_cfg.get("min_delta", es_cfg.get("min_delta", 0.0))),
            device=str(tr_cfg.get("device", "cpu")),
            random_state=int(seed),
        )
        model.fit(
            x_train=x_train,
            t_train=t_train,
            event_train=d_train,
            x_val=x_val,
            t_val=t_val,
            event_val=d_val,
        )
        if model.fit_info_ is not None:
            best_epoch = float(model.fit_info_.best_epoch)
            best_val = float(model.fit_info_.best_val_loss)
        meta["ood_score_survival_grid_n"] = float(times.size)
        meta["logistic_hazard_num_time_bins"] = float(model.num_time_bins)
        meta["logistic_hazard_discretization_scheme"] = str(model.discretization_scheme)

        def score_fn(x_eval: np.ndarray) -> np.ndarray:
            surv = model.predict_survival(x_eval, times=times)
            return _logtime_interval_width_from_survival(surv, times, alpha=alpha)

        return score_fn, float(best_epoch), float(best_val), float(best_lambda), meta

    if method == "rsf":
        x_train, _, _ = collect_arrays_from_loader(train_loader)
        t_train = np.exp(np.asarray(s_train, dtype=float).reshape(-1))
        rsf_cfg = cfg_seed.get("rsf", {})
        rsf = RandomSurvivalForestModel(
            n_estimators=int(rsf_cfg.get("n_estimators", 200)),
            max_depth=rsf_cfg.get("max_depth", None),
            max_features=rsf_cfg.get("max_features", "sqrt"),
            min_samples_split=int(rsf_cfg.get("min_samples_split", 10)),
            min_samples_leaf=int(rsf_cfg.get("min_samples_leaf", 15)),
            n_jobs=rsf_cfg.get("n_jobs", -1),
            random_state=int(seed),
        )
        rsf.fit(x_train=x_train, t_train=t_train, event_train=d_train)
        risk_train = rsf.predict_risk(x_train)
        center = float(np.mean(risk_train))
        scale = float(np.std(risk_train) + 1e-12)
        meta["ood_score_type"] = "risk_z"
        meta["ood_risk_center"] = float(center)
        meta["ood_risk_scale"] = float(scale)

        def score_fn(x_eval: np.ndarray) -> np.ndarray:
            risk = rsf.predict_risk(x_eval)
            return np.abs((risk - center) / scale).astype(float).reshape(-1)

        return score_fn, float(best_epoch), float(best_val), float(best_lambda), meta

    raise ValueError(f"Unknown method '{method}'.")
