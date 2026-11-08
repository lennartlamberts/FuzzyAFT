from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_ORDER = (
    "fuzzy_aft",
    "gaussian_hetero",
    "mcdropout",
    "coxph",
    "deep_surv",
    "deep_hit",
    "logistic_hazard",
    "rsf",
)
COMPONENT_ORDER = (
    "gaussian_hetero",
    "gaussian_hetero_hybrid",
    "fuzzy_aft_cnll",
    "fuzzy_aft_fixed",
)
OOD_ORDER = tuple(model for model in MODEL_ORDER if model != "mcdropout") + ("mcdropout",)
LABEL = {
    "fuzzy_aft": "Fuzzy-AFT",
    "gaussian_hetero": "Gaussian (Hetero)",
    "mcdropout": "MC Dropout",
    "coxph": "CoxPH",
    "deep_surv": "DeepSurv",
    "deep_hit": "DeepHit",
    "logistic_hazard": "LogisticHazard",
    "rsf": "RSF",
    "gaussian_hetero_hybrid": "Gaussian AFT + IS",
    "fuzzy_aft_cnll": "Fuzzy-AFT, cNLL only",
    "fuzzy_aft_fixed": "Fuzzy-AFT + IS",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the corresponding reproduce.py experiment first."
        )
    return pd.read_csv(path)


def _summary(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    return df.groupby(by, as_index=False)[metrics].agg(["mean", "std"])


def _cell(mean: float, std: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return "-"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _lookup(summary: pd.DataFrame, keys: tuple, metric: str, digits: int = 3) -> str:
    row = summary.loc[keys]
    return _cell(float(row[(metric, "mean")]), float(row[(metric, "std")]), digits)


def _write_named(
    df: pd.DataFrame,
    stem: str,
    caption: str,
    output_dir: Path,
    label: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    columns = "l" + "c" * (len(df.columns) - 1)
    lines = [
        "\\begin{table}",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        " & ".join(str(column).replace("_", "\\_") for column in df.columns) + " \\\\",
        "\\midrule",
    ]
    for row in df.itertuples(index=False, name=None):
        values = [str(value).replace("_", "\\_").replace("±", "$\\pm$") for value in row]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (output_dir / f"{stem}.tex").write_text("\n".join(lines), encoding="utf-8")


def _write(df: pd.DataFrame, number: int, caption: str, output_dir: Path) -> None:
    _write_named(
        df, f"table_{number}", caption, output_dir, f"tab:paper-{number}"
    )


def _in_domain_frame(runs: Path, dataset: str) -> pd.DataFrame:
    data = _read(runs / f"in_domain_{dataset}.csv")
    metrics = ["cindex", "ibs", "test_cNLL"]
    summary = _summary(data, ["model"], metrics).set_index("model")
    rows = []
    for model in MODEL_ORDER:
        rows.append(
            {
                "Model": LABEL[model],
                "C-index": _lookup(summary, model, "cindex"),
                "IBS": _lookup(summary, model, "ibs"),
                "test cNLL": _lookup(summary, model, "test_cNLL"),
            }
        )
    return pd.DataFrame(rows)


def _in_domain(runs: Path, dataset: str, number: int, output: Path) -> None:
    _write(
        _in_domain_frame(runs, dataset), number,
        f"Model comparison on {dataset.upper()} for the in-domain experiment (mean $\\pm$ std).",
        output,
    )


def _ood_combined_frame(runs: Path) -> pd.DataFrame:
    frames = [_read(runs / f"ood_{dataset}.csv") for dataset in ("metabric", "whas500")]
    data = pd.concat(frames, ignore_index=True)
    data = data[data["evaluation_mode"] == "combined"]
    summary = _summary(data, ["model", "dataset"], ["ood_auroc", "ood_auprc"]).set_index(
        ["model", "dataset"]
    )
    rows = []
    for model in OOD_ORDER:
        row = {"Model": LABEL[model]}
        for dataset in ("metabric", "whas500"):
            row[f"{dataset.upper()} AUROC"] = _lookup(summary, (model, dataset), "ood_auroc")
            row[f"{dataset.upper()} AUPRC"] = _lookup(summary, (model, dataset), "ood_auprc")
        rows.append(row)
    return pd.DataFrame(rows)


def _ood_combined(runs: Path, output: Path) -> None:
    _write(
        _ood_combined_frame(runs), 3,
        "Model comparison on OOD detection (mean $\\pm$ std).", output,
    )


def _ood_modes_frame(
    runs: Path,
    dataset: str,
    modes: tuple[str, ...] = ("perturb_only", "tail_holdout"),
) -> pd.DataFrame:
    data = _read(runs / f"ood_{dataset}.csv")
    data = data[data["evaluation_mode"].isin(modes)]
    summary = _summary(data, ["model", "evaluation_mode"], ["ood_auroc", "ood_auprc"]).set_index(
        ["model", "evaluation_mode"]
    )
    rows = []
    for model in OOD_ORDER:
        row = {"Model": LABEL[model]}
        for mode in modes:
            label = mode.upper()
            row[f"{label} AUROC"] = _lookup(summary, (model, mode), "ood_auroc")
            row[f"{label} AUPRC"] = _lookup(summary, (model, mode), "ood_auprc")
        rows.append(row)
    return pd.DataFrame(rows)


def _ood_modes(runs: Path, output: Path) -> None:
    _write(
        _ood_modes_frame(runs, "metabric"), 4,
        "METABRIC OOD detection split by OOD mode (mean $\\pm$ std).", output,
    )


def _scarcity_frame(runs: Path, dataset: str) -> pd.DataFrame:
    data = _read(runs / f"data_scarcity_{dataset}.csv")
    summary = _summary(data, ["model", "train_fraction"], ["cindex", "ibs", "test_cNLL"]).set_index(
        ["model", "train_fraction"]
    )
    rows = []
    for model in (item for item in MODEL_ORDER if item != "coxph"):
        row = {"Model": LABEL[model]}
        for fraction in (0.2, 0.5):
            level = f"{round(100 * fraction)}%"
            row[f"{level} C-index"] = _lookup(summary, (model, fraction), "cindex")
            row[f"{level} IBS"] = _lookup(summary, (model, fraction), "ibs")
            row[f"{level} test cNLL"] = _lookup(summary, (model, fraction), "test_cNLL")
        rows.append(row)
    return pd.DataFrame(rows)


def _scarcity(runs: Path, output: Path) -> None:
    _write(
        _scarcity_frame(runs, "whas500"), 5,
        "Data-scarcity performance on WHAS500 (mean $\\pm$ std).", output,
    )


def _components_frame(runs: Path, dataset: str) -> pd.DataFrame:
    scarcity = _summary(
        _read(runs / f"component_scarcity_{dataset}.csv"), ["model"],
        ["cindex", "test_cNLL", "coverage_uncensored", "mean_interval_width", "censored_interval_score"],
    ).set_index("model")
    ood = _summary(
        _read(runs / f"component_ood_{dataset}.csv"), ["model"],
        ["ood_auroc", "ood_auprc"],
    ).set_index("model")
    rows = []
    for model in COMPONENT_ORDER:
        rows.append(
            {
                "Model": {
                    "gaussian_hetero": "Gaussian AFT",
                    "gaussian_hetero_hybrid": "Gaussian AFT + IS",
                    "fuzzy_aft_cnll": "Fuzzy-AFT, cNLL only",
                    "fuzzy_aft_fixed": "Fuzzy-AFT + IS",
                }[model],
                "C-index": _lookup(scarcity, model, "cindex"),
                "cNLL": _lookup(scarcity, model, "test_cNLL"),
                "Coverage": _lookup(scarcity, model, "coverage_uncensored"),
                "Width": _lookup(scarcity, model, "mean_interval_width"),
                "CIS": _lookup(scarcity, model, "censored_interval_score"),
                "OOD AUROC": _lookup(ood, model, "ood_auroc"),
                "OOD AUPRC": _lookup(ood, model, "ood_auprc"),
            }
        )
    return pd.DataFrame(rows)


def _components(runs: Path, output: Path) -> None:
    _write(
        _components_frame(runs, "metabric"), 6,
        "Component ablation on METABRIC under severe data scarcity (20\\%) and combined OOD detection (mean $\\pm$ std).",
        output,
    )


def _ablations_frame(runs: Path, dataset: str) -> pd.DataFrame:
    data = _read(runs / f"ablation_{dataset}.csv")
    summary = _summary(data, ["parameter", "value"], ["censored_interval_score"]).set_index(
        ["parameter", "value"]
    )
    rows = []
    for lambda_value, eta_value in zip((0.0, 0.1, 1.0), (0.5, 1.0, 2.0)):
        rows.append(
            {
                "lambda": lambda_value,
                "Censored IS (lambda)": _lookup(
                    summary, ("lambda", lambda_value), "censored_interval_score", 2
                ),
                "eta": eta_value,
                "Censored IS (eta)": _lookup(
                    summary, ("eta", eta_value), "censored_interval_score", 2
                ),
            }
        )
    return pd.DataFrame(rows)


def _ablations(runs: Path, output: Path) -> None:
    _write(
        _ablations_frame(runs, "metabric"), 7,
        "Ablation analysis on lambda and eta on the censored Interval Score for METABRIC.",
        output,
    )


def _complete_tables(runs: Path, output: Path) -> None:
    """Write every experiment table separately for both datasets."""
    modes = ("perturb_only", "tail_holdout", "combined")
    for dataset in ("metabric", "whas500"):
        name = dataset.upper()
        _write_named(
            _in_domain_frame(runs, dataset),
            f"in_domain_{dataset}",
            f"In-domain model comparison on {name} (mean $\\pm$ std).",
            output,
            f"tab:complete-in-domain-{dataset}",
        )
        _write_named(
            _ood_modes_frame(runs, dataset, modes),
            f"ood_{dataset}",
            f"OOD detection on {name} by evaluation mode (mean $\\pm$ std).",
            output,
            f"tab:complete-ood-{dataset}",
        )
        _write_named(
            _scarcity_frame(runs, dataset),
            f"data_scarcity_{dataset}",
            f"Data-scarcity performance on {name} (mean $\\pm$ std).",
            output,
            f"tab:complete-scarcity-{dataset}",
        )
        _write_named(
            _components_frame(runs, dataset),
            f"component_ablation_{dataset}",
            f"Component ablation on {name} (mean $\\pm$ std).",
            output,
            f"tab:complete-components-{dataset}",
        )
        _write_named(
            _ablations_frame(runs, dataset),
            f"hyperparameter_ablation_{dataset}",
            f"Lambda and eta ablations on {name} (mean $\\pm$ std).",
            output,
            f"tab:complete-ablations-{dataset}",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the paper tables and complete tables for both datasets."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--compare-reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runs = args.results_dir / "runs"
    output = args.results_dir / "tables"
    _in_domain(runs, "metabric", 1, output)
    _in_domain(runs, "whas500", 2, output)
    _ood_combined(runs, output)
    _ood_modes(runs, output)
    _scarcity(runs, output)
    _components(runs, output)
    _ablations(runs, output)
    _complete_tables(runs, output / "complete")
    print(f"Wrote Tables 1--7 to {output}")
    print(f"Wrote complete tables for both datasets to {output / 'complete'}")
    if args.compare_reference:
        for number in range(1, 8):
            generated = pd.read_csv(output / f"table_{number}.csv", dtype=str).fillna("-")
            reference = pd.read_csv(
                args.results_dir / "reference" / f"table_{number}.csv", dtype=str
            ).fillna("-")
            equal = generated.equals(reference)
            print(f"Table {number}: {'matches' if equal else 'DIFFERS FROM'} reference")


if __name__ == "__main__":
    main()
