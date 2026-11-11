from pathlib import Path

import pandas as pd

import fuzzy_aft.tables as tables
from fuzzy_aft.tables import _lookup, _ood_modes, _summary, _write


def test_all_reference_tables_are_present() -> None:
    root = Path(__file__).parents[1]
    assert sorted(path.name for path in (root / "results" / "reference").glob("*.csv")) == [
        f"table_{number}.csv" for number in range(1, 8)
    ]


def test_all_paper_runs_are_archived_separately() -> None:
    root = Path(__file__).parents[1]
    assert sorted(path.name for path in (root / "results" / "paper_runs").glob("*.csv")) == [
        "ablation_metabric.csv",
        "ablation_whas500.csv",
        "component_ood_metabric.csv",
        "component_ood_whas500.csv",
        "component_scarcity_metabric.csv",
        "component_scarcity_whas500.csv",
        "data_scarcity_metabric.csv",
        "data_scarcity_whas500.csv",
        "in_domain_metabric.csv",
        "in_domain_whas500.csv",
        "ood_metabric.csv",
        "ood_whas500.csv",
    ]


def test_table_writer_aggregates_seed_results(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {"model": ["fuzzy_aft", "fuzzy_aft"], "cindex": [0.6, 0.8]}
    )
    summary = _summary(raw, ["model"], ["cindex"]).set_index("model")
    assert _lookup(summary, "fuzzy_aft", "cindex") == "0.700 ± 0.141"
    _write(pd.DataFrame([{"Model": "Fuzzy-AFT", "C-index": "0.700 ± 0.141"}]), 1, "Test", tmp_path)
    assert (tmp_path / "table_1.csv").exists()
    assert "\\begin{table}" in (tmp_path / "table_1.tex").read_text()


def test_table_4_uses_published_column_names(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    rows = []
    for model in (
        "fuzzy_aft",
        "gaussian_hetero",
        "coxph",
        "deep_surv",
        "deep_hit",
        "logistic_hazard",
        "rsf",
        "mcdropout",
    ):
        for mode in ("perturb_only", "tail_holdout"):
            for value in (0.4, 0.6):
                rows.append(
                    {
                        "model": model,
                        "evaluation_mode": mode,
                        "ood_auroc": value,
                        "ood_auprc": value,
                    }
                )
    pd.DataFrame(rows).to_csv(runs / "ood_metabric.csv", index=False)

    output = tmp_path / "tables"
    _ood_modes(runs, output)
    generated = pd.read_csv(output / "table_4.csv")
    reference = pd.read_csv(
        Path(__file__).parents[1] / "results" / "reference" / "table_4.csv"
    )
    assert generated.columns.tolist() == reference.columns.tolist()


def test_complete_tables_are_written_for_both_datasets(
    tmp_path: Path, monkeypatch
) -> None:
    frame = pd.DataFrame({"Model": ["example"], "metric": ["0.5 ± 0.1"]})
    monkeypatch.setattr(tables, "_in_domain_frame", lambda *args: frame)
    monkeypatch.setattr(tables, "_ood_modes_frame", lambda *args: frame)
    monkeypatch.setattr(tables, "_scarcity_frame", lambda *args: frame)
    monkeypatch.setattr(tables, "_components_frame", lambda *args: frame)
    monkeypatch.setattr(tables, "_ablations_frame", lambda *args: frame)

    output = tmp_path / "complete"
    tables._complete_tables(tmp_path / "runs", output)

    expected_stems = {
        f"{experiment}_{dataset}"
        for dataset in ("metabric", "whas500")
        for experiment in (
            "in_domain",
            "ood",
            "data_scarcity",
            "component_ablation",
            "hyperparameter_ablation",
        )
    }
    assert {path.stem for path in output.glob("*.csv")} == expected_stems
    assert {path.stem for path in output.glob("*.tex")} == expected_stems
