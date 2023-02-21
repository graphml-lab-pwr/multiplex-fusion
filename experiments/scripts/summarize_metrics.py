"""Script for summarizing metrics."""
import json
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import pandas as pd


def compute_stats(
    metrics: list[dict],
    key: str,
    multiplier: float = 1.0,
    precision: float = 2,
) -> str:
    values = []

    for mtr in metrics:
        v = mtr

        for k in key.split("/"):
            v = v[k]

        values.append(v * multiplier)

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


@click.command()
@click.option(
    "--metrics_dir",
    help="Path to metrics directory",
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--output_file",
    help="Path to output metrics file",
    type=click.Path(path_type=Path),
)
@click.option(
    "--paper_output_file",
    help="Path to output metrics file (in paper format)",
    type=click.Path(path_type=Path),
)
def main(metrics_dir: Path, output_file: Path, paper_output_file: Path):
    """Loads metrics from different models and saves them into a single file."""
    output_file.parent.mkdir(exist_ok=True, parents=True)

    all_metrics = defaultdict(dict)

    for dir_path in metrics_dir.glob("*/*/metrics/"):
        dataset = dir_path.parent.stem
        model = dir_path.parent.parent.stem

        mtrs = []

        for fname in dir_path.glob("*.json"):
            with fname.open("r") as fin:
                mtrs.append(json.load(fin))

        all_metrics[dataset][model] = mtrs

    # Layerwise embeddings
    for dir_path in metrics_dir.glob("layerwise/*/*/metrics/"):
        layer_name = dir_path.parent.stem
        dataset = dir_path.parent.parent.stem

        mtrs = []
        for fname in dir_path.glob("*.json"):
            with fname.open("r") as fin:
                mtrs.append(json.load(fin))

        all_metrics[dataset][f"layer_{layer_name}"] = mtrs

    # Post-hoc naive fusion
    for dir_path in metrics_dir.glob("dgi_naive_fusion/*/*/metrics/"):
        method = dir_path.parent.stem
        dataset = dir_path.parent.parent.stem

        mtrs = []
        for fname in dir_path.glob("*.json"):
            with fname.open("r") as fin:
                mtrs.append(json.load(fin))

        all_metrics[dataset][f"DGI-{method}"] = mtrs

    # Post-hoc lookup fusion
    for dir_path in metrics_dir.glob("dgi_trainable_fusion/*/*/metrics/"):
        method = dir_path.parent.stem
        dataset = dir_path.parent.parent.stem

        mtrs = []
        for fname in dir_path.glob("*.json"):
            with fname.open("r") as fin:
                mtrs.append(json.load(fin))

        all_metrics[dataset][f"DGI-{method}"] = mtrs

    pd.set_option('display.max_columns', None)

    paper_entries = []

    with output_file.open("w") as fout:
        for dataset in [
            "ACM",
            "Amazon",
            "Freebase",
            "IMDB",
            "Cora",
            "CiteSeer",
        ]:
            model_to_mtrs = all_metrics[dataset]
            fout.write(f"--- {dataset} ---\n")

            clf_entries = []
            clustering_entries = []
            sim_search_entries = []

            for model, mtrs in model_to_mtrs.items():
                # Classification
                acc = compute_stats(
                    metrics=mtrs,
                    key="classification/test/accuracy",
                    multiplier=100.,
                    precision=2,
                )
                prec = compute_stats(
                    metrics=mtrs,
                    key="classification/test/macro avg/precision",
                    multiplier=100.,
                    precision=2,
                )
                recall = compute_stats(
                    metrics=mtrs,
                    key="classification/test/macro avg/recall",
                    multiplier=100.,
                    precision=2,
                )
                macro_f1 = compute_stats(
                    metrics=mtrs,
                    key="classification/test/macro avg/f1-score",
                    multiplier=100.,
                    precision=2,
                )
                micro_f1 = compute_stats(
                    metrics=mtrs,
                    key="classification/test/micro avg/f1-score",
                    multiplier=100.,
                    precision=2,
                )
                auc = compute_stats(
                    metrics=mtrs,
                    key="classification/test/auc",
                    multiplier=100.,
                    precision=2,
                )

                clf_entries.append({
                    "model": model,
                    "accuracy": acc,
                    "macro_precision": prec,
                    "macro_recall": recall,
                    "macro_f1": macro_f1,
                    "micro_f1": micro_f1,
                    "auc": auc,
                })
                paper_entries.append({
                    "dataset": dataset,
                    "model": model,
                    "metric_name": "MaF1",
                    "metric_value": macro_f1,
                })

                # Clustering
                nmi = compute_stats(
                    metrics=mtrs,
                    key="clustering/test/NMI",
                    multiplier=100.,
                    precision=2,
                )
                ari = compute_stats(
                    metrics=mtrs,
                    key="clustering/test/ARI",
                    multiplier=1.,
                    precision=4,
                )

                clustering_entries.append({
                    "model": model,
                    "NMI": nmi,
                    "ARI": ari,
                })
                paper_entries.append({
                    "dataset": dataset,
                    "model": model,
                    "metric_name": "NMI",
                    "metric_value": nmi,
                })

                # Similarity search
                sim_5 = compute_stats(
                    metrics=mtrs,
                    key="similarity_search/test/Sim@5",
                    multiplier=100.,
                    precision=2,
                )

                sim_search_entries.append({
                    "model": model,
                    "Sim@5": sim_5,
                })
                paper_entries.append({
                    "dataset": dataset,
                    "model": model,
                    "metric_name": "Sim@5",
                    "metric_value": sim_5,
                })

            fout.write(
                pd.DataFrame
                .from_records(clf_entries)
                .set_index("model")
                .sort_values(by="macro_f1", ascending=False)
                .to_string()
                + "\n\n"
            )
            fout.write(
                pd.DataFrame
                .from_records(clustering_entries)
                .set_index("model")
                .sort_values(by="NMI", ascending=False)
                .to_string()
                + "\n\n"
            )
            fout.write(
                pd.DataFrame
                .from_records(sim_search_entries)
                .set_index("model")
                .sort_values(by="Sim@5", ascending=False)
                .to_string()
                + "\n\n\n"
            )

    paper_output_file.parent.mkdir(parents=True, exist_ok=True)
    paper_df = (
        pd.DataFrame
        .from_records(paper_entries)
        .pivot(index="model", columns=["dataset", "metric_name"])
    )

    no_fusion_names = [
        *[name for name in paper_df.index if "layer" in name],
        "node_features",
    ]
    graph_level_names = [
        "flattened_deepwalk",
        "flattened_supervised_gcn",
        "flattened_supervised_gat",
        "flattened_dgi",
        "mhgcn",
    ]
    gnn_level_names = [
        "dmgi",
        "hdgi",
        "s2mgrl",
        *[name for name in paper_df.index if "F_GBT" in name],
        *[name for name in paper_df.index if "F_DGI" in name],
    ]
    embedding_level_names = [
        "deepwalk",
        "supervised_gcn",
        "supervised_gat",
        "dgi",
        *[name for name in paper_df.index if "DGI-" in name]
    ]
    prediction_level_names = ["voting_soft", "voting_hard"]

    index_order = (
        no_fusion_names
        + graph_level_names
        + gnn_level_names
        + embedding_level_names
        + prediction_level_names
    )

    paper_df = paper_df.reindex(index_order)

    with paper_output_file.open("w") as fout:
        df_1 = paper_df.iloc[
            :,
            paper_df
            .columns
            .get_level_values("dataset")
            .isin(("ACM", "Amazon", "Freebase"))
        ]
        df_2 = paper_df.iloc[
            :,
            paper_df
            .columns
            .get_level_values("dataset")
            .isin(("IMDB", "Cora", "CiteSeer"))
        ]

        fout.write(df_1.to_string() + "\n\n")
        fout.write(df_2.to_string())


if __name__ == "__main__":
    main()
