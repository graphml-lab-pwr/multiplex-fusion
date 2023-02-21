"""Script for extracting best hyperparams from HPS."""
import json
from pathlib import Path

import click
import pandas as pd
import yaml


@click.command()
@click.option(
    "--input_dir",
    help="Path to directory with HPS logs",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--select_by_metric",
    help="Name of metric used for selecting the best hyperparameter set",
    type=str,
)
@click.option(
    "--hps_config_file",
    help="HPS config file with default values (constants)",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--output_file",
    help="Path to output hyperparameters file (YAML file)",
    type=click.Path(path_type=Path),
)
def main(
    input_dir: Path,
    select_by_metric: str,
    hps_config_file: Path,
    output_file: Path,
):
    """Extracts the best hyperparameters for each dataset."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    hparams = {}

    for path in input_dir.glob("*.csv"):
        log = pd.read_csv(filepath_or_buffer=path)

        dataset = path.stem
        index = (
            (log[select_by_metric] == log[select_by_metric].max())
            .values
            .nonzero()[0][0]
            .item()
        )
        best_hparams = {
            k: v for k, v in log.to_dict("records")[index].items()
            if k not in ("macro_f1", "micro_f1", "auc", "accuracy")
        }

        best_hparams = {
            k: json.loads(v) if k in ("omega_intra", "omega_inter") else v
            for k, v in best_hparams.items()
        }

        with hps_config_file.open("r") as fin:
            default_values = yaml.safe_load(fin)[dataset]["constants"]

        hparams[dataset] = {**default_values, **best_hparams}

    with output_file.open("w") as fout:
        yaml.safe_dump(data=hparams, stream=fout)


if __name__ == "__main__":
    main()
