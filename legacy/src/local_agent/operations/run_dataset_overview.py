# local_agent/run_dataset_overview.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from local_agent.operations.base import LocalOperationContext
from local_agent.operations.dataset_overview import (
    DatasetOverviewOperation,
)
from shared.schemas import AnalysisPlan, Operation


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local dataset-overview operation on one "
            "biobank CSV file."
        )
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to the site's private CSV file.",
    )

    parser.add_argument(
        "--site-id",
        required=True,
        help="Unique site identifier.",
    )

    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Dataset identifier used in the analysis plan.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the local JSON report will be saved.",
    )

    parser.add_argument(
        "--minimum-cell-size",
        type=int,
        default=10,
        help="Local minimum cell size. Default: 10.",
    )

    return parser.parse_args()


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {csv_path}"
        )

    if not csv_path.is_file():
        raise ValueError(
            f"Dataset path is not a file: {csv_path}"
        )

    if csv_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, received: {csv_path.suffix}"
        )

    return pd.read_csv(
        csv_path,
        low_memory=False,
    )


def save_json(
    value: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            value,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(output_path)


def print_summary(
    result: dict[str, Any],
    output_path: Path,
) -> None:
    overview = result["result"]

    shape = overview["shape"]
    missingness = overview["missingness"]
    duplicates = overview["duplicate_rows"]
    quality = overview["quality_flags"]

    print()
    print("Dataset overview completed")
    print("--------------------------")
    print(f"Site: {overview['site_id']}")
    print(f"Dataset: {overview['dataset_id']}")
    print(f"Rows: {shape['row_count']}")
    print(f"Columns: {shape['column_count']}")

    missing_fraction = missingness["missing_fraction"]

    if missing_fraction is None:
        print("Missing cells: not applicable")
    else:
        print(
            "Missing cells: "
            f"{missingness['total_missing_cells']} "
            f"({missing_fraction:.2%})"
        )

    print(
        "Duplicate rows: "
        f"{duplicates['count']}"
    )

    print(
        "All-missing columns: "
        f"{len(quality['all_missing_columns'])}"
    )

    print(
        "Constant columns: "
        f"{len(quality['constant_columns'])}"
    )

    print(
        "High-uniqueness columns: "
        f"{len(quality['high_uniqueness_columns'])}"
    )

    print(
        "Possible identifier columns: "
        f"{quality['possible_identifier_columns']}"
    )

    print(f"Full report: {output_path.resolve()}")


def main() -> None:
    args = parse_arguments()

    if args.minimum_cell_size < 1:
        raise ValueError(
            "--minimum-cell-size must be at least 1."
        )

    dataframe = load_dataframe(args.csv)

    plan = AnalysisPlan(
        operation=Operation.DISCOVER_SCHEMA,
        dataset_id=args.dataset_id,
    )

    context = LocalOperationContext(
        site_id=args.site_id,
        dataset_id=args.dataset_id,
        minimum_cell_size=args.minimum_cell_size,
    )

    operation = DatasetOverviewOperation()

    operation_result = operation.execute(
        dataframe=dataframe,
        plan=plan,
        context=context,
    )

    serialized_result = operation_result.to_dict()

    save_json(
        serialized_result,
        args.output,
    )

    print_summary(
        serialized_result,
        args.output,
    )


if __name__ == "__main__":
    main()