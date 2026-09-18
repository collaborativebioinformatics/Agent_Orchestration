# local_agent/run_schema_discovery.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from local_agent.operations.base import LocalOperationContext
from local_agent.operations.schema_discovery import (
    SchemaDiscoveryOperation,
)
from shared.schemas import AnalysisPlan, Operation


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run privacy-safe schema discovery on one local dataset."
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
        help="Local biobank site identifier.",
    )

    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Dataset identifier.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON report.",
    )

    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help=(
            "Optional column names to profile. "
            "If omitted, all columns are profiled."
        ),
    )

    parser.add_argument(
        "--minimum-cell-size",
        type=int,
        default=10,
    )

    return parser.parse_args()


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file does not exist: {csv_path}"
        )

    if not csv_path.is_file():
        raise ValueError(
            f"CSV path is not a file: {csv_path}"
        )

    print(f"Loading: {csv_path}")

    dataframe = pd.read_csv(
        csv_path,
        low_memory=False,
    )

    print(
        f"Loaded {len(dataframe):,} rows and "
        f"{len(dataframe.columns):,} columns."
    )

    return dataframe


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


def print_report_summary(
    serialized_result: dict[str, Any],
    output_path: Path,
) -> None:
    result = serialized_result["result"]
    summary = result["summary"]

    print()
    print("Schema discovery completed")
    print("==========================")
    print(f"Site: {result['site_id']}")
    print(f"Dataset: {result['dataset_id']}")
    print(f"Rows: {result['row_count']:,}")

    print(
        "Profiled columns: "
        f"{result['profiled_column_count']:,}"
    )

    print(
        "Columns with missing values: "
        f"{summary['columns_with_missing_values']}"
    )

    print(
        "All-missing columns: "
        f"{summary['all_missing_columns']}"
    )

    print(
        "Constant columns: "
        f"{summary['constant_columns']}"
    )

    print(
        "Possible identifier columns: "
        f"{summary['possible_identifier_columns']}"
    )

    print(
        "High-cardinality columns: "
        f"{summary['high_cardinality_columns']}"
    )

    print("\nColumns")
    print("-------")

    for position, column in enumerate(
        result["columns"],
        start=1,
    ):
        missing_fraction = column["missing_fraction"]

        missing_display = (
            f"{missing_fraction:.1%}"
            if missing_fraction is not None
            else "N/A"
        )

        canonical_names = column["canonical_names"]

        canonical_display = (
            ", ".join(canonical_names)
            if canonical_names
            else "-"
        )

        print(
            f"{position:3d}. "
            f"{column['name']} | "
            f"type={column['inferred_data_type']} | "
            f"missing={missing_display} | "
            f"unique={column['unique_count']} | "
            f"role={column['suggested_role']} | "
            f"canonical={canonical_display}"
        )

    warnings = serialized_result.get("warnings", [])

    if warnings:
        print("\nWarnings")
        print("--------")

        for warning in warnings:
            print(f"- {warning}")

    print(f"\nSaved report: {output_path.resolve()}")


def main() -> None:
    args = parse_arguments()

    if args.minimum_cell_size < 1:
        raise ValueError(
            "--minimum-cell-size must be at least 1."
        )

    dataframe = load_dataframe(args.csv)

    plan = AnalysisPlan(
        operation=Operation.PROFILE_COLUMNS,
        dataset_id=args.dataset_id,
        columns=args.columns or [],
    )

    context = LocalOperationContext(
        site_id=args.site_id,
        dataset_id=args.dataset_id,
        minimum_cell_size=args.minimum_cell_size,
        column_mapping={},
        metadata={},
    )

    operation = SchemaDiscoveryOperation()

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

    print_report_summary(
        serialized_result,
        args.output,
    )


if __name__ == "__main__":
    main()