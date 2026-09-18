# local_agent/operations/run_local_correlation.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from local_agent.operations.correlation import (
    CorrelationStatisticsOperation,
)
from local_agent.operations.metadata_loader import MetadataLoader
from local_agent.operations.metadata_validator import (
    MetadataValidator,
)
from local_agent.operations.base import (
    OperationValidationError,
)
from local_agent.operations.plotting import save_correlation_heatmap
from shared.schemas import AnalysisPlan, Operation


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local Pearson correlation analysis and save "
            "a JSON result and heatmap."
        )
    )

    parser.add_argument(
        "--site-directory",
        type=Path,
        required=True,
        help=(
            "Directory containing data.csv, catalog.json, "
            "mappings.yaml and policy.yaml."
        ),
    )

    parser.add_argument(
        "--site-id",
        required=True,
        help="Expected local site identifier.",
    )

    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Dataset identifier.",
    )

    parser.add_argument(
        "--columns",
        nargs="+",
        required=True,
        help=(
            "Canonical numeric columns to include. "
            "At least two columns are required."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help=(
            "Directory where correlation_result.json and "
            "correlation_heatmap.png will be saved."
        ),
    )

    parser.add_argument(
        "--title",
        default=None,
        help="Optional heatmap title.",
    )

    parser.add_argument(
        "--availability-tolerance",
        type=float,
        default=1.0,
        help=(
            "Allowed difference in availability percentage "
            "points between catalogue and data."
        ),
    )

    parser.add_argument(
        "--skip-metadata-validation",
        action="store_true",
        help=(
            "Skip metadata validation temporarily. "
            "Use only for development."
        ),
    )

    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Do not print correlation values inside heatmap cells.",
    )

    return parser.parse_args()


def save_json(
    value: dict[str, Any],
    output_path: Path,
) -> Path:
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

    return output_path


def print_metadata_validation(
    report,
) -> None:
    print()
    print("Metadata validation")
    print("-------------------")
    print(f"Valid: {report.valid}")
    print(f"Errors: {len(report.errors)}")
    print(f"Warnings: {len(report.warnings)}")

    for issue in report.errors:
        print(
            f"ERROR [{issue.code}]: {issue.message}"
        )

    for issue in report.warnings:
        print(
            f"WARNING [{issue.code}]: {issue.message}"
        )


def print_correlation_summary(
    *,
    operation_result,
    json_path: Path,
    heatmap_path: Path,
) -> None:
    result = operation_result.result
    metadata = operation_result.metadata

    print()
    print("Local correlation completed")
    print("===========================")

    print(
        "Canonical columns: "
        + ", ".join(result["columns"])
    )

    print(
        "Resolved local columns: "
        + ", ".join(
            metadata["resolved_local_columns"]
        )
    )

    print(
        "Source records: "
        f"{metadata['source_record_count']:,}"
    )

    print(
        "Complete cases: "
        f"{metadata['complete_case_count']:,}"
    )

    print(
        "Excluded records: "
        f"{metadata['excluded_record_count']:,}"
    )

    excluded_fraction = metadata.get(
        "excluded_fraction"
    )

    if excluded_fraction is not None:
        print(
            "Excluded fraction: "
            f"{excluded_fraction:.2%}"
        )

    if operation_result.warnings:
        print("\nOperation warnings")

        for warning in operation_result.warnings:
            print(f"- {warning}")

    print(f"\nJSON result: {json_path.resolve()}")
    print(f"Heatmap: {heatmap_path.resolve()}")


def main() -> None:
    args = parse_arguments()

    if len(args.columns) < 2:
        raise ValueError(
            "At least two columns are required."
        )

    if args.availability_tolerance < 0:
        raise ValueError(
            "--availability-tolerance cannot be negative."
        )

    print("Loading site metadata...")

    metadata_bundle = MetadataLoader().load(
        site_directory=args.site_directory,
        dataset_id=args.dataset_id,
        expected_site_id=args.site_id,
    )

    print(f"Loading data from: {metadata_bundle.data_path}")

    dataframe = pd.read_csv(
        metadata_bundle.data_path,
        low_memory=False,
    )

    print(
        f"Loaded {len(dataframe):,} rows and "
        f"{len(dataframe.columns):,} columns."
    )

    if not args.skip_metadata_validation:
        validation_report = MetadataValidator().validate(
            dataframe=dataframe,
            metadata=metadata_bundle,
            availability_tolerance_pct=(
                args.availability_tolerance
            ),
        )

        print_metadata_validation(validation_report)

        validation_report.raise_for_errors()

    plan = AnalysisPlan(
        operation=Operation.CORRELATION_STATISTICS,
        dataset_id=args.dataset_id,
        columns=args.columns,
        requested_sites=[args.site_id],
        minimum_cell_size=(
            metadata_bundle.minimum_cell_size
        ),
    )

    context = metadata_bundle.build_context()

    operation = CorrelationStatisticsOperation()

    try:
        operation_result = operation.execute(
            dataframe=dataframe,
            plan=plan,
            context=context,
        )

    except OperationValidationError as exc:
        print()
        print("Correlation analysis could not be completed.")
        print(f"Reason: {exc}")
        raise

    output_directory = args.output_directory.resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = output_directory / "correlation_result.json"
    heatmap_path = output_directory / "correlation_heatmap.png"

    serialized_result = operation_result.to_dict()

    serialized_result["site_id"] = metadata_bundle.site_id
    serialized_result["dataset_id"] = (
        metadata_bundle.dataset_id
    )

    save_json(
        serialized_result,
        json_path,
    )

    title = args.title

    if title is None:
        display_name = metadata_bundle.catalog.get(
            "display_name",
            metadata_bundle.site_id,
        )

        title = f"{display_name}: Local correlation heatmap"

    save_correlation_heatmap(
        correlation_result=operation_result.result,
        output_path=heatmap_path,
        title=title,
        annot=(
            False
            if args.no_annotations
            else None
        ),
    )

    print_correlation_summary(
        operation_result=operation_result,
        json_path=json_path,
        heatmap_path=heatmap_path,
    )


if __name__ == "__main__":
    main()