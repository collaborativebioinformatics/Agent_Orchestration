# local_agent/run_cohort_count.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from local_agent.operations.cohort import CohortCountOperation
from local_agent.operations.metadata_loader import MetadataLoader
from local_agent.operations.metadata_validator import (
    MetadataValidator,
)
from shared.schemas import (
    AnalysisPlan,
    FilterCondition,
    Operation,
)



CLI_OPERATOR_ALIASES = {
    "eq": "eq",
    "equal": "eq",
    "equals": "eq",
    "==": "eq",

    "ne": "ne",
    "not_equal": "ne",
    "not_equals": "ne",
    "!=": "ne",

    "gt": "gt",
    "greater_than": "gt",
    ">": "gt",

    "ge": "ge",
    "gte": "ge",
    "greater_than_or_equal": "ge",
    ">=": "ge",

    "lt": "lt",
    "less_than": "lt",
    "<": "lt",

    "le": "le",
    "lte": "le",
    "less_than_or_equal": "le",
    "<=": "le",

    "between": "between",
    "in": "in",
    "not_in": "not_in",
    "is_null": "is_null",
    "is_missing": "is_null",
    "is_not_null": "is_not_null",
    "is_not_missing": "is_not_null",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count a privacy-safe cohort in one local biobank."
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
        "--dataset-id",
        required=True,
        help="Dataset identifier.",
    )

    parser.add_argument(
        "--site-id",
        required=True,
        help="Expected site identifier.",
    )

    parser.add_argument(
        "--filter",
        nargs=3,
        action="append",
        metavar=("COLUMN", "OPERATOR", "VALUE"),
        help=(
            "Filter expressed as COLUMN OPERATOR VALUE. "
            "VALUE must be JSON-compatible. The option may be "
            "provided multiple times."
        ),
    )

    parser.add_argument(
        "--skip-metadata-validation",
        action="store_true",
        help=(
            "Skip metadata validation temporarily. "
            "Use only during development."
        ),
    )

    return parser.parse_args()


def parse_filter_value(raw_value: str) -> Any:
    """
    Parse command-line values as JSON.

    Examples:
        60        -> integer
        60.5      -> float
        true      -> boolean
        null      -> None
        "case"    -> string
        ["A","B"] -> list
    """
    try:
        return json.loads(raw_value)

    except json.JSONDecodeError:
        # Permit convenient unquoted strings such as:
        # --filter diagnosis equals case
        return raw_value


def create_filters(
    raw_filters: list[list[str]] | None,
) -> list[FilterCondition]:
    filters: list[FilterCondition] = []

    for column, raw_operator, raw_value in raw_filters or []:
        normalized_operator = (
            raw_operator.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        schema_operator = CLI_OPERATOR_ALIASES.get(
            normalized_operator
        )

        if schema_operator is None:
            supported = ", ".join(
                sorted(set(CLI_OPERATOR_ALIASES.values()))
            )

            raise ValueError(
                f"Unsupported filter operator "
                f"{raw_operator!r}. Supported schema operators: "
                f"{supported}."
            )

        filters.append(
            FilterCondition(
                column=column,
                operator=schema_operator,
                value=parse_filter_value(raw_value),
            )
        )

    return filters


def main() -> None:
    args = parse_arguments()

    metadata = MetadataLoader().load(
        site_directory=args.site_directory,
        dataset_id=args.dataset_id,
        expected_site_id=args.site_id,
    )

    dataframe = pd.read_csv(
        metadata.data_path,
        low_memory=False,
    )

    if not args.skip_metadata_validation:
        validation_report = MetadataValidator().validate(
            dataframe=dataframe,
            metadata=metadata,
        )

        print(
            "Metadata validation: "
            f"errors={len(validation_report.errors)}, "
            f"warnings={len(validation_report.warnings)}"
        )

        for issue in validation_report.warnings:
            print(
                f"WARNING [{issue.code}]: "
                f"{issue.message}"
            )

        validation_report.raise_for_errors()

    filters = create_filters(args.filter)

    plan = AnalysisPlan(
        operation=Operation.COHORT_COUNT,
        dataset_id=args.dataset_id,
        filters=filters,
        requested_sites=[args.site_id],
        minimum_cell_size=metadata.minimum_cell_size,
    )

    context = metadata.build_context()

    operation_result = CohortCountOperation().execute(
        dataframe=dataframe,
        plan=plan,
        context=context,
    )

    internal_count = operation_result.result["count"]

    print()
    print("Cohort count completed")
    print("======================")
    print(f"Site: {metadata.site_id}")
    print(f"Dataset: {metadata.dataset_id}")
    print(
        f"Number of filters: "
        f"{operation_result.metadata['filter_count']}"
    )

    for position, applied_filter in enumerate(
        operation_result.metadata["applied_filters"],
        start=1,
    ):
        print(
            f"Filter {position}: "
            f"{applied_filter['requested_column']} "
            f"-> {applied_filter['resolved_column']} "
            f"({applied_filter['operator']}, "
            f"derived={applied_filter['derived']})"
        )

    print(
        "Minimum cell size: "
        f"{metadata.minimum_cell_size}"
    )

    # This simulates the release rule. The exact internal count
    # should not leave the local site if it is below the threshold.
    if (
        internal_count > 0
        and internal_count < metadata.minimum_cell_size
    ):
        print("Released cohort count: SUPPRESSED")
        print(
            "Reason: the cohort is smaller than the "
            "minimum cell size."
        )
    else:
        print(f"Released cohort count: {internal_count}")


if __name__ == "__main__":
    main()