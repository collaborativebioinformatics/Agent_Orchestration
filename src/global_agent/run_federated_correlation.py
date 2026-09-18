# global_agent/run_federated_correlation.py

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from local_agent.operations.plotting import (
    save_correlation_heatmap,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate privacy-safe local correlation statistics "
            "and generate a federated heatmap."
        )
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "Local correlation_result.json files. "
            "At least two sites are required."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--title",
        default="Federated correlation heatmap",
    )

    return parser.parse_args()


def load_local_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Local correlation result does not exist: {path}"
        )

    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"{path} must contain a JSON object."
        )

    result = value.get("result")

    if not isinstance(result, dict):
        raise ValueError(
            f"{path} does not contain a valid result object."
        )

    return value


def validate_local_results(
    local_results: list[dict[str, Any]],
) -> list[str]:
    if len(local_results) < 2:
        raise ValueError(
            "Federated correlation requires at least two sites."
        )

    expected_columns = local_results[0]["result"].get(
        "columns"
    )

    if (
        not isinstance(expected_columns, list)
        or len(expected_columns) < 2
    ):
        raise ValueError(
            "The first result has an invalid column list."
        )

    for position, local_result in enumerate(
        local_results,
        start=1,
    ):
        result = local_result["result"]
        columns = result.get("columns")

        if columns != expected_columns:
            raise ValueError(
                f"Site result {position} has a different "
                "column list or column order."
            )

        if (
            result.get("missing_data_strategy")
            != "listwise_complete_cases"
        ):
            raise ValueError(
                f"Site result {position} did not use the "
                "required listwise complete-case strategy."
            )

        count = result.get("count")

        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 2
        ):
            raise ValueError(
                f"Site result {position} has an invalid count."
            )

        dimension = len(expected_columns)

        sums = result.get("sums")
        cross_products = result.get("cross_products")

        if (
            not isinstance(sums, list)
            or len(sums) != dimension
        ):
            raise ValueError(
                f"Site result {position} has invalid sums."
            )

        if (
            not isinstance(cross_products, list)
            or len(cross_products) != dimension
            or any(
                not isinstance(row, list)
                or len(row) != dimension
                for row in cross_products
            )
        ):
            raise ValueError(
                f"Site result {position} has an invalid "
                "cross-product matrix."
            )

    return list(expected_columns)


def calculate_correlation(
    *,
    count: int,
    sums: np.ndarray,
    cross_products: np.ndarray,
) -> list[list[float | None]]:
    centered = (
        cross_products
        - np.outer(sums, sums) / count
    )

    diagonal = np.maximum(
        np.diag(centered),
        0.0,
    )

    dimension = len(sums)
    correlation_matrix: list[
        list[float | None]
    ] = []

    for row_index in range(dimension):
        row: list[float | None] = []

        for column_index in range(dimension):
            denominator = math.sqrt(
                float(diagonal[row_index])
                * float(diagonal[column_index])
            )

            if math.isclose(
                denominator,
                0.0,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                row.append(None)
                continue

            correlation = float(
                centered[row_index, column_index]
                / denominator
            )

            row.append(
                max(-1.0, min(1.0, correlation))
            )

        correlation_matrix.append(row)

    return correlation_matrix


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


def main() -> None:
    args = parse_arguments()

    local_results = [
        load_local_result(path)
        for path in args.inputs
    ]

    columns = validate_local_results(local_results)
    dimension = len(columns)

    total_count = 0
    total_sums = np.zeros(
        dimension,
        dtype=np.float64,
    )
    total_cross_products = np.zeros(
        (dimension, dimension),
        dtype=np.float64,
    )

    contributing_sites: list[str] = []

    for local_result in local_results:
        result = local_result["result"]

        total_count += int(result["count"])

        total_sums += np.asarray(
            result["sums"],
            dtype=np.float64,
        )

        total_cross_products += np.asarray(
            result["cross_products"],
            dtype=np.float64,
        )

        contributing_sites.append(
            str(
                local_result.get(
                    "site_id",
                    "unknown_site",
                )
            )
        )

    correlation_matrix = calculate_correlation(
        count=total_count,
        sums=total_sums,
        cross_products=total_cross_products,
    )

    federated_result = {
        "operation": "correlation_statistics",
        "result": {
            "columns": columns,
            "count": total_count,
            "sums": total_sums.tolist(),
            "cross_products": (
                total_cross_products.tolist()
            ),
            "correlation_matrix": correlation_matrix,
            "missing_data_strategy": (
                "listwise_complete_cases"
            ),
        },
        "contributing_sites": contributing_sites,
        "site_count": len(contributing_sites),
        "warnings": [],
    }

    output_directory = args.output_directory.resolve()
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_directory
        / "correlation_result.json"
    )

    heatmap_path = (
        output_directory
        / "correlation_heatmap.png"
    )

    save_json(
        federated_result,
        json_path,
    )

    save_correlation_heatmap(
        correlation_result=federated_result["result"],
        output_path=heatmap_path,
        title=args.title,
    )

    print("Federated correlation completed")
    print("===============================")
    print("Sites:", contributing_sites)
    print("Columns:", columns)
    print("Total complete cases:", total_count)
    print("JSON:", json_path)
    print("Heatmap:", heatmap_path)


if __name__ == "__main__":
    main()