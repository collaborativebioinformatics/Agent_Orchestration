# local_agent/operations/correlation.py

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from local_agent.operations.base import (
    DataOperation,
    LocalOperationContext,
    LocalOperationResult,
    OperationValidationError,
)
from shared.schemas import AnalysisPlan, Operation


class CorrelationStatisticsOperation(DataOperation):
    """
    Calculate local Pearson correlations and federated sufficient statistics.

    The operation returns:

    - canonical column order;
    - local resolved column order;
    - complete-case count;
    - column sums;
    - cross-product matrix;
    - local Pearson correlation matrix.

    No raw observations or patient identifiers are returned.

    Missing-data strategy
    ---------------------
    This initial implementation uses listwise complete cases. A row
    contributes only when every requested variable is finite and observed.
    """

    operation = Operation.CORRELATION_STATISTICS
    requires_nonempty_dataframe = True

    DEFAULT_MAXIMUM_COLUMNS = 30
    ABSOLUTE_MAXIMUM_COLUMNS = 100

    def validate(
        self,
        *,
        dataframe: pd.DataFrame,
        plan: AnalysisPlan,
        context: LocalOperationContext,
        resolved_columns: list[str],
        resolved_group_by: list[str],
    ) -> None:
        del dataframe

        if plan.dataset_id != context.dataset_id:
            raise OperationValidationError(
                f"Plan dataset {plan.dataset_id!r} does not match "
                f"the loaded dataset {context.dataset_id!r}."
            )

        requested_columns = list(
            getattr(plan, "columns", []) or []
        )

        if len(requested_columns) < 2:
            raise OperationValidationError(
                "Correlation analysis requires at least two columns."
            )

        if len(resolved_columns) != len(requested_columns):
            raise OperationValidationError(
                "Multiple requested variables resolved to the same "
                "local column. Correlation variables must resolve "
                "one-to-one."
            )

        maximum_columns = self._maximum_columns(context)

        if len(requested_columns) > maximum_columns:
            raise OperationValidationError(
                f"Correlation analysis supports at most "
                f"{maximum_columns} columns at this site."
            )

        if resolved_group_by:
            raise OperationValidationError(
                "Correlation analysis does not support group_by."
            )

        if getattr(plan, "model", None) is not None:
            raise OperationValidationError(
                "Correlation analysis does not support a model request."
            )

        # We will add the reusable cohort-filter engine later.
        if getattr(plan, "filters", None):
            raise OperationValidationError(
                "Filtered correlation is not enabled yet. "
                "Run correlation without filters for the initial "
                "site-level exploration."
            )

        identifier_columns = self._declared_identifier_columns(
            context
        )

        blocked = sorted(
            column
            for column in resolved_columns
            if column in identifier_columns
        )

        if blocked:
            raise OperationValidationError(
                "Identifier columns cannot be used in correlation "
                f"analysis: {blocked}."
            )

    def _run(
        self,
        *,
        dataframe: pd.DataFrame,
        plan: AnalysisPlan,
        context: LocalOperationContext,
        resolved_columns: list[str],
        resolved_group_by: list[str],
    ) -> LocalOperationResult:
        del resolved_group_by

        canonical_columns = [
            str(column)
            for column in plan.columns
        ]

        numeric_dataframe, conversion_warnings = (
            self._prepare_numeric_dataframe(
                dataframe=dataframe,
                resolved_columns=resolved_columns,
            )
        )

        source_record_count = int(len(numeric_dataframe))

        finite_dataframe = numeric_dataframe.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        complete_dataframe = finite_dataframe.dropna(
            axis=0,
            how="any",
        )

        complete_case_count = int(len(complete_dataframe))
        excluded_record_count = (
            source_record_count - complete_case_count
        )

        if complete_case_count < context.minimum_cell_size:
            raise OperationValidationError(
                "The complete-case cohort is smaller than the "
                "local minimum cell size. Correlation statistics "
                "cannot be released."
            )

        matrix = complete_dataframe.to_numpy(
            dtype=np.float64,
            copy=True,
        )

        if not np.isfinite(matrix).all():
            raise OperationValidationError(
                "Non-finite values remain after numerical cleaning."
            )

        column_sums = matrix.sum(
            axis=0,
            dtype=np.float64,
        )

        cross_products = matrix.T @ matrix

        correlation_matrix, constant_columns = (
            self._calculate_correlation_matrix(
                count=complete_case_count,
                sums=column_sums,
                cross_products=cross_products,
                column_names=canonical_columns,
            )
        )

        warnings = list(conversion_warnings)

        if excluded_record_count > 0:
            warnings.append(
                f"{excluded_record_count} record(s) were excluded "
                "by listwise complete-case analysis."
            )

        if constant_columns:
            warnings.append(
                "Correlation is undefined for constant variables: "
                f"{constant_columns}."
            )

        missing_fraction = (
            excluded_record_count / source_record_count
            if source_record_count > 0
            else None
        )

        result: dict[str, Any] = {
            # Use canonical names here so all federated sites return
            # the same column order.
            "columns": canonical_columns,
            "count": complete_case_count,
            "sums": [
                float(value)
                for value in column_sums.tolist()
            ],
            "cross_products": [
                [
                    float(value)
                    for value in row
                ]
                for row in cross_products.tolist()
            ],
            "correlation_matrix": correlation_matrix,
            "missing_data_strategy": (
                "listwise_complete_cases"
            ),
        }

        return LocalOperationResult(
            operation=self.operation,
            result=result,
            record_count=complete_case_count,
            warnings=warnings,
            metadata={
                "resolved_local_columns": list(
                    resolved_columns
                ),
                "source_record_count": source_record_count,
                "complete_case_count": complete_case_count,
                "excluded_record_count": (
                    excluded_record_count
                ),
                "excluded_fraction": missing_fraction,
                "minimum_cell_size": (
                    context.minimum_cell_size
                ),
                "correlation_method": "pearson",
            },
        )

    @staticmethod
    def _prepare_numeric_dataframe(
        *,
        dataframe: pd.DataFrame,
        resolved_columns: list[str],
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Convert selected columns to numeric values.

        Boolean columns are converted to 0/1. Non-numeric values become
        missing and are subsequently removed by complete-case filtering.
        """
        prepared: dict[str, pd.Series] = {}
        warnings: list[str] = []

        for column in resolved_columns:
            series = dataframe[column]

            if pd.api.types.is_bool_dtype(series):
                converted = series.astype("Int64").astype(
                    "float64"
                )

            elif pd.api.types.is_numeric_dtype(series):
                converted = pd.to_numeric(
                    series,
                    errors="coerce",
                ).astype("float64")

            else:
                original_non_missing = int(
                    series.notna().sum()
                )

                converted = pd.to_numeric(
                    series,
                    errors="coerce",
                ).astype("float64")

                converted_non_missing = int(
                    converted.notna().sum()
                )

                failed_conversion_count = (
                    original_non_missing
                    - converted_non_missing
                )

                if failed_conversion_count > 0:
                    warnings.append(
                        f"Column {column!r}: "
                        f"{failed_conversion_count} non-missing "
                        "value(s) could not be converted to numeric "
                        "and were treated as missing."
                    )

            if int(converted.notna().sum()) == 0:
                raise OperationValidationError(
                    f"Column {column!r} contains no usable "
                    "numeric values."
                )

            prepared[column] = converted

        return pd.DataFrame(
            prepared,
            index=dataframe.index,
        ), warnings

    @staticmethod
    def _calculate_correlation_matrix(
        *,
        count: int,
        sums: np.ndarray,
        cross_products: np.ndarray,
        column_names: list[str],
    ) -> tuple[
        list[list[float | None]],
        list[str],
    ]:
        """
        Calculate Pearson correlation from sufficient statistics.

        C = X^T X - ss^T / n

        r_ij = C_ij / sqrt(C_ii C_jj)
        """
        dimension = len(sums)

        centered_cross_products = (
            cross_products
            - np.outer(sums, sums) / count
        )

        # Small floating-point errors can make a theoretically zero
        # diagonal slightly negative.
        diagonal = np.maximum(
            np.diag(centered_cross_products),
            0.0,
        )

        constant_indices = [
            index
            for index, value in enumerate(diagonal)
            if math.isclose(
                float(value),
                0.0,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ]

        constant_columns = [
            column_names[index]
            for index in constant_indices
        ]

        result: list[list[float | None]] = []

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
                    centered_cross_products[
                        row_index,
                        column_index,
                    ]
                    / denominator
                )

                # Prevent floating-point outputs slightly outside
                # the mathematically valid [-1, 1] interval.
                correlation = max(
                    -1.0,
                    min(1.0, correlation),
                )

                row.append(correlation)

            result.append(row)

        return result, constant_columns

    @classmethod
    def _maximum_columns(
        cls,
        context: LocalOperationContext,
    ) -> int:
        policy = context.metadata.get("policy", {})

        configured_value: Any = None

        if isinstance(policy, dict):
            configured_value = policy.get(
                "maximum_correlation_columns"
            )

            limits = policy.get("limits")

            if (
                configured_value is None
                and isinstance(limits, dict)
            ):
                configured_value = limits.get(
                    "maximum_correlation_columns"
                )

        if configured_value is None:
            return cls.DEFAULT_MAXIMUM_COLUMNS

        if (
            not isinstance(configured_value, int)
            or isinstance(configured_value, bool)
            or configured_value < 2
        ):
            raise OperationValidationError(
                "Policy maximum_correlation_columns must "
                "be an integer of at least 2."
            )

        return min(
            configured_value,
            cls.ABSOLUTE_MAXIMUM_COLUMNS,
        )

    @staticmethod
    def _declared_identifier_columns(
        context: LocalOperationContext,
    ) -> set[str]:
        column_metadata = context.metadata.get(
            "column_metadata",
            {},
        )

        if not isinstance(column_metadata, dict):
            return set()

        identifiers: set[str] = set()

        for column_name, metadata in (
            column_metadata.items()
        ):
            if not isinstance(metadata, dict):
                continue

            sensitivity = str(
                metadata.get("sensitivity", "")
            ).strip().lower()

            if sensitivity in {
                "direct_identifier",
                "identifier",
            }:
                identifiers.add(str(column_name))

        return identifiers