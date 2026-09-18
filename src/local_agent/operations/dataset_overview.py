# local_agent/operations/dataset_overview.py

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from local_agent.operations.base import (
    DataOperation,
    LocalOperationContext,
    LocalOperationResult,
    OperationValidationError,
)
from shared.schemas import AnalysisPlan, Operation


class DatasetOverviewOperation(DataOperation):
    """
    Produce a high-level structural overview of one local dataset.

    This operation reports:

    - number of rows and columns;
    - column names;
    - broad data-type counts;
    - total missingness;
    - duplicate-row count;
    - constant and all-missing columns;
    - high-uniqueness columns;
    - possible identifier columns;
    - approximate DataFrame memory usage.

    It never returns individual values or dataset rows.
    """

    operation = Operation.DISCOVER_SCHEMA
    requires_nonempty_dataframe = False

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
        del resolved_columns

        if plan.dataset_id != context.dataset_id:
            raise OperationValidationError(
                f"Plan dataset {plan.dataset_id!r} does not match "
                f"the locally loaded dataset {context.dataset_id!r}."
            )

        if resolved_group_by:
            raise OperationValidationError(
                "Dataset overview does not support group_by."
            )

        if getattr(plan, "filters", None):
            raise OperationValidationError(
                "Dataset overview does not support cohort filters."
            )

        if getattr(plan, "model", None) is not None:
            raise OperationValidationError(
                "Dataset overview does not support model requests."
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
        del plan
        del resolved_group_by

        # If the plan contains columns, create an overview of only those
        # columns. Otherwise inspect the complete dataset.
        selected_dataframe = (
            dataframe.loc[:, resolved_columns]
            if resolved_columns
            else dataframe
        )

        row_count = int(len(selected_dataframe))
        column_count = int(len(selected_dataframe.columns))
        total_cell_count = row_count * column_count

        warnings: list[str] = []

        missing_count_by_column = (
            selected_dataframe.isna()
            .sum()
            .astype(int)
            .to_dict()
        )

        total_missing_cells = int(
            sum(missing_count_by_column.values())
        )

        missing_fraction = (
            total_missing_cells / total_cell_count
            if total_cell_count > 0
            else None
        )

        duplicate_row_count = self._count_duplicate_rows(
            selected_dataframe,
            warnings,
        )

        type_by_column = {
            str(column): self._classify_dtype(
                selected_dataframe[column]
            )
            for column in selected_dataframe.columns
        }

        type_counts = dict(
            sorted(Counter(type_by_column.values()).items())
        )

        all_missing_columns: list[str] = []
        constant_columns: list[str] = []
        high_uniqueness_columns: list[str] = []
        possible_identifier_columns: list[str] = []

        for column in selected_dataframe.columns:
            column_name = str(column)
            series = selected_dataframe[column]

            non_missing_count = int(series.notna().sum())

            unique_count = self._safe_unique_count(
                series=series,
                column_name=column_name,
                warnings=warnings,
            )

            if non_missing_count == 0:
                all_missing_columns.append(column_name)
                continue

            if unique_count == 1:
                constant_columns.append(column_name)

            uniqueness_fraction = (
                unique_count / non_missing_count
                if non_missing_count > 0
                else 0.0
            )

            if (
                non_missing_count >= 10
                and uniqueness_fraction >= 0.95
            ):
                high_uniqueness_columns.append(column_name)

            if self._looks_like_identifier(
                column_name=column_name,
                series=series,
                non_missing_count=non_missing_count,
                unique_count=unique_count,
            ):
                possible_identifier_columns.append(column_name)

        memory_bytes = int(
            selected_dataframe.memory_usage(
                index=True,
                deep=True,
            ).sum()
        )

        result: dict[str, Any] = {
            "dataset_id": context.dataset_id,
            "site_id": context.site_id,
            "shape": {
                "row_count": row_count,
                "column_count": column_count,
            },
            "columns": [
                str(column)
                for column in selected_dataframe.columns
            ],
            "data_types": {
                "by_column": type_by_column,
                "counts": type_counts,
            },
            "missingness": {
                "total_missing_cells": total_missing_cells,
                "total_cell_count": total_cell_count,
                "missing_fraction": missing_fraction,
                "columns_with_missing_values": int(
                    sum(
                        count > 0
                        for count
                        in missing_count_by_column.values()
                    )
                ),
            },
            "duplicate_rows": {
                "count": duplicate_row_count,
                "fraction": (
                    duplicate_row_count / row_count
                    if (
                        duplicate_row_count is not None
                        and row_count > 0
                    )
                    else None
                ),
            },
            "quality_flags": {
                "all_missing_columns": sorted(
                    all_missing_columns
                ),
                "constant_columns": sorted(
                    constant_columns
                ),
                "high_uniqueness_columns": sorted(
                    high_uniqueness_columns
                ),
                "possible_identifier_columns": sorted(
                    possible_identifier_columns
                ),
            },
            "memory": {
                "estimated_bytes": memory_bytes,
                "estimated_megabytes": (
                    memory_bytes / (1024 * 1024)
                ),
            },
        }

        return LocalOperationResult(
            operation=self.operation,
            result=result,
            record_count=row_count,
            warnings=warnings,
            metadata={
                "selected_column_count": column_count,
                "full_dataset_column_count": int(
                    len(dataframe.columns)
                ),
                "overview_scope": (
                    "selected_columns"
                    if resolved_columns
                    else "complete_dataset"
                ),
            },
        )

    @staticmethod
    def _classify_dtype(
        series: pd.Series,
    ) -> str:
        """
        Convert pandas dtypes into stable, general data categories.
        """
        if pd.api.types.is_bool_dtype(series):
            return "boolean"

        if pd.api.types.is_integer_dtype(series):
            return "integer"

        if pd.api.types.is_float_dtype(series):
            return "float"

        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        if isinstance(series.dtype, pd.CategoricalDtype):
            return "categorical"

        if pd.api.types.is_string_dtype(series):
            return "string"

        if pd.api.types.is_object_dtype(series):
            return "object"

        return "unknown"

    @staticmethod
    def _safe_unique_count(
        *,
        series: pd.Series,
        column_name: str,
        warnings: list[str],
    ) -> int:
        """
        Count unique non-missing values without returning those values.

        Some object columns can contain unhashable values such as lists or
        dictionaries. These cannot always be processed by Series.nunique().
        """
        try:
            return int(series.nunique(dropna=True))

        except (TypeError, ValueError):
            warnings.append(
                f"Could not calculate the unique-value count for "
                f"column {column_name!r}."
            )
            return 0

    @staticmethod
    def _count_duplicate_rows(
        dataframe: pd.DataFrame,
        warnings: list[str],
    ) -> int | None:
        """
        Return the number of duplicate rows.

        Duplicate detection may fail for columns containing unhashable
        Python objects.
        """
        if dataframe.empty:
            return 0

        try:
            return int(dataframe.duplicated().sum())

        except (TypeError, ValueError):
            warnings.append(
                "Duplicate-row detection was skipped because one or more "
                "columns contain values that cannot be compared safely."
            )
            return None

    @staticmethod
    def _looks_like_identifier(
        *,
        column_name: str,
        series: pd.Series,
        non_missing_count: int,
        unique_count: int,
    ) -> bool:
        """
        Conservatively identify columns that may contain record identifiers.

        This is a warning heuristic, not an automatic semantic decision.
        """
        if non_missing_count < 10:
            return False

        uniqueness_fraction = unique_count / non_missing_count

        normalized_name = (
            column_name.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        identifier_names = {
            "id",
            "identifier",
            "patient_id",
            "participant_id",
            "person_id",
            "subject_id",
            "record_id",
            "sample_id",
            "study_id",
            "visit_id",
            "case_id",
        }

        name_suggests_identifier = (
            normalized_name in identifier_names
            or normalized_name.endswith("_id")
            or normalized_name.endswith("_identifier")
        )

        is_identifier_compatible_type = (
            pd.api.types.is_string_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_integer_dtype(series)
        )

        return (
            name_suggests_identifier
            and is_identifier_compatible_type
            and uniqueness_fraction >= 0.90
        )