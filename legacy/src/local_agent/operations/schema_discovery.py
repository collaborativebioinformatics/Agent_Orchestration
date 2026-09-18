#detailed column-level 
#It implements Operation.PROFILE_COLUMNS, while dataset_overview.py implements the broader DISCOVER_SCHEMA.


# local_agent/operations/schema_discovery.py

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import pandas as pd

from local_agent.operations.base import (
    DataOperation,
    LocalOperationContext,
    LocalOperationResult,
    OperationValidationError,
)
from shared.schemas import AnalysisPlan, Operation


class SchemaDiscoveryOperation(DataOperation):
    """
    Build detailed, privacy-safe metadata for dataset columns.

    The operation reports structural information only. It never returns:

    - individual patient rows;
    - minimum or maximum patient values;
    - category labels;
    - example values;
    - free-text contents.

    Metadata from catalog.json or mappings.yaml can be passed through
    LocalOperationContext.metadata["column_metadata"].
    """

    operation = Operation.PROFILE_COLUMNS
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
                f"the loaded dataset {context.dataset_id!r}."
            )

        if resolved_group_by:
            raise OperationValidationError(
                "Schema discovery does not support group_by."
            )

        if getattr(plan, "filters", None):
            raise OperationValidationError(
                "Schema discovery does not support cohort filters."
            )

        if getattr(plan, "model", None) is not None:
            raise OperationValidationError(
                "Schema discovery does not support model requests."
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

        selected_columns = (
            resolved_columns
            if resolved_columns
            else [str(column) for column in dataframe.columns]
        )

        declared_metadata = self._get_declared_column_metadata(
            context
        )

        reverse_mapping = self._reverse_column_mapping(
            context.column_mapping
        )

        columns: list[dict[str, Any]] = []
        warnings: list[str] = []

        for local_column in selected_columns:
            series = dataframe[local_column]

            column_profile = self._profile_column(
                column_name=local_column,
                series=series,
                row_count=int(len(dataframe)),
                declared_metadata=declared_metadata.get(
                    local_column,
                    {},
                ),
                canonical_names=reverse_mapping.get(
                    local_column,
                    [],
                ),
                warnings=warnings,
            )

            columns.append(column_profile)

        type_counts = Counter(
            column["inferred_data_type"]
            for column in columns
        )

        role_counts = Counter(
            column["suggested_role"]
            for column in columns
        )

        result = {
            "dataset_id": context.dataset_id,
            "site_id": context.site_id,
            "row_count": int(len(dataframe)),
            "profiled_column_count": len(columns),
            "total_dataset_column_count": int(
                len(dataframe.columns)
            ),
            "columns": columns,
            "summary": {
                "data_type_counts": dict(
                    sorted(type_counts.items())
                ),
                "suggested_role_counts": dict(
                    sorted(role_counts.items())
                ),
                "columns_with_missing_values": sum(
                    column["missing_count"] > 0
                    for column in columns
                ),
                "all_missing_columns": sum(
                    column["quality_flags"]["all_missing"]
                    for column in columns
                ),
                "constant_columns": sum(
                    column["quality_flags"]["constant"]
                    for column in columns
                ),
                "possible_identifier_columns": sum(
                    column["quality_flags"][
                        "possible_identifier"
                    ]
                    for column in columns
                ),
                "high_cardinality_columns": sum(
                    column["quality_flags"][
                        "high_cardinality"
                    ]
                    for column in columns
                ),
            },
        }

        return LocalOperationResult(
            operation=self.operation,
            result=result,
            record_count=int(len(dataframe)),
            warnings=warnings,
            metadata={
                "scope": (
                    "selected_columns"
                    if resolved_columns
                    else "complete_dataset"
                ),
                "used_declared_metadata": bool(
                    declared_metadata
                ),
                "used_column_mapping": bool(
                    context.column_mapping
                ),
            },
        )

    def _profile_column(
        self,
        *,
        column_name: str,
        series: pd.Series,
        row_count: int,
        declared_metadata: Mapping[str, Any],
        canonical_names: list[str],
        warnings: list[str],
    ) -> dict[str, Any]:
        non_missing_count = int(series.notna().sum())
        missing_count = int(row_count - non_missing_count)

        missing_fraction = (
            missing_count / row_count
            if row_count > 0
            else None
        )

        unique_count = self._safe_unique_count(
            series=series,
            column_name=column_name,
            warnings=warnings,
        )

        uniqueness_fraction = (
            unique_count / non_missing_count
            if non_missing_count > 0
            else None
        )

        inferred_data_type = self._infer_data_type(series)

        quality_flags = self._quality_flags(
            column_name=column_name,
            series=series,
            row_count=row_count,
            non_missing_count=non_missing_count,
            unique_count=unique_count,
            inferred_data_type=inferred_data_type,
        )

        suggested_role = self._suggest_role(
            inferred_data_type=inferred_data_type,
            quality_flags=quality_flags,
            unique_count=unique_count,
        )

        suggested_uses = self._suggest_uses(
            suggested_role=suggested_role,
            quality_flags=quality_flags,
        )

        metadata_source = (
            "declared"
            if declared_metadata
            else "inferred"
        )

        return {
            "name": column_name,
            "canonical_names": sorted(canonical_names),
            "display_name": declared_metadata.get(
                "display_name"
            ),
            "description": declared_metadata.get(
                "description"
            ),
            "pandas_dtype": str(series.dtype),
            "inferred_data_type": inferred_data_type,
            "declared_data_type": declared_metadata.get(
                "data_type"
            ),
            "semantic_type": declared_metadata.get(
                "semantic_type"
            ),
            "unit": declared_metadata.get("unit"),
            "sensitivity": declared_metadata.get(
                "sensitivity"
            ),
            "row_count": row_count,
            "non_missing_count": non_missing_count,
            "missing_count": missing_count,
            "missing_fraction": missing_fraction,
            "unique_count": unique_count,
            "uniqueness_fraction": uniqueness_fraction,
            "nullable": missing_count > 0,
            "suggested_role": suggested_role,
            "suggested_uses": suggested_uses,
            "declared_allowed_uses": list(
                declared_metadata.get(
                    "allowed_uses",
                    [],
                )
            ),
            "quality_flags": quality_flags,
            "metadata_source": metadata_source,
        }

    @staticmethod
    def _infer_data_type(
        series: pd.Series,
    ) -> str:
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

    @classmethod
    def _quality_flags(
        cls,
        *,
        column_name: str,
        series: pd.Series,
        row_count: int,
        non_missing_count: int,
        unique_count: int,
        inferred_data_type: str,
    ) -> dict[str, bool]:
        all_missing = non_missing_count == 0

        constant = (
            non_missing_count > 0
            and unique_count == 1
        )

        uniqueness_fraction = (
            unique_count / non_missing_count
            if non_missing_count > 0
            else 0.0
        )

        high_uniqueness = (
            non_missing_count >= 10
            and uniqueness_fraction >= 0.95
        )

        high_cardinality = cls._is_high_cardinality(
            row_count=row_count,
            non_missing_count=non_missing_count,
            unique_count=unique_count,
            inferred_data_type=inferred_data_type,
        )

        possible_identifier = cls._looks_like_identifier(
            column_name=column_name,
            series=series,
            non_missing_count=non_missing_count,
            unique_count=unique_count,
        )

        return {
            "all_missing": all_missing,
            "constant": constant,
            "high_uniqueness": high_uniqueness,
            "high_cardinality": high_cardinality,
            "possible_identifier": possible_identifier,
        }

    @staticmethod
    def _is_high_cardinality(
        *,
        row_count: int,
        non_missing_count: int,
        unique_count: int,
        inferred_data_type: str,
    ) -> bool:
        if row_count == 0 or non_missing_count == 0:
            return False

        if inferred_data_type not in {
            "string",
            "object",
            "categorical",
        }:
            return False

        uniqueness_fraction = unique_count / non_missing_count

        return (
            unique_count >= 50
            or (
                unique_count >= 20
                and uniqueness_fraction >= 0.50
            )
        )

    @staticmethod
    def _looks_like_identifier(
        *,
        column_name: str,
        series: pd.Series,
        non_missing_count: int,
        unique_count: int,
    ) -> bool:
        if non_missing_count < 10:
            return False

        normalized_name = (
            column_name.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        name_suggests_identifier = (
            normalized_name in {
                "id",
                "identifier",
                "patient_id",
                "participant_id",
                "subject_id",
                "person_id",
                "record_id",
                "sample_id",
                "study_id",
                "visit_id",
                "case_id",
            }
            or normalized_name.endswith("_id")
            or normalized_name.endswith("_identifier")
        )

        uniqueness_fraction = (
            unique_count / non_missing_count
        )

        identifier_compatible_type = (
            pd.api.types.is_integer_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_object_dtype(series)
        )

        return (
            name_suggests_identifier
            and identifier_compatible_type
            and uniqueness_fraction >= 0.90
        )

    @staticmethod
    def _suggest_role(
        *,
        inferred_data_type: str,
        quality_flags: Mapping[str, bool],
        unique_count: int,
    ) -> str:
        if quality_flags["possible_identifier"]:
            return "identifier"

        if quality_flags["all_missing"]:
            return "unusable"

        if quality_flags["constant"]:
            return "constant"

        if inferred_data_type in {
            "integer",
            "float",
            "numeric",
        }:
            return "numeric_feature"

        if inferred_data_type == "boolean":
            return "categorical_feature"

        if inferred_data_type == "categorical":
            return "categorical_feature"

        if inferred_data_type == "datetime":
            return "datetime_feature"

        if inferred_data_type in {"string", "object"}:
            if quality_flags["high_cardinality"]:
                return "text_or_identifier_candidate"

            if unique_count <= 50:
                return "categorical_feature"

            return "text_candidate"

        return "unknown"

    @staticmethod
    def _suggest_uses(
        *,
        suggested_role: str,
        quality_flags: Mapping[str, bool],
    ) -> list[str]:
        """
        Suggested uses are advisory only.

        Local policy remains authoritative.
        """
        if quality_flags["possible_identifier"]:
            return []

        if quality_flags["all_missing"]:
            return []

        if quality_flags["constant"]:
            return []

        if suggested_role == "numeric_feature":
            return [
                "analysis",
                "filtering",
                "grouping",
                "modeling",
            ]

        if suggested_role == "categorical_feature":
            return [
                "analysis",
                "filtering",
                "grouping",
                "modeling",
            ]

        if suggested_role == "datetime_feature":
            return [
                "analysis",
                "filtering",
                "grouping",
            ]

        if suggested_role in {
            "text_candidate",
            "text_or_identifier_candidate",
        }:
            return ["analysis"]

        return []

    @staticmethod
    def _safe_unique_count(
        *,
        series: pd.Series,
        column_name: str,
        warnings: list[str],
    ) -> int:
        try:
            return int(series.nunique(dropna=True))

        except (TypeError, ValueError):
            warnings.append(
                f"Could not calculate unique-value count for "
                f"column {column_name!r}."
            )
            return 0

    @staticmethod
    def _reverse_column_mapping(
        column_mapping: Mapping[str, str],
    ) -> dict[str, list[str]]:
        """
        Convert canonical -> local mappings into local -> canonical mappings.

        Example:

        {
            "age": "age_at_diagnosis"
        }

        becomes:

        {
            "age_at_diagnosis": ["age"]
        }
        """
        reverse: dict[str, list[str]] = {}

        for canonical_name, local_name in column_mapping.items():
            reverse.setdefault(
                local_name,
                [],
            ).append(canonical_name)

        return reverse

    @staticmethod
    def _get_declared_column_metadata(
        context: LocalOperationContext,
    ) -> dict[str, dict[str, Any]]:
        """
        Read optional metadata supplied by catalog.json.

        Expected structure:

        context.metadata = {
            "column_metadata": {
                "age_at_diagnosis": {
                    "display_name": "Age at diagnosis",
                    "description": "...",
                    "data_type": "float",
                    "semantic_type": "age",
                    "unit": "years",
                    "sensitivity": "quasi_identifier",
                    "allowed_uses": [
                        "analysis",
                        "filtering",
                        "grouping",
                        "modeling"
                    ]
                }
            }
        }
        """
        raw_metadata = context.metadata.get(
            "column_metadata",
            {},
        )

        if not isinstance(raw_metadata, Mapping):
            raise OperationValidationError(
                "context.metadata['column_metadata'] must "
                "be an object keyed by local column name."
            )

        normalized: dict[str, dict[str, Any]] = {}

        for column_name, metadata in raw_metadata.items():
            if not isinstance(metadata, Mapping):
                raise OperationValidationError(
                    "Metadata for column "
                    f"{column_name!r} must be an object."
                )

            normalized[str(column_name)] = dict(metadata)

        return normalized