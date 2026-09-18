# local_agent/metadata_validator.py

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd

from local_agent.operations.metadata_loader import SiteMetadataBundle
from shared.schemas import Operation


class ValidationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class MetadataValidationIssue:
    severity: ValidationSeverity
    code: str
    message: str
    column: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "column": self.column,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class MetadataValidationReport:
    site_id: str
    dataset_id: str
    valid: bool
    issues: tuple[MetadataValidationIssue, ...]
    summary: dict[str, Any]

    @property
    def errors(self) -> tuple[MetadataValidationIssue, ...]:
        return tuple(
            issue
            for issue in self.issues
            if issue.severity == ValidationSeverity.ERROR
        )

    @property
    def warnings(self) -> tuple[MetadataValidationIssue, ...]:
        return tuple(
            issue
            for issue in self.issues
            if issue.severity == ValidationSeverity.WARNING
        )

    @property
    def information(self) -> tuple[MetadataValidationIssue, ...]:
        return tuple(
            issue
            for issue in self.issues
            if issue.severity == ValidationSeverity.INFO
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "dataset_id": self.dataset_id,
            "valid": self.valid,
            "summary": dict(self.summary),
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
        }

    def raise_for_errors(self) -> None:
        if not self.errors:
            return

        messages = "; ".join(
            issue.message
            for issue in self.errors
        )

        raise MetadataValidationError(
            f"Metadata validation failed: {messages}",
            report=self,
        )


class MetadataValidationError(Exception):
    def __init__(
        self,
        message: str,
        *,
        report: MetadataValidationReport,
    ) -> None:
        super().__init__(message)
        self.report = report


class MetadataValidator:
    """
    Validate catalogue, mappings and policy against a local DataFrame.

    This validator never returns individual data values.
    """

    LEGACY_OUTCOME_EXPRESSION = re.compile(
        r"""
        ^\s*
        (?P<column>[A-Za-z_][A-Za-z0-9_]*)
        \s*==\s*
        (?P<quote>['"])
        (?P<value>.*)
        (?P=quote)
        \s*$
        """,
        re.VERBOSE,
    )

    @staticmethod
    def _derived_column_names(
        metadata: SiteMetadataBundle,
    ) -> set[str]:
        """
        Return canonical variables calculated from local columns.

        For example:

            event_cancer_death:
                death_from_cancer == 'Died of Disease'
        """
        outcome = metadata.mappings.get("outcome")

        if not isinstance(outcome, Mapping):
            return set()

        return {
            str(name)
            for name in outcome
            if name != "time"
        }


    def _outcome_source_columns(
        self,
        metadata: SiteMetadataBundle,
    ) -> set[str]:
        """
        Return physical columns referenced by outcome definitions.
        """
        outcome = metadata.mappings.get("outcome")

        if not isinstance(outcome, Mapping):
            return set()

        source_columns: set[str] = set()

        time_column = outcome.get("time")

        if isinstance(time_column, str) and time_column.strip():
            source_columns.add(time_column.strip())

        for name, definition in outcome.items():
            if name == "time":
                continue

            if isinstance(definition, str):
                match = self.LEGACY_OUTCOME_EXPRESSION.fullmatch(
                    definition
                )

                if match is not None:
                    source_columns.add(
                        match.group("column")
                    )

            elif isinstance(definition, Mapping):
                source_column = (
                    definition.get("source_column")
                    or definition.get("column")
                )

                if (
                    isinstance(source_column, str)
                    and source_column.strip()
                ):
                    source_columns.add(
                        source_column.strip()
                    )

        return source_columns

    def validate(
        self,
        *,
        dataframe: pd.DataFrame,
        metadata: SiteMetadataBundle,
        availability_tolerance_pct: float = 1.0,
    ) -> MetadataValidationReport:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        if availability_tolerance_pct < 0:
            raise ValueError(
                "availability_tolerance_pct cannot be negative."
            )

        issues: list[MetadataValidationIssue] = []

        self._validate_duplicate_columns(
            dataframe=dataframe,
            issues=issues,
        )

        catalogue_columns = set(metadata.column_metadata)

        dataframe_columns = {
            str(column)
            for column in dataframe.columns
        }

        derived_columns = self._derived_column_names(
            metadata
        )

        resolved_catalogue_columns = {
            canonical_name
            for canonical_name, local_column
            in metadata.column_mapping.items()
            if local_column in dataframe_columns
        }

        known_local_columns = {
            local_column
            for local_column in metadata.column_mapping.values()
        }

        known_local_columns.update(
            self._outcome_source_columns(metadata)
        )

        missing_catalogue_columns = sorted(
            catalogue_columns
            - dataframe_columns
            - resolved_catalogue_columns
            - derived_columns
        )

        unexpected_dataframe_columns = sorted(
            dataframe_columns
            - catalogue_columns
            - known_local_columns
        )

       

        for column in missing_catalogue_columns:
            issues.append(
                MetadataValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="catalogue_column_missing_from_data",
                    message=(
                        f"Catalogue column {column!r} does not "
                        "exist in the dataset."
                    ),
                    column=column,
                )
            )

        for column in unexpected_dataframe_columns:
            issues.append(
                MetadataValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="data_column_missing_from_catalogue",
                    message=(
                        f"Dataset column {column!r} is not "
                        "declared in catalog.json."
                    ),
                    column=column,
                )
            )

        self._validate_catalogue_columns(
            dataframe=dataframe,
            metadata=metadata,
            common_columns=sorted(
                catalogue_columns & dataframe_columns
            ),
            availability_tolerance_pct=(
                availability_tolerance_pct
            ),
            issues=issues,
        )

        self._validate_mappings(
            metadata=metadata,
            dataframe_columns=dataframe_columns,
            issues=issues,
        )

        self._validate_allowed_operations(
            metadata=metadata,
            issues=issues,
        )

        self._validate_patient_count_range(
            row_count=int(len(dataframe)),
            declared_range=metadata.catalog.get(
                "n_patients"
            ),
            issues=issues,
        )

        self._validate_outcome_configuration(
            metadata=metadata,
            dataframe_columns=dataframe_columns,
            issues=issues,
        )

        severity_counts = Counter(
            issue.severity.value
            for issue in issues
        )

        report = MetadataValidationReport(
            site_id=metadata.site_id,
            dataset_id=metadata.dataset_id,
            valid=not any(
                issue.severity == ValidationSeverity.ERROR
                for issue in issues
            ),
            issues=tuple(issues),
            summary={
                "row_count": int(len(dataframe)),
                "dataframe_column_count": int(
                    len(dataframe.columns)
                ),
                "catalogue_column_count": len(
                    catalogue_columns
                ),
                "mapping_count": len(
                    metadata.column_mapping
                ),
                "missing_catalogue_column_count": len(
                    missing_catalogue_columns
                ),
                "unexpected_dataframe_column_count": len(
                    unexpected_dataframe_columns
                ),
                "error_count": severity_counts.get(
                    ValidationSeverity.ERROR.value,
                    0,
                ),
                "warning_count": severity_counts.get(
                    ValidationSeverity.WARNING.value,
                    0,
                ),
                "info_count": severity_counts.get(
                    ValidationSeverity.INFO.value,
                    0,
                ),
            },
        )

        return report

    @staticmethod
    def _validate_duplicate_columns(
        *,
        dataframe: pd.DataFrame,
        issues: list[MetadataValidationIssue],
    ) -> None:
        duplicate_columns = [
            str(column)
            for column in dataframe.columns[
                dataframe.columns.duplicated()
            ]
        ]

        for column in sorted(set(duplicate_columns)):
            issues.append(
                MetadataValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="duplicate_dataframe_column",
                    message=(
                        f"Dataset contains duplicate column "
                        f"name {column!r}."
                    ),
                    column=column,
                )
            )

    def _validate_catalogue_columns(
        self,
        *,
        dataframe: pd.DataFrame,
        metadata: SiteMetadataBundle,
        common_columns: list[str],
        availability_tolerance_pct: float,
        issues: list[MetadataValidationIssue],
    ) -> None:
        row_count = int(len(dataframe))

        for column in common_columns:
            declared = metadata.column_metadata[column]
            series = dataframe[column]

            self._validate_declared_type(
                column=column,
                series=series,
                declared=declared,
                issues=issues,
            )

            self._validate_availability(
                column=column,
                series=series,
                row_count=row_count,
                declared=declared,
                tolerance_pct=availability_tolerance_pct,
                issues=issues,
            )

            self._validate_identifier_permissions(
                column=column,
                declared=declared,
                issues=issues,
            )

    def _validate_declared_type(
        self,
        *,
        column: str,
        series: pd.Series,
        declared: Mapping[str, Any],
        issues: list[MetadataValidationIssue],
    ) -> None:
        declared_type = declared.get("data_type")

        if declared_type is None:
            issues.append(
                MetadataValidationIssue(
                    severity=ValidationSeverity.INFO,
                    code="column_type_not_declared",
                    message=(
                        f"No declared data type is available for "
                        f"column {column!r}."
                    ),
                    column=column,
                )
            )
            return

        normalized_type = self._normalize_declared_type(
            str(declared_type)
        )

        compatible = self._is_type_compatible(
            series=series,
            declared_type=normalized_type,
        )

        if not compatible:
            issues.append(
                MetadataValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="declared_type_mismatch",
                    message=(
                        f"Column {column!r} is declared as "
                        f"{declared_type!r}, but pandas loaded it as "
                        f"{str(series.dtype)!r}."
                    ),
                    column=column,
                    details={
                        "declared_type": str(declared_type),
                        "pandas_dtype": str(series.dtype),
                    },
                )
            )

    @staticmethod
    def _normalize_declared_type(
        declared_type: str,
    ) -> str:
        normalized = (
            declared_type.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        aliases = {
            "continuous": "numeric",
            "number": "numeric",
            "float": "numeric",
            "double": "numeric",
            "decimal": "numeric",
            "int": "integer",
            "int64": "integer",
            "bool": "boolean",
            "binary": "boolean",
            "category": "categorical",
            "factor": "categorical",
            "str": "string",
            "varchar": "string",
            "date": "datetime",
            "timestamp": "datetime",
        }

        return aliases.get(normalized, normalized)

    @staticmethod
    def _is_type_compatible(
        *,
        series: pd.Series,
        declared_type: str,
    ) -> bool:
        if declared_type == "numeric":
            return pd.api.types.is_numeric_dtype(series)

        if declared_type == "integer":
            return pd.api.types.is_integer_dtype(series)

        if declared_type == "boolean":
            return (
                pd.api.types.is_bool_dtype(series)
                or (
                    pd.api.types.is_numeric_dtype(series)
                    and series.dropna().nunique() <= 2
                )
            )

        if declared_type == "categorical":
            return (
                isinstance(
                    series.dtype,
                    pd.CategoricalDtype,
                )
                or pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or pd.api.types.is_bool_dtype(series)
                or (
                    pd.api.types.is_numeric_dtype(series)
                    and series.dropna().nunique() <= 50
                )
            )

        if declared_type in {"string", "text"}:
            return (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            )

        if declared_type == "datetime":
            return pd.api.types.is_datetime64_any_dtype(
                series
            )

        # Unknown custom types cannot be validated automatically.
        return True

    @staticmethod
    def _validate_availability(
        *,
        column: str,
        series: pd.Series,
        row_count: int,
        declared: Mapping[str, Any],
        tolerance_pct: float,
        issues: list[MetadataValidationIssue],
    ) -> None:
        declared_availability = declared.get(
            "available_pct"
        )

        if declared_availability is None:
            return

        actual_availability = (
            float(series.notna().sum()) / row_count * 100.0
            if row_count > 0
            else 0.0
        )

        difference = abs(
            actual_availability
            - float(declared_availability)
        )

        if difference > tolerance_pct:
            issues.append(
                MetadataValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="availability_mismatch",
                    message=(
                        f"Column {column!r} has declared "
                        f"availability {float(declared_availability):.2f}% "
                        f"but actual availability "
                        f"{actual_availability:.2f}%."
                    ),
                    column=column,
                    details={
                        "declared_available_pct": float(
                            declared_availability
                        ),
                        "actual_available_pct": (
                            actual_availability
                        ),
                        "difference_pct_points": difference,
                        "tolerance_pct_points": tolerance_pct,
                    },
                )
            )

    @staticmethod
    def _validate_identifier_permissions(
        *,
        column: str,
        declared: Mapping[str, Any],
        issues: list[MetadataValidationIssue],
    ) -> None:
        sensitivity = str(
            declared.get("sensitivity", "")
        ).strip().lower()

        allowed_uses = declared.get("allowed_uses", [])

        if allowed_uses is None:
            allowed_uses = []

        if not isinstance(allowed_uses, list):
            issues.append(
                MetadataValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="invalid_allowed_uses",
                    message=(
                        f"allowed_uses for column {column!r} "
                        "must be a list."
                    ),
                    column=column,
                )
            )
            return

        if (
            sensitivity == "direct_identifier"
            and allowed_uses
        ):
            issues.append(
                MetadataValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="identifier_has_allowed_uses",
                    message=(
                        f"Direct identifier {column!r} must not "
                        "be approved for analysis."
                    ),
                    column=column,
                    details={
                        "allowed_uses": list(allowed_uses)
                    },
                )
            )

    @staticmethod
    def _validate_mappings(
        *,
        metadata: SiteMetadataBundle,
        dataframe_columns: set[str],
        issues: list[MetadataValidationIssue],
    ) -> None:
        local_to_canonical: defaultdict[
            str,
            list[str],
        ] = defaultdict(list)

        for canonical, local_column in (
            metadata.column_mapping.items()
        ):
            if local_column not in dataframe_columns:
                issues.append(
                    MetadataValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="mapping_target_missing",
                        message=(
                            f"Canonical variable {canonical!r} maps "
                            f"to missing local column "
                            f"{local_column!r}."
                        ),
                        column=local_column,
                        details={
                            "canonical_name": canonical,
                        },
                    )
                )

            local_to_canonical[local_column].append(
                canonical
            )

        for local_column, canonical_names in (
            local_to_canonical.items()
        ):
            if len(canonical_names) > 1:
                issues.append(
                    MetadataValidationIssue(
                        severity=ValidationSeverity.INFO,
                        code="multiple_canonical_names",
                        message=(
                            f"Local column {local_column!r} maps to "
                            "multiple canonical names."
                        ),
                        column=local_column,
                        details={
                            "canonical_names": sorted(
                                canonical_names
                            )
                        },
                    )
                )

    @staticmethod
    def _validate_allowed_operations(
        *,
        metadata: SiteMetadataBundle,
        issues: list[MetadataValidationIssue],
    ) -> None:
        if not metadata.allowed_operations:
            issues.append(
                MetadataValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="allowed_operations_not_declared",
                    message=(
                        "policy.yaml does not declare "
                        "allowed_operations."
                    ),
                )
            )
            return

        known_operations = {
            operation.value
            for operation in Operation
        }

        for operation in metadata.allowed_operations:
            if operation not in known_operations:
                issues.append(
                    MetadataValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="unknown_allowed_operation",
                        message=(
                            f"Policy declares unknown operation "
                            f"{operation!r}."
                        ),
                        details={"operation": operation},
                    )
                )

    @staticmethod
    def _validate_patient_count_range(
        *,
        row_count: int,
        declared_range: Any,
        issues: list[MetadataValidationIssue],
    ) -> None:
        if declared_range is None:
            return

        if isinstance(declared_range, int):
            if declared_range != row_count:
                issues.append(
                    MetadataValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="patient_count_mismatch",
                        message=(
                            "Declared patient count differs from "
                            "the number of dataset rows."
                        ),
                        details={
                            "declared_count": declared_range,
                            "row_count": row_count,
                        },
                    )
                )
            return

        if not isinstance(declared_range, str):
            issues.append(
                MetadataValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="invalid_patient_count_range",
                    message=(
                        "Catalogue n_patients must be an integer "
                        "or a range such as '500-999'."
                    ),
                )
            )
            return

        match = re.fullmatch(
            r"\s*(\d+)\s*-\s*(\d+)\s*",
            declared_range,
        )

        if match is None:
            issues.append(
                MetadataValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="invalid_patient_count_range",
                    message=(
                        f"Could not parse n_patients value "
                        f"{declared_range!r}."
                    ),
                )
            )
            return

        lower = int(match.group(1))
        upper = int(match.group(2))

        if lower > upper:
            issues.append(
                MetadataValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="reversed_patient_count_range",
                    message=(
                        f"Patient-count range {declared_range!r} "
                        "has a lower bound greater than its "
                        "upper bound."
                    ),
                )
            )
            return

        if not lower <= row_count <= upper:
            issues.append(
                MetadataValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="patient_count_outside_range",
                    message=(
                        f"Dataset row count is outside declared "
                        f"patient-count range {declared_range!r}."
                    ),
                    details={
                        "declared_lower": lower,
                        "declared_upper": upper,
                        "row_count": row_count,
                    },
                )
            )

    def _validate_outcome_configuration(
        self,
        *,
        metadata: SiteMetadataBundle,
        dataframe_columns: set[str],
        issues: list[MetadataValidationIssue],
    ) -> None:
        outcome = metadata.mappings.get("outcome")

        if outcome is None:
            return

        if not isinstance(outcome, Mapping):
            issues.append(
                MetadataValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="invalid_outcome_mapping",
                    message=(
                        "mappings.yaml outcome must be an object."
                    ),
                )
            )
            return

        time_column = outcome.get("time")

        if time_column is not None:
            if not isinstance(time_column, str):
                issues.append(
                    MetadataValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="invalid_outcome_time",
                        message=(
                            "Outcome time must be a column-name string."
                        ),
                    )
                )

            elif time_column not in dataframe_columns:
                issues.append(
                    MetadataValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="outcome_time_column_missing",
                        message=(
                            f"Outcome time column {time_column!r} "
                            "does not exist in the dataset."
                        ),
                        column=time_column,
                    )
                )

        for outcome_name, definition in outcome.items():
            if outcome_name == "time":
                continue

            self._validate_outcome_definition(
                outcome_name=str(outcome_name),
                definition=definition,
                dataframe_columns=dataframe_columns,
                issues=issues,
            )

    def _validate_outcome_definition(
        self,
        *,
        outcome_name: str,
        definition: Any,
        dataframe_columns: set[str],
        issues: list[MetadataValidationIssue],
    ) -> None:
        if isinstance(definition, str):
            match = self.LEGACY_OUTCOME_EXPRESSION.fullmatch(
                definition
            )

            if match is None:
                issues.append(
                    MetadataValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="unparsed_outcome_expression",
                        message=(
                            f"Outcome {outcome_name!r} uses an "
                            "expression that was not interpreted. "
                            "It will not be executed automatically."
                        ),
                    )
                )
                return

            source_column = match.group("column")

            if source_column not in dataframe_columns:
                issues.append(
                    MetadataValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="outcome_source_column_missing",
                        message=(
                            f"Outcome {outcome_name!r} uses missing "
                            f"source column {source_column!r}."
                        ),
                        column=source_column,
                    )
                )

            return

        if isinstance(definition, Mapping):
            source_column = (
                definition.get("source_column")
                or definition.get("column")
            )

            if not isinstance(source_column, str):
                issues.append(
                    MetadataValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="invalid_outcome_source_column",
                        message=(
                            f"Outcome {outcome_name!r} requires "
                            "a source_column."
                        ),
                    )
                )
                return

            if source_column not in dataframe_columns:
                issues.append(
                    MetadataValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="outcome_source_column_missing",
                        message=(
                            f"Outcome {outcome_name!r} uses missing "
                            f"source column {source_column!r}."
                        ),
                        column=source_column,
                    )
                )

            allowed_operators = {
                "equals",
                "not_equals",
                "is_null",
                "is_not_null",
                "in",
            }

            operator = definition.get("operator")

            if operator not in allowed_operators:
                issues.append(
                   MetadataValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="unsupported_outcome_operator",
                        message=(
                            f"Outcome {outcome_name!r} uses "
                            f"unsupported operator {operator!r}."
                        ),
                    )
                )

            return

        issues.append(
            MetadataValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="invalid_outcome_definition",
                message=(
                    f"Outcome {outcome_name!r} must be a string "
                    "or an object."
                ),
            )
        )