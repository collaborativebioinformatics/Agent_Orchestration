

#This defines the common contract for every local operation. It supports both the sequential EDA workflow and later federated execution.

# local_agent/operations/base.py

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd

from shared.schemas import AnalysisPlan, Operation



class LocalOperationError(Exception):
    """Base exception raised by local data operations."""


class OperationValidationError(LocalOperationError):
    """Raised when an operation request is invalid."""


class MissingColumnError(OperationValidationError):
    """Raised when requested columns do not exist locally."""


class OperationExecutionError(LocalOperationError):
    """Raised when an operation fails during calculation."""



MappingType = dict[str, str]


@dataclass(frozen=True, slots=True)
class LocalOperationContext:
    """
    Information available while executing a local operation.

    This context contains site and dataset metadata only. It must not contain
    patient records.
    """

    site_id: str
    dataset_id: str
    minimum_cell_size: int = 10
    column_mapping: MappingType = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_column(self, column: str) -> str:
        """
        Resolve a canonical column name to the site's local column name.

        If no mapping is present, the supplied name is assumed to already be
        a local column name.
        """
        return self.column_mapping.get(column, column)


@dataclass(slots=True)
class LocalOperationResult:
    """
    Internal result returned by a local operation.

    This is not yet a SiteResponse. Privacy enforcement is applied after the
    operation and before the result leaves the site.
    """

    operation: Operation
    result: dict[str, Any]
    record_count: int | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "result": self.result,
            "record_count": self.record_count,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
            "elapsed_seconds": self.elapsed_seconds,
        }


class DataOperation(ABC):
    """
    Abstract base class for all local dataset operations.

    Subclasses implement `_run()`. The public `execute()` method provides a
    common sequence:

    1. Check the operation type.
    2. Validate the DataFrame.
    3. Validate requested columns.
    4. Perform operation-specific validation.
    5. Run the calculation.
    6. Ensure the result is JSON serializable.
    7. Return execution metadata.

    Subclasses should not override `execute()` unless absolutely necessary.
    """

    operation: ClassVar[Operation]
    requires_nonempty_dataframe: ClassVar[bool] = True

    def execute(
        self,
        *,
        dataframe: pd.DataFrame,
        plan: AnalysisPlan,
        context: LocalOperationContext,
    ) -> LocalOperationResult:
        started_at = time.monotonic()

        self._validate_operation(plan)
        self._validate_dataframe(dataframe)
        self._validate_context(context)

        resolved_columns = self.resolve_columns(
            columns=list(getattr(plan, "columns", []) or []),
            context=context,
        )

        resolved_group_by = self.resolve_columns(
            columns=list(getattr(plan, "group_by", []) or []),
            context=context,
        )

        self.validate_columns_exist(
            dataframe=dataframe,
            columns=[
                *resolved_columns,
                *resolved_group_by,
            ],
        )

        self.validate(
            dataframe=dataframe,
            plan=plan,
            context=context,
            resolved_columns=resolved_columns,
            resolved_group_by=resolved_group_by,
        )

        try:
            result = self._run(
                dataframe=dataframe,
                plan=plan,
                context=context,
                resolved_columns=resolved_columns,
                resolved_group_by=resolved_group_by,
            )
        except LocalOperationError:
            raise
        except Exception as exc:
            raise OperationExecutionError(
                f"{self.operation.value} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if not isinstance(result, LocalOperationResult):
            raise OperationExecutionError(
                f"{self.__class__.__name__}._run() must return "
                "LocalOperationResult."
            )

        if result.operation != self.operation:
            raise OperationExecutionError(
                "The returned operation does not match the operation "
                f"handler: expected {self.operation.value!r}, received "
                f"{result.operation.value!r}."
            )

        self.validate_json_serializable(result.result)
        self.validate_json_serializable(result.metadata)

        result.elapsed_seconds = time.monotonic() - started_at

        return result

    def validate(
        self,
        *,
        dataframe: pd.DataFrame,
        plan: AnalysisPlan,
        context: LocalOperationContext,
        resolved_columns: list[str],
        resolved_group_by: list[str],
    ) -> None:
        """
        Optional operation-specific validation hook.

        Subclasses may override this without replacing execute().
        """
        del dataframe
        del plan
        del context
        del resolved_columns
        del resolved_group_by

    @abstractmethod
    def _run(
        self,
        *,
        dataframe: pd.DataFrame,
        plan: AnalysisPlan,
        context: LocalOperationContext,
        resolved_columns: list[str],
        resolved_group_by: list[str],
    ) -> LocalOperationResult:
        """Perform the operation-specific calculation."""
        raise NotImplementedError

    def _validate_operation(
        self,
        plan: AnalysisPlan,
    ) -> None:
        if plan.operation != self.operation:
            raise OperationValidationError(
                f"{self.__class__.__name__} handles "
                f"{self.operation.value!r}, but received "
                f"{plan.operation.value!r}."
            )

    def _validate_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise OperationValidationError(
                "dataframe must be a pandas DataFrame."
            )

        if (
            self.requires_nonempty_dataframe
            and dataframe.empty
        ):
            raise OperationValidationError(
                f"{self.operation.value} cannot run on an empty dataset."
            )

        duplicate_columns = dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()

        if duplicate_columns:
            raise OperationValidationError(
                "The dataset contains duplicate column names: "
                f"{sorted(set(map(str, duplicate_columns)))}."
            )

    @staticmethod
    def _validate_context(
        context: LocalOperationContext,
    ) -> None:
        if not context.site_id.strip():
            raise OperationValidationError(
                "context.site_id cannot be empty."
            )

        if not context.dataset_id.strip():
            raise OperationValidationError(
                "context.dataset_id cannot be empty."
            )

        if context.minimum_cell_size < 1:
            raise OperationValidationError(
                "minimum_cell_size must be at least 1."
            )

    @staticmethod
    def resolve_columns(
        *,
        columns: list[str],
        context: LocalOperationContext,
    ) -> list[str]:
        """
        Translate canonical names from the global plan into local names.
        """
        resolved: list[str] = []
        seen: set[str] = set()

        for column in columns:
            if not isinstance(column, str) or not column.strip():
                raise OperationValidationError(
                    "Column names must be non-empty strings."
                )

            local_column = context.resolve_column(column.strip())

            if local_column not in seen:
                resolved.append(local_column)
                seen.add(local_column)

        return resolved

    @staticmethod
    def validate_columns_exist(
        *,
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> None:
        available = set(map(str, dataframe.columns))

        missing = sorted(
            column
            for column in columns
            if column not in available
        )

        if missing:
            raise MissingColumnError(
                f"Requested local columns do not exist: {missing}."
            )

    @staticmethod
    def validate_numeric_columns(
        *,
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> None:
        non_numeric = [
            column
            for column in columns
            if not pd.api.types.is_numeric_dtype(
                dataframe[column]
            )
        ]

        if non_numeric:
            raise OperationValidationError(
                "The following columns must be numeric: "
                f"{sorted(non_numeric)}."
            )

    @staticmethod
    def validate_categorical_columns(
        *,
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> None:
        invalid: list[str] = []

        for column in columns:
            series = dataframe[column]

            is_categorical = (
                isinstance(series.dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or pd.api.types.is_bool_dtype(series)
            )

            if not is_categorical:
                invalid.append(column)

        if invalid:
            raise OperationValidationError(
                "The following columns must be categorical: "
                f"{sorted(invalid)}."
            )

    @staticmethod
    def require_columns(
        *,
        columns: list[str],
        minimum: int = 1,
        maximum: int | None = None,
    ) -> None:
        if len(columns) < minimum:
            raise OperationValidationError(
                f"At least {minimum} column(s) must be requested."
            )

        if maximum is not None and len(columns) > maximum:
            raise OperationValidationError(
                f"At most {maximum} column(s) may be requested."
            )

    @staticmethod
    def validate_json_serializable(
        value: Any,
    ) -> None:
        """
        Ensure operation output can cross a JSON/FLARE boundary.

        NaN and infinity are rejected because they are not valid strict JSON.
        """
        try:
            json.dumps(
                value,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise OperationExecutionError(
                "Operation output is not valid JSON-compatible data: "
                f"{exc}"
            ) from exc

    @staticmethod
    def safe_record_count(
        dataframe: pd.DataFrame,
    ) -> int:
        return int(len(dataframe))


