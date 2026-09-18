# local_agent/operations/cohort.py

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from local_agent.operations.base import (
    DataOperation,
    LocalOperationContext,
    LocalOperationResult,
    OperationValidationError,
)
from shared.schemas import AnalysisPlan, Operation


class CohortCountOperation(DataOperation):
    """
    Count local records satisfying an approved set of filters.

    All filters are combined using logical AND.

    The operation returns only aggregate counts. It never returns matching
    rows, row indices, patient identifiers, or individual values.

    Privacy suppression is intentionally handled after execution by the
    local privacy layer.
    """

    operation = Operation.COHORT_COUNT
    requires_nonempty_dataframe = False

    DERIVED_EQUALITY_EXPRESSION = re.compile(
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

    OPERATOR_ALIASES = {
        "eq": "equals",
        "equal": "equals",
        "equals": "equals",
        "==": "equals",

        "ne": "not_equals",
        "not_equal": "not_equals",
        "not_equals": "not_equals",
        "!=": "not_equals",

        "gt": "greater_than",
        "greater_than": "greater_than",
        ">": "greater_than",

        "ge": "greater_than_or_equal",
        "gte": "greater_than_or_equal",
        "greater_than_or_equal": "greater_than_or_equal",
        ">=": "greater_than_or_equal",

        "lt": "less_than",
        "less_than": "less_than",
        "<": "less_than",

        "le": "less_than_or_equal",
        "lte": "less_than_or_equal",
        "less_than_or_equal": "less_than_or_equal",
        "<=": "less_than_or_equal",

        "in": "in",
        "not_in": "not_in",

        "between": "between",

        "is_null": "is_null",
        "is_missing": "is_null",

        "is_not_null": "is_not_null",
        "is_not_missing": "is_not_null",

        "contains": "contains",
        "starts_with": "starts_with",
        "ends_with": "ends_with",
    }

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
                "Cohort count does not support group_by. "
                "Use grouped summary for group-specific counts."
            )

        if getattr(plan, "model", None) is not None:
            raise OperationValidationError(
                "Cohort count does not support model requests."
            )

        filters = list(getattr(plan, "filters", []) or [])

        for position, filter_condition in enumerate(filters):
            self._extract_filter(
                filter_condition=filter_condition,
                position=position,
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
        del resolved_columns
        del resolved_group_by

        filters = list(getattr(plan, "filters", []) or [])

        mask = pd.Series(
            True,
            index=dataframe.index,
            dtype=bool,
        )

        applied_filters: list[dict[str, Any]] = []

        for position, filter_condition in enumerate(filters):
            column, operator, value = self._extract_filter(
                filter_condition=filter_condition,
                position=position,
            )

            normalized_operator = self._normalize_operator(
                operator=operator,
                position=position,
            )

            series, resolved_column, derived = (
                self._resolve_filter_series(
                    dataframe=dataframe,
                    requested_column=column,
                    context=context,
                )
            )

            condition_mask = self._apply_operator(
                series=series,
                operator=normalized_operator,
                value=value,
                column=column,
            )

            # Missing comparisons may create nullable Boolean masks.
            condition_mask = condition_mask.fillna(False).astype(bool)

            mask &= condition_mask

            applied_filters.append(
                {
                    "requested_column": column,
                    "resolved_column": resolved_column,
                    "operator": normalized_operator,
                    "derived": derived,
                }
            )

        cohort_count = int(mask.sum())
        source_count = int(len(dataframe))

        return LocalOperationResult(
            operation=self.operation,
            result={
                "count": cohort_count,
            },
            record_count=cohort_count,
            warnings=[],
            metadata={
                "source_record_count": source_count,
                "filter_count": len(filters),
                "applied_filters": applied_filters,
            },
        )

    def _resolve_filter_series(
        self,
        *,
        dataframe: pd.DataFrame,
        requested_column: str,
        context: LocalOperationContext,
    ) -> tuple[pd.Series, str, bool]:
        """
        Resolve a requested canonical column to either:

        1. a physical local column; or
        2. an allowlisted derived outcome.
        """
        local_column = context.resolve_column(
            requested_column
        )

        if local_column in dataframe.columns:
            return (
                dataframe[local_column],
                local_column,
                False,
            )

        # If the mapping does not resolve to a physical column,
        # check whether the requested variable is a derived outcome.
        derived_series = self._derive_outcome_series(
            dataframe=dataframe,
            outcome_name=requested_column,
            context=context,
        )

        if derived_series is not None:
            return (
                derived_series,
                requested_column,
                True,
            )

        raise OperationValidationError(
            f"Filter column {requested_column!r} could not be "
            "resolved to a local or derived column."
        )

    def _derive_outcome_series(
        self,
        *,
        dataframe: pd.DataFrame,
        outcome_name: str,
        context: LocalOperationContext,
    ) -> pd.Series | None:
        """
        Safely construct a configured derived outcome.

        Arbitrary Python expressions are never evaluated.
        """
        mapping_metadata = context.metadata.get(
            "mapping_metadata",
            {},
        )

        if not isinstance(mapping_metadata, Mapping):
            return None

        outcome_configuration = mapping_metadata.get(
            "outcome"
        )

        if not isinstance(outcome_configuration, Mapping):
            return None

        definition = outcome_configuration.get(outcome_name)

        if definition is None:
            return None

        if isinstance(definition, str):
            return self._derive_from_legacy_expression(
                dataframe=dataframe,
                outcome_name=outcome_name,
                expression=definition,
            )

        if isinstance(definition, Mapping):
            return self._derive_from_structured_definition(
                dataframe=dataframe,
                outcome_name=outcome_name,
                definition=definition,
            )

        raise OperationValidationError(
            f"Derived outcome {outcome_name!r} has an "
            "invalid configuration."
        )

    def _derive_from_legacy_expression(
        self,
        *,
        dataframe: pd.DataFrame,
        outcome_name: str,
        expression: str,
    ) -> pd.Series:
        """
        Support the existing safe equality format:

            death_from_cancer == 'Died of Disease'
        """
        match = self.DERIVED_EQUALITY_EXPRESSION.fullmatch(
            expression
        )

        if match is None:
            raise OperationValidationError(
                f"Derived outcome {outcome_name!r} uses an "
                "unsupported expression. Only simple quoted equality "
                "expressions are accepted."
            )

        source_column = match.group("column")
        expected_value = match.group("value")

        if source_column not in dataframe.columns:
            raise OperationValidationError(
                f"Derived outcome {outcome_name!r} requires "
                f"missing source column {source_column!r}."
            )

        return dataframe[source_column].eq(expected_value)

    def _derive_from_structured_definition(
        self,
        *,
        dataframe: pd.DataFrame,
        outcome_name: str,
        definition: Mapping[str, Any],
    ) -> pd.Series:
        source_column = (
            definition.get("source_column")
            or definition.get("column")
        )

        if not isinstance(source_column, str):
            raise OperationValidationError(
                f"Derived outcome {outcome_name!r} requires "
                "a source_column."
            )

        if source_column not in dataframe.columns:
            raise OperationValidationError(
                f"Derived outcome {outcome_name!r} requires "
                f"missing source column {source_column!r}."
            )

        operator = self._normalize_operator(
            operator=definition.get("operator"),
            position=None,
        )

        return self._apply_operator(
            series=dataframe[source_column],
            operator=operator,
            value=definition.get("value"),
            column=source_column,
        )

    def _apply_operator(
        self,
        *,
        series: pd.Series,
        operator: str,
        value: Any,
        column: str,
    ) -> pd.Series:
        if operator == "equals":
            return series.eq(value)

        if operator == "not_equals":
            return series.ne(value)

        if operator == "greater_than":
            self._require_value(
                value=value,
                operator=operator,
                column=column,
            )
            return series.gt(value)

        if operator == "greater_than_or_equal":
            self._require_value(
                value=value,
                operator=operator,
                column=column,
            )
            return series.ge(value)

        if operator == "less_than":
            self._require_value(
                value=value,
                operator=operator,
                column=column,
            )
            return series.lt(value)

        if operator == "less_than_or_equal":
            self._require_value(
                value=value,
                operator=operator,
                column=column,
            )
            return series.le(value)

        if operator == "in":
            values = self._require_sequence(
                value=value,
                operator=operator,
                column=column,
            )
            return series.isin(values)

        if operator == "not_in":
            values = self._require_sequence(
                value=value,
                operator=operator,
                column=column,
            )
            return ~series.isin(values)

        if operator == "between":
            values = self._require_sequence(
                value=value,
                operator=operator,
                column=column,
            )

            if len(values) != 2:
                raise OperationValidationError(
                    f"Filter operator 'between' for column "
                    f"{column!r} requires exactly two values."
                )

            lower, upper = values

            return series.between(
                lower,
                upper,
                inclusive="both",
            )

        if operator == "is_null":
            return series.isna()

        if operator == "is_not_null":
            return series.notna()

        if operator == "contains":
            string_value = self._require_string(
                value=value,
                operator=operator,
                column=column,
            )

            return series.astype("string").str.contains(
                string_value,
                regex=False,
                na=False,
            )

        if operator == "starts_with":
            string_value = self._require_string(
                value=value,
                operator=operator,
                column=column,
            )

            return series.astype("string").str.startswith(
                string_value,
                na=False,
            )

        if operator == "ends_with":
            string_value = self._require_string(
                value=value,
                operator=operator,
                column=column,
            )

            return series.astype("string").str.endswith(
                string_value,
                na=False,
            )

        raise OperationValidationError(
            f"Unsupported filter operator {operator!r}."
        )

    def _normalize_operator(
        self,
        *,
        operator: Any,
        position: int | None,
    ) -> str:
        if hasattr(operator, "value"):
            operator = operator.value

        if not isinstance(operator, str):
            location = (
                f" at position {position}"
                if position is not None
                else ""
            )

            raise OperationValidationError(
                f"Filter operator{location} must be a string "
                "or FilterOperator enum."
            )

        normalized = (
            operator.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        resolved = self.OPERATOR_ALIASES.get(normalized)

        if resolved is None:
            raise OperationValidationError(
                f"Unsupported filter operator {operator!r}."
            )

        return resolved

    @staticmethod
    def _extract_filter(
        *,
        filter_condition: Any,
        position: int,
    ) -> tuple[str, Any, Any]:
        """
        Support both Pydantic FilterCondition objects and dictionaries.
        """
        if isinstance(filter_condition, Mapping):
            column = filter_condition.get("column")
            operator = filter_condition.get("operator")
            value = filter_condition.get("value")

        else:
            column = getattr(
                filter_condition,
                "column",
                None,
            )
            operator = getattr(
                filter_condition,
                "operator",
                None,
            )
            value = getattr(
                filter_condition,
                "value",
                None,
            )

        if not isinstance(column, str) or not column.strip():
            raise OperationValidationError(
                f"Filter at position {position} requires a "
                "non-empty column name."
            )

        if operator is None:
            raise OperationValidationError(
                f"Filter at position {position} requires an operator."
            )

        return column.strip(), operator, value

    @staticmethod
    def _require_value(
        *,
        value: Any,
        operator: str,
        column: str,
    ) -> None:
        if value is None:
            raise OperationValidationError(
                f"Filter operator {operator!r} for column "
                f"{column!r} requires a value."
            )

    @staticmethod
    def _require_sequence(
        *,
        value: Any,
        operator: str,
        column: str,
    ) -> list[Any]:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
        ):
            raise OperationValidationError(
                f"Filter operator {operator!r} for column "
                f"{column!r} requires a list of values."
            )

        values = list(value)

        if not values:
            raise OperationValidationError(
                f"Filter operator {operator!r} for column "
                f"{column!r} requires at least one value."
            )

        return values

    @staticmethod
    def _require_string(
        *,
        value: Any,
        operator: str,
        column: str,
    ) -> str:
        if not isinstance(value, str):
            raise OperationValidationError(
                f"Filter operator {operator!r} for column "
                f"{column!r} requires a string value."
            )

        return value