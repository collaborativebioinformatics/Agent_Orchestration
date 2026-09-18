"""Safe declarative cohort predicates—no generated Python or SQL."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_cohorts(data: pd.DataFrame, definitions: list[dict[str, Any]]) -> dict[str, pd.Series]:
    cohorts: dict[str, pd.Series] = {}
    for definition in definitions:
        name = str(definition.get("name", "")).strip()
        if not name or name in cohorts:
            raise ValueError("Cohort names must be non-empty and unique")
        cohorts[name] = evaluate_predicate(data, definition.get("predicate"))
    return cohorts


def evaluate_predicate(data: pd.DataFrame, predicate: Any) -> pd.Series:
    if not isinstance(predicate, dict) or len(predicate) != 1:
        raise ValueError("A predicate must contain exactly one operator")
    operator, operand = next(iter(predicate.items()))
    if operator in {"all", "any"}:
        if not isinstance(operand, list) or not operand:
            raise ValueError(f"Predicate {operator} requires a non-empty list")
        values = [evaluate_predicate(data, item) for item in operand]
        result = values[0]
        for value in values[1:]:
            result = result & value if operator == "all" else result | value
        return result.fillna(False)
    if operator == "not":
        return ~evaluate_predicate(data, operand)
    if operator not in {"eq", "eq_ci", "in", "in_ci", "lt", "le", "gt", "ge", "is_missing"}:
        raise ValueError(f"Predicate operator is not allowed: {operator}")
    if not isinstance(operand, dict) or not isinstance(operand.get("field"), str):
        raise ValueError(f"Predicate {operator} requires a field")
    field = operand["field"]
    if field not in data:
        raise ValueError(f"Cohort field is unavailable after harmonization: {field}")
    series = data[field]
    if operator == "is_missing":
        result = series.isna()
        return ~result if operand.get("value") is False else result
    if operator in {"eq_ci", "in_ci"}:
        normalized = series.astype("string").str.strip().str.casefold()
        values = operand.get("values") if operator == "in_ci" else [operand.get("value")]
        return normalized.isin({str(value).strip().casefold() for value in values}).fillna(False)
    if operator == "eq":
        return series.eq(operand.get("value")).fillna(False)
    if operator == "in":
        return series.isin(operand.get("values", [])).fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    value = float(operand["value"])
    return {"lt": numeric.lt, "le": numeric.le, "gt": numeric.gt, "ge": numeric.ge}[operator](value).fillna(False)
