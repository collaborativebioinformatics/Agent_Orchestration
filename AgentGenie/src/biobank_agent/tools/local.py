"""Boundary-local implementations of the registered statistical primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _normalized_scalar(value: Any) -> str | None:
    """Normalize semantically equal numeric/string contract values to one token."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return format(float(value), ".15g")
    text = str(value).strip().casefold()
    try:
        return format(float(text), ".15g")
    except ValueError:
        return text


def execute_local_tool(
    analysis: dict[str, Any],
    data: pd.DataFrame,
    cohorts: dict[str, pd.Series],
    min_cell: int,
) -> dict[str, Any]:
    tool = analysis["tool"]
    if tool == "federated_histogram":
        return histogram(data, cohorts, analysis["field"], analysis["cohorts"], analysis["bins"])
    if tool == "categorical_contingency":
        return contingency(data, cohorts, analysis["field"], analysis["cohorts"], min_cell)
    if tool == "federated_kaplan_meier":
        return survival_histograms(data, cohorts, analysis, min_cell)
    if tool == "numeric_sufficient_statistics":
        return sufficient_statistics(data, cohorts, analysis["fields"], analysis["cohorts"], min_cell)
    if tool == "missingness_summary":
        return missingness(data, analysis["fields"])
    raise ValueError(f"No local implementation for {tool}")


def histogram(
    data: pd.DataFrame,
    cohorts: dict[str, pd.Series],
    field: str,
    cohort_names: list[str],
    bins: list[float],
) -> dict[str, Any]:
    _require_fields(data, [field])
    values = pd.to_numeric(data[field], errors="coerce")
    groups = {}
    for name in cohort_names:
        selected = _cohort(cohorts, name)
        counts, _ = np.histogram(values[selected].dropna(), bins=bins)
        groups[name] = {"counts": counts.astype(int).tolist(), "n": int(values[selected].notna().sum())}
    return {"status": "ok", "bins": bins, "groups": groups}


def contingency(
    data: pd.DataFrame,
    cohorts: dict[str, pd.Series],
    field: str,
    cohort_names: list[str],
    min_cell: int,
) -> dict[str, Any]:
    _require_fields(data, [field])
    categories = sorted(str(item) for item in data[field].dropna().unique())
    cells = []
    for category in categories:
        cell: dict[str, Any] = {"category": category, "counts": {}}
        for name in cohort_names:
            count = int((_cohort(cohorts, name) & data[field].astype("string").eq(category)).sum())
            cell["counts"][name] = count if count == 0 or count >= min_cell else None
        cells.append(cell)
    return {"status": "ok", "cohorts": cohort_names, "cells": cells, "min_cell_count": min_cell}


def survival_histograms(
    data: pd.DataFrame,
    cohorts: dict[str, pd.Series],
    analysis: dict[str, Any],
    min_cell: int,
) -> dict[str, Any]:
    time_field = analysis["time_field"]
    event_field = analysis["event"]["field"]
    group_by = analysis["group_by"]
    _require_fields(data, [time_field, event_field, group_by])
    selected = pd.Series(True, index=data.index)
    if analysis.get("subset_cohort"):
        selected &= _cohort(cohorts, analysis["subset_cohort"])
    frame = data[selected].copy()
    frame["_time"] = pd.to_numeric(frame[time_field], errors="coerce")
    frame = frame[frame["_time"].notna()]
    event_values = {_normalized_scalar(value) for value in analysis["event"]["values"]}
    frame["_event"] = frame[event_field].map(_normalized_scalar).isin(event_values)
    bins = analysis["time_bins"]
    groups = {}
    for value, group in frame.groupby(group_by, dropna=False):
        if len(group) < min_cell:
            continue
        times = group["_time"].clip(lower=bins[0], upper=bins[-1] - 1e-9)
        events, _ = np.histogram(times[group["_event"]], bins=bins)
        censored, _ = np.histogram(times[~group["_event"]], bins=bins)
        groups[str(value)] = {
            "n": int(len(group)),
            "events": events.astype(int).tolist(),
            "censored": censored.astype(int).tolist(),
        }
    return {"status": "ok" if groups else "suppressed", "bins": bins, "groups": groups}


def sufficient_statistics(
    data: pd.DataFrame,
    cohorts: dict[str, pd.Series],
    fields: list[str],
    cohort_names: list[str],
    min_cell: int,
) -> dict[str, Any]:
    result = {}
    for field in fields:
        if field not in data:
            result[field] = {"status": "unavailable"}
            continue
        numeric = pd.to_numeric(data[field], errors="coerce")
        groups = {}
        for name in cohort_names:
            values = numeric[_cohort(cohorts, name)].dropna()
            groups[name] = (
                {"status": "suppressed"}
                if len(values) < min_cell
                else {"n": int(len(values)), "sum": float(values.sum()), "sum_squares": float(np.square(values).sum())}
            )
        result[field] = {"status": "ok", "groups": groups}
    return {"status": "ok", "fields": result}


def missingness(data: pd.DataFrame, fields: list[str]) -> dict[str, Any]:
    return {
        "status": "ok",
        "fields": {
            field: (
                {"status": "unavailable"}
                if field not in data
                else {"missing": int(data[field].isna().sum()), "observed": int(data[field].notna().sum())}
            )
            for field in fields
        },
    }


def _cohort(cohorts: dict[str, pd.Series], name: str) -> pd.Series:
    if name not in cohorts:
        raise ValueError(f"Unknown cohort: {name}")
    return cohorts[name]


def _require_fields(data: pd.DataFrame, fields: list[str]) -> None:
    missing = sorted(set(fields) - set(data.columns))
    if missing:
        raise ValueError(f"Fields unavailable after harmonization: {missing}")
