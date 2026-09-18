"""Server-side pooling for registered aggregate-only tool outputs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import chi2_contingency

from biobank_agent.tools.dynamic import execute_dynamic_server


def aggregate_tool(
    analysis: dict[str, Any], outputs: list[dict[str, Any]], dynamic_tools: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    tool = analysis["tool"]
    if tool == "federated_histogram":
        return _histogram(outputs)
    if tool == "categorical_contingency":
        return _contingency(outputs, analysis["cohorts"])
    if tool == "federated_kaplan_meier":
        return _survival(outputs)
    if tool == "numeric_sufficient_statistics":
        return _statistics(outputs)
    if tool == "missingness_summary":
        return _missingness(outputs)
    if dynamic_tools and tool in dynamic_tools:
        return execute_dynamic_server(dynamic_tools[tool], analysis, outputs)
    raise ValueError(f"No server implementation for {tool}")


def _histogram(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    available = [output for output in outputs if output.get("status") == "ok"]
    if not available:
        return {"status": "unavailable"}
    bins = available[0]["bins"]
    groups = {}
    for name in available[0]["groups"]:
        counts = np.sum([output["groups"][name]["counts"] for output in available], axis=0).astype(int)
        cumulative = np.cumsum(counts)
        index = int(np.searchsorted(cumulative, cumulative[-1] / 2)) if cumulative[-1] else 0
        groups[name] = {"counts": counts.tolist(), "n": int(counts.sum()), "median_interval": [bins[index], bins[index + 1]]}
    return {"status": "ok", "bins": bins, "groups": groups, "median_is_binned": True}


def _contingency(outputs: list[dict[str, Any]], cohort_names: list[str]) -> dict[str, Any]:
    pooled: dict[str, dict[str, int | None]] = defaultdict(lambda: {name: 0 for name in cohort_names})
    suppressed: set[tuple[str, str]] = set()
    for output in outputs:
        for cell in output.get("cells", []):
            for name in cohort_names:
                value = cell["counts"].get(name)
                if value is None:
                    suppressed.add((cell["category"], name))
                elif (cell["category"], name) not in suppressed:
                    pooled[cell["category"]][name] = int(pooled[cell["category"]][name] or 0) + int(value)
    cells = []
    for category, counts in sorted(pooled.items()):
        cells.append({"category": category, "counts": {name: None if (category, name) in suppressed else counts[name] for name in cohort_names}})
    complete = [cell for cell in cells if all(value is not None for value in cell["counts"].values())]
    if len(complete) >= 2:
        matrix = [[cell["counts"][name] for name in cohort_names] for cell in complete]
        chi2, p_value, dof, _ = chi2_contingency(matrix)
        test = {"method": "Pearson chi-square", "chi2": chi2, "p_value": p_value, "degrees_of_freedom": dof}
    else:
        test = {"status": "insufficient_unsuppressed_cells"}
    return {"status": "ok", "cohorts": cohort_names, "cells": cells, "test": test, "suppression_propagated": True}


def _survival(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    available = [output for output in outputs if output.get("status") == "ok"]
    if not available:
        return {"status": "unavailable_or_suppressed"}
    bins = available[0]["bins"]
    result = {}
    names = sorted({name for output in available for name in output["groups"]})
    for name in names:
        groups = [output["groups"][name] for output in available if name in output["groups"]]
        events = np.sum([group["events"] for group in groups], axis=0).astype(int)
        censored = np.sum([group["censored"] for group in groups], axis=0).astype(int)
        at_risk = int(events.sum() + censored.sum())
        survival = 1.0
        curve = [{"time": bins[0], "survival": survival, "at_risk": at_risk}]
        for index, (event_count, censor_count) in enumerate(zip(events, censored)):
            if at_risk <= 0:
                break
            survival *= 1.0 - int(event_count) / at_risk
            at_risk -= int(event_count + censor_count)
            curve.append({"time": bins[index + 1], "survival": survival, "at_risk": at_risk})
        result[name] = {"n": int(events.sum() + censored.sum()), "curve": curve}
    return {"status": "ok", "groups": result, "bins": bins, "estimator": "binned Kaplan-Meier"}


def _statistics(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {}
    names = sorted({name for output in outputs for name in output.get("fields", {})})
    for field in names:
        cohort_names = sorted({name for output in outputs for name in output.get("fields", {}).get(field, {}).get("groups", {})})
        groups = {}
        for cohort in cohort_names:
            values = [output["fields"][field]["groups"][cohort] for output in outputs if output.get("fields", {}).get(field, {}).get("groups", {}).get(cohort, {}).get("n")]
            n = sum(value["n"] for value in values)
            total = sum(value["sum"] for value in values)
            squares = sum(value["sum_squares"] for value in values)
            groups[cohort] = {"n": n, "mean": total / n if n else None, "standard_deviation": ((squares - total * total / n) / (n - 1)) ** 0.5 if n > 1 else None}
        fields[field] = {"groups": groups}
    return {"status": "ok", "fields": fields}


def _missingness(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {}
    names = sorted({name for output in outputs for name in output.get("fields", {})})
    for field in names:
        values = [output["fields"][field] for output in outputs if "missing" in output.get("fields", {}).get(field, {})]
        missing = sum(value["missing"] for value in values)
        observed = sum(value["observed"] for value in values)
        fields[field] = {"missing": missing, "observed": observed, "missing_fraction": missing / (missing + observed) if missing + observed else None}
    return {"status": "ok", "fields": fields}
