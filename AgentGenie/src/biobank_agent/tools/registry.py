"""Server-published tool capabilities and contract validation.

These manifests describe statistical primitives only. Disease semantics,
cohorts, variables, encodings, and endpoints belong in an agent-proposed
contract and are never embedded in a tool.
"""

from __future__ import annotations

from typing import Any


TOOL_MANIFESTS: dict[str, dict[str, Any]] = {
    "federated_histogram": {
        "version": 1,
        "description": "Pool fixed-bin numeric histograms by named cohort and estimate median intervals.",
        "local_output": "bin counts and non-missing count",
        "server_operation": "element-wise sum and binned quantile localization",
        "required_parameters": ["field", "cohorts", "bins"],
    },
    "categorical_contingency": {
        "version": 1,
        "description": "Pool category-by-cohort counts and run a Pearson chi-square association test.",
        "local_output": "disclosure-controlled category-by-cohort counts",
        "server_operation": "cell-wise sum and chi-square test",
        "required_parameters": ["field", "cohorts"],
    },
    "federated_kaplan_meier": {
        "version": 1,
        "description": "Estimate survival curves from pooled fixed-bin event and censor histograms.",
        "provenance": "Generalized from NVIDIA FLARE's time-binned Kaplan-Meier example.",
        "local_output": "event and censor counts per time bin and group",
        "server_operation": "sum histograms and calculate a binned Kaplan-Meier product limit",
        "required_parameters": ["time_field", "event", "group_by", "time_bins"],
        "optional_parameters": ["subset_cohort"],
    },
    "numeric_sufficient_statistics": {
        "version": 1,
        "description": "Pool count, sum, and sum-of-squares for arbitrary numeric fields by cohort.",
        "local_output": "n, sum, and sum-of-squares",
        "server_operation": "pooled mean and sample standard deviation",
        "required_parameters": ["fields", "cohorts"],
    },
    "missingness_summary": {
        "version": 1,
        "description": "Pool missing and observed counts for configured fields.",
        "local_output": "missing and observed counts",
        "server_operation": "count summation and missing percentage",
        "required_parameters": ["fields"],
    },
}


def tool_names() -> set[str]:
    return set(TOOL_MANIFESTS)


def validate_analysis(analysis: dict[str, Any]) -> None:
    name = analysis.get("tool")
    if name not in TOOL_MANIFESTS:
        raise ValueError(f"Tool is not registered: {name}")
    missing = [key for key in TOOL_MANIFESTS[name]["required_parameters"] if key not in analysis]
    if missing:
        raise ValueError(f"Analysis {name} is missing parameters: {missing}")
    if name == "federated_histogram":
        _validate_bins(analysis["bins"], "bins")
    if name == "federated_kaplan_meier":
        _validate_bins(analysis["time_bins"], "time_bins")
        event = analysis["event"]
        if not isinstance(event, dict) or not isinstance(event.get("field"), str) or "values" not in event:
            raise ValueError("Kaplan-Meier event must contain field and values")
    for key in ("cohorts", "fields"):
        if key in analysis and (not isinstance(analysis[key], list) or not analysis[key]):
            raise ValueError(f"{name}.{key} must be a non-empty list")


def _validate_bins(value: Any, name: str) -> None:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{name} must contain at least two ordered boundaries")
    numeric = [float(item) for item in value]
    if any(right <= left for left, right in zip(numeric, numeric[1:])):
        raise ValueError(f"{name} boundaries must be strictly increasing")
