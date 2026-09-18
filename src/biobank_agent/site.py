"""Boundary-local execution of a user-approved, task-specific contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from biobank_agent.contracts import AnalysisContract
from biobank_agent.tools.cohorts import build_cohorts
from biobank_agent.tools.harmonize import harmonize
from biobank_agent.tools.local import execute_local_tool, survival_eligibility_mask
from biobank_agent.tools.dynamic import validate_dynamic_tools


class SiteExecutor:
    """Apply general tools locally; contains no disease or endpoint semantics."""

    def __init__(self, site_dir: Path) -> None:
        self.site_dir = site_dir
        self.site_id = site_dir.name
        self.catalog = json.loads((site_dir / "catalog.json").read_text(encoding="utf-8"))

    def execute(self, contract: AnalysisContract) -> dict[str, Any]:
        contract.require_approval()
        data = pd.read_csv(self.site_dir / "data.csv", low_memory=False)
        data, harmonization = harmonize(data, self.site_id, contract.payload.get("site_harmonization", {}))
        cohorts = build_cohorts(data, contract.payload["cohorts"])
        min_cell = int(contract.payload["privacy"]["min_cell_count"])
        cohort_sizes = {
            name: (int(mask.sum()) if int(mask.sum()) == 0 or int(mask.sum()) >= min_cell else None)
            for name, mask in cohorts.items()
        }
        result: dict[str, Any] = {
            "schema_version": "biobank.site_aggregate.v2",
            "site_id": self.site_id,
            "contract_digest": contract.digest,
            "patient_rows_exported": 0,
            "local_total_rows": int(len(data)) if len(data) == 0 or len(data) >= min_cell else None,
            "harmonization": harmonization,
            "cohort_sizes": cohort_sizes,
            "analysis_row_counts": {},
            "row_exclusion_reason": None,
            "analyses": [],
        }
        dynamic_tools = validate_dynamic_tools(contract.payload.get("dynamic_tools", []))
        for index, analysis in enumerate(contract.payload["analyses"]):
            target_site = analysis.get("site_id")
            if isinstance(target_site, str) and target_site != self.site_id:
                continue
            output = execute_local_tool(analysis, data, cohorts, min_cell, dynamic_tools)
            if analysis["tool"] in dynamic_tools:
                output = {**output, "site_id": self.site_id}
            analysis_id = analysis.get("analysis_id", f"analysis_{index + 1}")
            subset = analysis.get("subset_cohort")
            if analysis["tool"] == "federated_kaplan_meier":
                eligible = survival_eligibility_mask(data, cohorts, analysis)
                rows_used = int(eligible.sum())
                rows_excluded = int(len(data) - rows_used)
                disclosure_safe = (rows_used == 0 or rows_used >= min_cell) and (
                    rows_excluded == 0 or rows_excluded >= min_cell
                )
                rows_used_range = None
                if not disclosure_safe:
                    if 0 < rows_excluded < min_cell:
                        rows_used_range = {
                            "minimum": max(0, len(data) - (min_cell - 1)),
                            "maximum": len(data) - 1,
                        }
                    elif 0 < rows_used < min_cell:
                        rows_used_range = {"minimum": 1, "maximum": min_cell - 1}
                result["analysis_row_counts"][analysis_id] = {
                    "rows_used": rows_used if disclosure_safe else None,
                    "rows_used_range": rows_used_range,
                    "selection": "complete cases for the approved time, event, and group fields",
                }
                if result["row_exclusion_reason"] is None:
                    result["row_exclusion_reason"] = _survival_exclusion_reason(data, eligible, min_cell)
            elif subset in cohort_sizes:
                result["analysis_row_counts"][analysis_id] = {
                    "rows_used": cohort_sizes[subset],
                    "selection": subset,
                }
            result["analyses"].append(
                {
                    "analysis_id": analysis_id,
                    "tool": analysis["tool"],
                    "output": output,
                }
            )
        if result["row_exclusion_reason"] is None:
            result["row_exclusion_reason"] = _row_exclusion_reason(data, contract.payload, cohorts)
        return result


def _survival_exclusion_reason(data: pd.DataFrame, eligible: pd.Series, min_cell: int) -> str:
    excluded = int(len(data) - eligible.sum())
    if excluded == 0:
        return "All local rows had valid nonmissing time, event, and group values for this analysis."
    if excluded < min_cell:
        return (
            "A small disclosure-controlled number of local rows was excluded because an approved time, "
            "event, or group value was missing, nonnumeric, or negative."
        )
    return (
        f"{excluded} local rows were excluded because an approved time, event, or group value was "
        "missing, nonnumeric, or negative."
    )


def _row_exclusion_reason(
    data: pd.DataFrame,
    contract: dict[str, Any],
    cohorts: dict[str, pd.Series],
) -> str:
    """Produce one boundary-local sentence from approved cohort logic, without row data."""
    analysis = next((item for item in contract.get("analyses", []) if item.get("subset_cohort")), None)
    if not analysis:
        return "The approved analysis did not define a row-filtering subset cohort."
    subset = str(analysis["subset_cohort"])
    selected = cohorts.get(subset)
    if selected is None:
        return "The client could not identify the approved subset used to filter local rows."
    total = int(len(data))
    used = int(selected.sum())
    excluded = total - used
    if excluded == 0:
        return f"All {total} local rows met the approved {subset.replace('_', ' ')} criteria."
    cohort = next((item for item in contract.get("cohorts", []) if item.get("name") == subset), {})
    required = _required_non_missing_fields(cohort.get("predicate", {}))
    missing_required = [field for field in required if field in data and bool(data[field].isna().any())]
    if missing_required:
        labels = ", ".join(field.replace("_", " ") for field in missing_required[:3])
        if len(missing_required) > 3:
            labels += ", and other required fields"
        return f"{excluded} of {total} local rows were excluded because required fields were missing ({labels})."
    return f"{excluded} of {total} local rows did not meet the approved {subset.replace('_', ' ')} criteria."


def _required_non_missing_fields(predicate: Any) -> list[str]:
    if not isinstance(predicate, dict):
        return []
    if "all" in predicate and isinstance(predicate["all"], list):
        return sorted({field for item in predicate["all"] for field in _required_non_missing_fields(item)})
    inner = predicate.get("not")
    if isinstance(inner, dict) and isinstance(inner.get("is_missing"), dict):
        field = inner["is_missing"].get("field")
        return [field] if isinstance(field, str) and field else []
    return []
