"""Typed analysis-contract validation.

The model may propose a contract, but this module—not the model—decides whether
the proposal is executable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from biobank_agent.tools.registry import tool_names, validate_analysis
from biobank_agent.tools.dynamic import dynamic_manifests, validate_dynamic_tools
from biobank_agent.tools.registry import TOOL_MANIFESTS

APPROVED_TOOLS = tool_names()


@dataclass(frozen=True)
class AnalysisContract:
    payload: dict[str, Any]

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def approval_digest(self) -> str:
        """Digest of everything the researcher reviews, excluding approval metadata."""
        content = {key: value for key, value in self.payload.items() if key != "approval"}
        encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def study_id(self) -> str:
        return str(self.payload["study_id"])

    @property
    def is_approved(self) -> bool:
        approval = self.payload.get("approval")
        return (
            isinstance(approval, dict)
            and approval.get("status") == "approved"
            and approval.get("contract_digest") == self.approval_digest
        )

    def require_approval(self) -> None:
        if not self.payload.get("analyses"):
            raise PermissionError("Analysis contract contains no executable analyses")
        if not self.is_approved:
            raise PermissionError(
                "Analysis contract is not user-approved. Review it and run the approve command before execution."
            )

    @classmethod
    def parse(cls, value: dict[str, Any]) -> "AnalysisContract":
        if value.get("schema_version") != "biobank.analysis_contract.v2":
            raise ValueError("Unsupported or missing analysis-contract schema_version")
        if not str(value.get("study_id", "")).strip():
            raise ValueError("Contract requires a study_id")
        if not str(value.get("question", "")).strip():
            raise ValueError("Contract requires the original question")
        if value.get("training_allowed") is not False:
            raise ValueError("This application requires training_allowed=false")
        privacy = value.get("privacy")
        if not isinstance(privacy, dict) or int(privacy.get("min_cell_count", 0)) < 5:
            raise ValueError("Contract requires privacy.min_cell_count >= 5")
        dynamic_tools = validate_dynamic_tools(value.get("dynamic_tools", []))
        manifests = {**TOOL_MANIFESTS, **dynamic_manifests(dynamic_tools)}
        tools = set(value.get("approved_tools", []))
        unknown_tools = tools - set(manifests)
        if unknown_tools:
            raise ValueError(f"Tools are not approved: {sorted(unknown_tools)}")
        analyses = value.get("analyses")
        if not isinstance(analyses, list):
            raise ValueError("Contract analyses must be a list")
        if not analyses and not value.get("unavailable_requests"):
            raise ValueError("Contract requires an analysis or an explicit unavailable request")
        for item in analyses:
            if not isinstance(item, dict) or item.get("tool") not in tools:
                raise ValueError("Every analysis must name an approved tool")
            validate_analysis(item, manifests)
        cohorts = value.get("cohorts")
        if not isinstance(cohorts, list):
            raise ValueError("Contract cohorts must be a list")
        cohort_required = any(
            "cohorts" in manifests[item["tool"]].get("required_parameters", []) or item.get("subset_cohort")
            for item in analyses
        )
        if cohort_required and not cohorts:
            raise ValueError("Executable analyses require declarative cohort definitions")
        names = [item.get("name") for item in cohorts if isinstance(item, dict)]
        if len(names) != len(cohorts) or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("Every cohort requires a name")
        if len(set(names)) != len(names):
            raise ValueError("Cohort names must be unique")
        for cohort in cohorts:
            _validate_predicate_shape(cohort.get("predicate"))
        referenced = {name for item in analyses for name in item.get("cohorts", [])}
        referenced |= {item.get("subset_cohort") for item in analyses if item.get("subset_cohort")}
        unknown_cohorts = referenced - set(names)
        if unknown_cohorts:
            raise ValueError(f"Analyses reference undefined cohorts: {sorted(unknown_cohorts)}")
        harmonization = value.get("site_harmonization", {})
        if not isinstance(harmonization, dict):
            raise ValueError("site_harmonization must be an object")
        return cls(value)


def _validate_predicate_shape(predicate: Any) -> None:
    if not isinstance(predicate, dict) or len(predicate) != 1:
        raise ValueError("Each cohort predicate must contain exactly one operator")
    operator, operand = next(iter(predicate.items()))
    if operator in {"all", "any"}:
        if not isinstance(operand, list) or not operand:
            raise ValueError(f"Predicate {operator} requires a non-empty list")
        for item in operand:
            _validate_predicate_shape(item)
        return
    if operator == "not":
        _validate_predicate_shape(operand)
        return
    if operator not in {"eq", "eq_ci", "in", "in_ci", "lt", "le", "gt", "ge", "is_missing"}:
        raise ValueError(f"Predicate operator is not approved: {operator}")
    if not isinstance(operand, dict) or not isinstance(operand.get("field"), str):
        raise ValueError(f"Predicate {operator} requires a field")
    if operator in {"in", "in_ci"} and not isinstance(operand.get("values"), list):
        raise ValueError(f"Predicate {operator} requires values")
    if operator not in {"in", "in_ci", "is_missing"} and "value" not in operand:
        raise ValueError(f"Predicate {operator} requires value")
