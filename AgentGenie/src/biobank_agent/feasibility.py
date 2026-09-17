"""Schema-only feasibility review before human approval."""

from __future__ import annotations

from typing import Any

from biobank_agent.contracts import AnalysisContract


def assess_feasibility(contract: AnalysisContract, catalogs: list[dict[str, Any]]) -> dict[str, Any]:
    sites = []
    for catalog in catalogs:
        site_id = str(catalog.get("site_id"))
        raw_fields = set(catalog.get("schema_fields") or catalog.get("clinical_features", {}))
        harmonization = contract.payload.get("site_harmonization", {}).get(site_id, {}).get("fields", {})
        site_assessment = catalog.get("site_agent_assessment", {})
        proposed_adapter_fields = site_assessment.get("data_adapter", {}).get("fields", {})
        canonical_fields = set()
        invalid_mappings = []
        for canonical, specification in harmonization.items():
            source = specification.get("source", canonical) if isinstance(specification, dict) else canonical
            if source in raw_fields:
                canonical_fields.add(canonical)
            else:
                invalid_mappings.append({"canonical": canonical, "missing_source": source})
            proposed = proposed_adapter_fields.get(canonical, {})
            if not isinstance(proposed, dict) or _mapping_signature(proposed, canonical) != _mapping_signature(
                specification, canonical
            ):
                invalid_mappings.append(
                    {
                        "canonical": canonical,
                        "unverified_mapping": specification,
                        "reason": "the complete mapping was not proposed by the client-local data adapter agent",
                    }
                )
        available = raw_fields | canonical_fields
        cohort_fields = _cohort_fields(contract.payload["cohorts"])
        analyses = []
        for index, analysis in enumerate(contract.payload["analyses"]):
            required = _analysis_fields(analysis) | cohort_fields
            missing = sorted(required - available)
            analyses.append(
                {
                    "analysis_id": analysis.get("analysis_id", f"analysis_{index + 1}"),
                    "tool": analysis["tool"],
                    "required_fields": sorted(required),
                    "missing_fields": missing,
                    "supported": not missing,
                }
            )
        sites.append(
            {
                "site_id": site_id,
                "supported": [item for item in analyses if item["supported"]],
                "unsupported": [item for item in analyses if not item["supported"]],
                "invalid_harmonization_mappings": invalid_mappings,
                "site_agent_assessment_available": bool(site_assessment),
                "data_adapter": {
                    "status": site_assessment.get("data_adapter", {}).get("status"),
                    "digest": site_assessment.get("data_adapter", {}).get("digest"),
                    "fields": proposed_adapter_fields,
                    "unresolved": site_assessment.get("data_adapter", {}).get("unresolved", []),
                },
                "client_agent_proposal": {
                    "supported_concepts": site_assessment.get("supported_concepts", []),
                    "unavailable_concepts": site_assessment.get("unavailable_concepts", []),
                    "analysis_proposals": site_assessment.get("analysis_proposals", []),
                },
                "all_requested_analyses_supported": bool(analyses)
                and all(item["supported"] for item in analyses)
                and not invalid_mappings,
            }
        )
    return {
        "schema_version": "biobank.feasibility_report.v2",
        "study_id": contract.study_id,
        "schema_only": True,
        "patient_rows_accessed": False,
        "sites": sites,
        "unavailable_requests": contract.payload.get("unavailable_requests", []),
    }


def _mapping_signature(specification: Any, canonical: str) -> dict[str, Any] | None:
    if not isinstance(specification, dict):
        return None
    return {
        "source": specification.get("source", canonical),
        "multiply": float(specification.get("multiply", 1.0)),
        "value_map": specification.get("value_map", {}),
    }


def _analysis_fields(analysis: dict[str, Any]) -> set[str]:
    fields = set()
    for key in ("field", "time_field", "group_by"):
        if isinstance(analysis.get(key), str):
            fields.add(analysis[key])
    fields.update(item for item in analysis.get("fields", []) if isinstance(item, str))
    event = analysis.get("event")
    if isinstance(event, dict) and isinstance(event.get("field"), str):
        fields.add(event["field"])
    return fields


def _cohort_fields(definitions: list[dict[str, Any]]) -> set[str]:
    fields = set()
    for definition in definitions:
        fields |= _predicate_fields(definition.get("predicate"))
    return fields


def _predicate_fields(predicate: Any) -> set[str]:
    if not isinstance(predicate, dict):
        return set()
    fields = set()
    for operator, operand in predicate.items():
        if operator in {"all", "any"} and isinstance(operand, list):
            for item in operand:
                fields |= _predicate_fields(item)
        elif operator == "not":
            fields |= _predicate_fields(operand)
        elif isinstance(operand, dict) and isinstance(operand.get("field"), str):
            fields.add(operand["field"])
    return fields
