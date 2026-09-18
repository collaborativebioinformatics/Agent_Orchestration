"""Codex-backed adapter generation inside each site boundary."""

from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path
from typing import Any

from biobank_agent.codex import CodexPlanner
from biobank_agent.local_profile import verify_adapter_locally
from biobank_agent.tools.registry import TOOL_MANIFESTS


class CodexSiteAgent:
    """Interpret a user question against one site's own declared metadata."""

    def __init__(self, binary: str = "codex", model: str | None = None) -> None:
        self.planner = CodexPlanner(binary=binary, model=model)

    def assess(
        self,
        *,
        question: str,
        catalog: dict[str, Any],
        mappings_yaml: str,
        local_profile: dict[str, Any] | None = None,
        local_data: Any | None = None,
    ) -> dict[str, Any]:
        # mappings_yaml remains in the API for compatibility with older callers,
        # but is intentionally not supplied to the agent or used as a fallback.
        prompt = self._prompt(question, catalog, local_profile or {})
        validation_error: ValueError | None = None
        with tempfile.TemporaryDirectory(prefix=f"biobank-site-agent-{catalog.get('site_id', 'site')}-") as directory:
            work_dir = Path(directory)
            for attempt in range(2):
                result = self.planner.run_json(prompt, work_dir / f"attempt-{attempt + 1}")
                try:
                    self._validate(result, catalog)
                    if local_data is not None:
                        verify_adapter_locally(
                            result["data_adapter"], local_data, str((local_profile or {}).get("digest", ""))
                        )
                    break
                except ValueError as exc:
                    validation_error = exc
                    prompt = self._repair_prompt(prompt, catalog, exc)
            else:
                result = self._incomplete_assessment(catalog, validation_error)
        adapter = result["data_adapter"]
        adapter["digest"] = hashlib.sha256(
            json.dumps(adapter, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return result

    @staticmethod
    def _repair_prompt(original_prompt: str, catalog: dict[str, Any], error: ValueError) -> str:
        fields = catalog.get("schema_fields", [])
        return f"""{original_prompt}

Your previous JSON was rejected by the deterministic boundary verifier:
{error}

Return a corrected JSON object. The following CSV header names are the absolute source field allowlist for this site:
{json.dumps(fields, indent=2)}

A canonical output such as time_months may be a key in data_adapter.fields, but its
source must be one exact header above. Express a unit conversion with multiply (for
example, years to months uses 12.0). Catalog concepts and mapping YAML entries can be
stale or derived and are not valid source names unless they also occur in this header
allowlist. If a safe mapping cannot be made, omit it from fields, put it in unresolved,
set status to incomplete, and omit analyses that depend on it.
"""

    @staticmethod
    def _incomplete_assessment(catalog: dict[str, Any], error: ValueError | None) -> dict[str, Any]:
        reason = f"Local adapter proposal failed boundary verification after one retry: {error}"
        return {
            "schema_version": "biobank.site_agent_assessment.v1",
            "site_id": catalog.get("site_id"),
            "supported_concepts": [],
            "unavailable_concepts": [{"concept": "verified local data adapter", "reason": reason}],
            "data_adapter": {
                "schema_version": "biobank.declarative_data_adapter.v1",
                "status": "incomplete",
                "fields": {},
                "unresolved": [{"canonical_field": "requested analysis fields", "reason": reason}],
            },
            "analysis_proposals": [],
        }

    @staticmethod
    def _prompt(
        question: str, catalog: dict[str, Any], local_profile: dict[str, Any]
    ) -> str:
        return f"""You are a site agent inside one federated biobank boundary.
Inspect the supplied catalog, CSV header names, and disclosure-controlled local profile.
Do not request or emit patient rows. Decide locally which scientific concepts are
supported, how local fields could map to canonical names, and which registered aggregate-
only tools could execute the request. Generate the adapter from locally observed profile
evidence. Do not rely on a pre-supplied mapping and do not expect a deterministic mapping
fallback. A field's presence alone does not prove an
encoding; use observed categories and privacy-safe associations when available. Do not
substitute mRNA z-scores for raw expression or
claim log2 fold changes from z-scores. Return exactly one JSON object and no prose.
The catalog's schema_fields array is the absolute allowlist for every data-adapter
source. Other catalog concepts and declared mapping entries may describe derived or
stale fields; never use one as a source unless it is also in schema_fields. Canonical
field names are outputs, not sources. You may map an existing raw duration into a
canonical duration and use multiply for a declared unit conversion.

Research question:
{question}

Local catalog and header:
{json.dumps(catalog, indent=2, sort_keys=True)}

Disclosure-controlled profile generated locally from this site's data:
{json.dumps(local_profile, indent=2, sort_keys=True)}

Registered analysis primitives:
{json.dumps(TOOL_MANIFESTS, indent=2, sort_keys=True)}

The registered analysis primitives above are the preferred server toolbox presented for
human approval. Prefer them when they correctly implement the request. If none can do so,
request a new general-purpose dynamic algorithm rather than declaring an otherwise
feasible analysis unavailable. Describe the aggregate statistics the client must produce
and how the server should combine them; the server agent will author the implementation,
and its complete source will require explicit human approval. A catalog's
allowed_tools field is legacy descriptive metadata, not a second execution allowlist;
do not reject or caveat a registered tool merely because it is absent there. Nothing
executes until the researcher approves the final contract.

Return this exact shape:
{{
  "schema_version": "biobank.site_agent_assessment.v1",
  "site_id": "the exact catalog site_id",
  "supported_concepts": [{{"concept": "...", "source_fields": ["..."], "evidence": "..."}}],
  "unavailable_concepts": [{{"concept": "...", "reason": "..."}}],
  "data_adapter": {{
    "schema_version": "biobank.declarative_data_adapter.v1",
    "status": "ready or incomplete",
    "fields": {{"canonical_field": {{"source": "existing header field", "value_map": {{}}, "multiply": 1.0}}}},
    "unresolved": [{{"canonical_field": "...", "reason": "..."}}]
  }},
  "analysis_proposals": [{{"tool": "registered tool name or requested dynamic name", "parameters": {{}}, "caveats": ["..."]}}],
  "dynamic_tool_requests": [{{
    "name": "dynamic_general_purpose_name",
    "purpose": "...",
    "required_fields": ["canonical_field"],
    "local_aggregate_requirements": ["..."],
    "server_computation": "...",
    "privacy_constraints": ["..."]
  }}]
}}
Use empty arrays or objects when needed. Never invent a source field. A dynamic tool name
must start with dynamic_ and appear identically in dynamic_tool_requests and any related
analysis_proposals.
The deterministic boundary verifier checks only safety, structure, observed categorical
coverage, and binary event output shape. It never creates, repairs, or substitutes a
semantic mapping. Your data-supported adapter is the sole mapping sent to the server; if
the evidence is insufficient, mark the field unresolved.
"""

    @staticmethod
    def _validate(value: dict[str, Any], catalog: dict[str, Any]) -> None:
        if value.get("schema_version") != "biobank.site_agent_assessment.v1":
            raise ValueError("Unsupported site-agent assessment schema")
        if value.get("site_id") != catalog.get("site_id"):
            raise ValueError("Site-agent assessment identity mismatch")
        fields = set(catalog.get("schema_fields", []))
        adapter = value.get("data_adapter")
        if not isinstance(adapter, dict) or adapter.get("schema_version") != "biobank.declarative_data_adapter.v1":
            raise ValueError("Site agent must return a declarative data adapter")
        proposals = adapter.get("fields", {})
        if not isinstance(proposals, dict):
            raise ValueError("data_adapter.fields must be an object")
        for specification in proposals.values():
            if not isinstance(specification, dict) or not isinstance(specification.get("value_map"), dict):
                continue
            normalized_map = {}
            for key, mapped_value in specification["value_map"].items():
                sentinel = str(key).strip().upper().replace("__OTHER_NONMISSING__", "__OTHER_NON_MISSING__")
                # Missing source values remain missing without an explicit mapping.
                if sentinel == "__MISSING__" and mapped_value is None:
                    continue
                normalized_map[sentinel if sentinel == "__OTHER_NON_MISSING__" else key] = mapped_value
            specification["value_map"] = normalized_map
        invented = sorted(
            str(spec.get("source"))
            for spec in proposals.values()
            if not isinstance(spec, dict)
            or not isinstance(spec.get("source"), str)
            or spec.get("source") not in fields
        )
        if invented:
            raise ValueError(f"Site agent invented source fields: {invented}")
        analyses = value.get("analysis_proposals", [])
        if not isinstance(analyses, list):
            raise ValueError("analysis_proposals must be a list")
        requests = value.get("dynamic_tool_requests", [])
        if not isinstance(requests, list):
            raise ValueError("dynamic_tool_requests must be a list")
        dynamic_names = {
            item.get("name")
            for item in requests
            if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"].startswith("dynamic_")
        }
        malformed_requests = [
            item for item in requests if not isinstance(item, dict) or item.get("name") not in dynamic_names
        ]
        if malformed_requests:
            raise ValueError("Every dynamic tool request requires a dynamic_<name> identifier")
        unknown = sorted(
            str(item.get("tool") if isinstance(item, dict) else item)
            for item in analyses
            if not isinstance(item, dict) or item.get("tool") not in set(TOOL_MANIFESTS) | dynamic_names
        )
        if unknown:
            raise ValueError(f"Site agent proposed unregistered tools: {unknown}")
