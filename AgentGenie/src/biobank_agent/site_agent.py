"""Codex-backed, metadata-only planning inside each site boundary."""

from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path
from typing import Any

from biobank_agent.codex import CodexPlanner
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
    ) -> dict[str, Any]:
        prompt = self._prompt(question, catalog, mappings_yaml)
        validation_error: ValueError | None = None
        with tempfile.TemporaryDirectory(prefix=f"biobank-site-agent-{catalog.get('site_id', 'site')}-") as directory:
            work_dir = Path(directory)
            for attempt in range(2):
                result = self.planner.run_json(prompt, work_dir / f"attempt-{attempt + 1}")
                try:
                    self._validate(result, catalog)
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
    def _prompt(question: str, catalog: dict[str, Any], mappings_yaml: str) -> str:
        return f"""You are a site agent inside one federated biobank boundary.
Inspect only the supplied catalog, CSV header field names, and declared mappings. Do not
request, infer, or emit patient values. Decide locally which scientific concepts are
supported, how local fields could map to canonical names, and which registered aggregate-
only tools could execute the request. A field's presence does not prove an encoding; flag
unknown value encodings explicitly. Do not substitute mRNA z-scores for raw expression or
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

Declared mappings YAML:
{mappings_yaml}

Registered analysis primitives:
{json.dumps(TOOL_MANIFESTS, indent=2, sort_keys=True)}

The registered analysis primitives above are the current server registry presented for
human approval. Treat that registry as authoritative for proposing tools. A catalog's
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
  "analysis_proposals": [{{"tool": "registered tool name", "parameters": {{}}, "caveats": ["..."]}}]
}}
Use empty arrays or objects when needed. Never invent a source field or tool name.
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
        unknown = sorted(
            str(item.get("tool") if isinstance(item, dict) else item)
            for item in analyses
            if not isinstance(item, dict) or item.get("tool") not in TOOL_MANIFESTS
        )
        if unknown:
            raise ValueError(f"Site agent proposed unregistered tools: {unknown}")
