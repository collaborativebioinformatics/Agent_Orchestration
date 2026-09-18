"""Request a bounded analysis contract from the Codex CLI."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from biobank_agent.contracts import AnalysisContract
from biobank_agent.tools.registry import TOOL_MANIFESTS


class CodexPlanner:
    """The workflow's only generative-agent backend."""

    def __init__(self, binary: str = "codex", model: str | None = None, timeout: int = 600) -> None:
        self.binary = binary
        self.model = model
        self.timeout = timeout

    def plan(self, question: str, catalogs: list[dict[str, Any]], work_dir: Path) -> AnalysisContract:
        prompt = self._prompt(question, catalogs)
        validation_error: ValueError | None = None
        for attempt in range(3):
            value = self.run_json(prompt, work_dir / f"attempt-{attempt + 1}")
            try:
                return AnalysisContract.parse(value)
            except ValueError as exc:
                validation_error = exc
                prompt = self._contract_repair_prompt(prompt, value, exc)
        raise ValueError(f"Server agent could not produce a valid analysis contract after two retries: {validation_error}")

    @staticmethod
    def _contract_repair_prompt(original_prompt: str, invalid: dict[str, Any], error: ValueError) -> str:
        return f"""{original_prompt}

The deterministic contract verifier rejected your previous JSON:
{error}

Rejected JSON:
{json.dumps(invalid, indent=2, sort_keys=True)}

Return a corrected complete contract. In each analyses item, put every tool parameter
directly at the top level beside analysis_id and tool. Never use a nested parameters
object. For federated_kaplan_meier this means top-level time_field, event, group_by,
time_bins, and optional subset_cohort. For missingness_summary, fields is top-level.
For federated_kaplan_meier, event MUST be an object and time_bins MUST be an array:
"event": {{"field": "canonical_event_field", "values": [1]}},
"time_bins": [0, 12, 24, 36]. Never emit event as a string and never wrap time_bins
in a boundaries object. Select event values only from the client-proposed endpoint
encoding; if that encoding is unresolved, do not emit the analysis.
Every cohort predicate must contain a real non-empty predicate; never emit all: [].
Use the exact complete field specifications proposed by each client data adapter,
including source, multiply, and value_map. Preserve the verified
__OTHER_NON_MISSING__ default when a client proposed it; missing values remain missing
without an explicit __MISSING__ entry. Preserve all privacy and scientific caveats.
If the rejected contract contains a dynamic tool, preserve its complete dynamic_tools
entry and correct it according to the verifier error. Dynamic source contains exactly
one function and no imports: def run_local(data, analysis, min_cell_count) for the client
and def run_server(outputs, analysis) for the server.
"""

    def synthesize_human_review(
        self,
        question: str,
        contract: AnalysisContract,
        feasibility: dict[str, Any],
        work_dir: Path,
    ) -> dict[str, Any]:
        """Condense client proposals into a human decision and hidden revision payload."""
        prompt = f"""You are the server agent at the human decision boundary of a
federated biobank workflow. Read the client-agent proposals and the proposed server
contract below. Return exactly one JSON object and no prose. Do not invent patient
values, encodings, mappings, or scientific equivalences.

Research question:
{question}

Proposed server contract:
{json.dumps(contract.payload, indent=2, sort_keys=True)}

Client-agent proposals and feasibility:
{json.dumps(feasibility, indent=2, sort_keys=True)}

Contract analysis parameters are intentionally top-level beside analysis_id and tool,
as required by the registered tool manifests. The nested parameters object in each
client analysis_proposals item is only that proposal message's shape; do not ask to
move valid contract parameters into it.

Return this exact shape:
{{
  "schema_version": "biobank.server_agent_review.v1",
  "action": "approve or revise",
  "summary": "at most two short plain-language sentences",
  "confirmation_items": ["three to six concise scientific or governance decisions"],
  "full_revision_guidance": "complete, precise instructions for all client agents and the server planner"
}}

If the contract has no executable analyses, action must be revise. Confirmation items
must describe decisions a researcher can understand, not raw adapter dumps. Put exact
canonical field names, site-specific sources and conversions, event/censor rules,
missing-data rules, tool parameters, shared bins, and privacy requirements in
full_revision_guidance. A registered tool is eligible for human approval even when a
legacy catalog allowed_tools field omits it. If action is approve, leave
full_revision_guidance empty and summarize the exact endpoint, population, tools,
harmonization assumptions, and privacy policy being approved.
For every dynamic tool, explicitly tell the researcher that agent-generated local and
server source code is included in the approval-bound contract. Summarize its local
aggregates, pooling method, missing-data rule, and privacy checks. Never describe a
dynamic tool as pre-validated or registered.

When every site reports all_requested_analyses_supported=true, recommend approve. Do
not request another revision for assumptions already incorporated into the proposed
contract and verified client adapters.

Do not ask the researcher to supply source encodings, category vocabularies, or other
technical metadata that the client agents explicitly reported as unavailable. When the
requested endpoint is unavailable but every client proposes the same scientifically
meaningful executable alternative, recommend that alternative as an explicit human
decision and rename it accurately; never represent it as the original endpoint. Prefer
a runnable, honestly labeled descriptive analysis over instructions that merely restate
an unavailable-data blocker. You may require exact canonical names across clients,
declared unit conversion, complete-case exclusion, preservation of opaque category
labels, and common tool parameters when those operations are supported by the client
proposals.
"""
        value = self.run_json(prompt, work_dir)
        sites = feasibility.get("sites", [])
        all_sites_support_contract = bool(sites) and all(
            site.get("all_requested_analyses_supported") is True for site in sites
        )
        if contract.payload.get("analyses") and all_sites_support_contract:
            value["action"] = "approve"
            value["summary"] = (
                "All participating sites support the proposed analyses and their verified local adapters match "
                "the server contract. Confirm the analysis contract before execution."
            )
            value["confirmation_items"] = [
                "Confirm the displayed endpoint, cohorts, and requested analyses.",
                "Confirm the displayed site-specific harmonization and shared tool parameters.",
                "Confirm the minimum-cell and aggregate-only privacy controls.",
            ]
            if contract.payload.get("dynamic_tools"):
                value["confirmation_items"].insert(
                    2,
                    "Confirm the displayed agent-generated local and server algorithm source included in this contract.",
                )
            value["full_revision_guidance"] = ""
        if value.get("schema_version") != "biobank.server_agent_review.v1":
            raise ValueError("Unsupported server-agent review schema")
        if value.get("action") not in {"approve", "revise"}:
            raise ValueError("Server-agent review action must be approve or revise")
        if not contract.payload.get("analyses") and value.get("action") != "revise":
            raise ValueError("A non-executable contract must be revised")
        items = value.get("confirmation_items")
        if not isinstance(items, list) or not 1 <= len(items) <= 6 or not all(
            isinstance(item, str) and item.strip() for item in items
        ):
            raise ValueError("Server-agent review requires one to six confirmation items")
        if not isinstance(value.get("summary"), str) or not value["summary"].strip():
            raise ValueError("Server-agent review requires a summary")
        guidance = value.get("full_revision_guidance")
        if value["action"] == "revise" and (not isinstance(guidance, str) or not guidance.strip()):
            raise ValueError("A revision recommendation requires full guidance")
        value["revision_digest"] = hashlib.sha256(str(guidance or "").encode()).hexdigest()
        return value

    def run_json(self, prompt: str, work_dir: Path) -> dict[str, Any]:
        """Run the Codex-only backend and parse its final message as JSON."""
        work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=work_dir, suffix=".json", delete=False) as handle:
            output_path = Path(handle.name)
        command = [
            self._resolve_binary(),
            "--ask-for-approval",
            "never",
            "exec",
            "--skip-git-repo-check",
            "--cd",
            str(work_dir.resolve()),
            "--sandbox",
            "read-only",
            "--output-last-message",
            str(output_path.resolve()),
            "--color",
            "never",
        ]
        if self.model:
            command.extend(["--model", self.model])
        command.append("-")
        result = subprocess.run(command, input=prompt, text=True, capture_output=True, timeout=self.timeout)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "no process output")[-1000:]
            raise RuntimeError(f"Codex contract planning failed (exit {result.returncode}): {detail}")
        raw = output_path.read_text(encoding="utf-8").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Codex output must be a JSON object")
        return parsed

    def _resolve_binary(self) -> str:
        """Resolve the Codex backend, preferring the desktop-bundled CLI on macOS.

        The Homebrew Node launcher can be killed when invoked recursively from
        the Codex desktop host. Its bundled native executable uses the existing
        signed-in Codex session and remains the same Codex-only backend.
        """
        override = os.environ.get("BIOBANK_CODEX_BINARY")
        if override:
            return override
        if self.binary != "codex":
            return self.binary
        desktop_binary = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
        if desktop_binary.is_file():
            return str(desktop_binary)
        return shutil.which(self.binary) or self.binary

    @staticmethod
    def _prompt(question: str, catalogs: list[dict[str, Any]]) -> str:
        return f"""You are the server agent for a federated, analysis-only biobank study.
Return exactly one JSON object and no prose. Never request row-level data, identifiers,
model training, or a tool/field outside the allowlists below. Use catalog availability
to mark requested but unavailable concepts explicitly; do not invent proxy variables.

Question:
{question}

Server-visible site catalogs:
{json.dumps(catalogs, indent=2, sort_keys=True)}

General-purpose tool registry:
{json.dumps(TOOL_MANIFESTS, indent=2, sort_keys=True)}

This registry is the preferred toolbox. Catalog allowed_tools fields are legacy
descriptive metadata and must not be treated as a second execution gate. If no registered
tool correctly implements the question and client dynamic_tool_requests agree on the
needed aggregate computation, author a general-purpose dynamic tool. Its complete source
becomes part of the digest-bound contract and cannot execute until human approval.

The object must follow this exact shape (arrays must remain arrays):
{{
  "schema_version": "biobank.analysis_contract.v2",
  "study_id": "short_identifier",
  "question": "the original question",
  "training_allowed": false,
  "cohorts": [
    {{"name": "cohort_name", "predicate": {{"all": [
      {{"eq_ci": {{"field": "canonical_field", "value": "value"}}}}
    ]}}}}
  ],
  "site_harmonization": {{
    "exact_site_id": {{"fields": {{
      "canonical_field": {{"source": "existing_source_field", "value_map": {{}}, "multiply": 1.0}}
    }}}}
  }},
  "approved_tools": ["registered_tool_name"],
  "dynamic_tools": [{{
    "name": "dynamic_general_purpose_name",
    "version": 1,
    "description": "scientific operation without disease-specific semantics",
    "required_parameters": ["fields"],
    "local_output": "aggregate-only output description",
    "server_operation": "pooling and final calculation description",
    "local_code": "def run_local(data, analysis, min_cell_count):\n    ...",
    "server_code": "def run_server(outputs, analysis):\n    ..."
  }}],
  "privacy": {{"min_cell_count": 10, "forbidden_outputs": ["row_level", "patient_ids", "exact_min_max"]}},
  "analyses": [{{"analysis_id": "id", "tool": "registered_tool_name", "cohorts": ["cohort_name"]}}],
  "unavailable_requests": [{{"concept": "...", "reason": "..."}}],
  "approval": {{"status": "proposed"}}
}}
Predicate operators are exactly all, any, not, eq, eq_ci, in, in_ci, lt, le,
gt, ge, and is_missing; each leaf maps the operator to an object containing
field and value/values. Parameters in analyses must match the selected tool
manifest. If nothing is executable, return empty cohorts, approved_tools, and
analyses arrays plus non-empty unavailable_requests so the researcher can review
the site agents' evidence instead of receiving a workflow error. The tools are
proposals until a human changes approval through the workflow's approve command.
Translate the user's scientific concepts into the contract. Do not put disease-
specific assumptions in tool names and do not use a proxy for an unavailable field.
Each catalog contains a client-local site_agent_assessment with a declarative
data_adapter. A site_harmonization mapping may be promoted only when the exact
canonical field and source were proposed by that site's adapter. Preserve any
unresolved adapter concepts in unavailable_requests. The server must not invent
or override a client-local mapping.
Every tool parameter must be placed directly in its analyses item; never create a
nested parameters object. Every all/any predicate must contain at least one child.
Registered-tool parameter shapes are exact. In particular, a Kaplan–Meier analysis uses:
{{"analysis_id":"id","tool":"federated_kaplan_meier",
"time_field":"canonical_time","event":{{"field":"canonical_event","values":[1]}},
"group_by":"canonical_group","time_bins":[0,12,24,36]}}.
The event member is never a string, and time_bins is never an object. Choose event
values only from the client-proposed endpoint encoding.

Dynamic implementation interface and constraints:
- Use a dynamic tool only when the fixed registry cannot correctly perform the task.
- Its name starts with dynamic_ and appears in approved_tools and analyses.
- local_code contains exactly def run_local(data, analysis, min_cell_count). It receives
  a pandas DataFrame limited to analysis.fields, the analysis object, and the privacy
  threshold. It returns finite JSON with schema_version
  biobank.dynamic_local_output.v1 and only disclosure-controlled aggregates.
- server_code contains exactly def run_server(outputs, analysis). It receives only local
  aggregate objects and returns finite JSON.
- No imports, filesystem, network, processes, reflection, private/dunder attributes,
  identifiers, row records, or patient values. numpy is available as np, pandas as pd,
  and math as math.
- Suppress each local result whose n is below min_cell_count. Never reconstruct a
  suppressed value at the server. Use pairwise complete cases when required.
- Source, parameters, adapters, and privacy rules are all reviewed and approval-bound.
"""
