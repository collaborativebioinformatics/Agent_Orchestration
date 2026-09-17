"""NVFlare Controller implementing discovery, human approval, and analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from biobank_agent.aggregate import aggregate_site_results
from biobank_agent.codex import CodexPlanner
from biobank_agent.contracts import AnalysisContract
from biobank_agent.feasibility import assess_feasibility
from biobank_agent.flare import ANALYSIS_TASK, CATALOG_TASK, PROGRESS_TOPIC
from biobank_agent.flare.approval import ApprovalExpired, ApprovalRejected, HumanApprovalGate
from biobank_agent.flare.executor import PAYLOAD_KEY
from biobank_agent.render import render_report
from biobank_agent.tools.registry import TOOL_MANIFESTS

from nvflare.apis.controller_spec import Task
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal


class BiobankAnalysisController(Controller):
    """Own the complete analysis lifecycle, including the human gate.

    Patient data are first accessed only after ``HumanApprovalGate.wait``
    returns a validated, digest-bound approved contract.
    """

    def __init__(
        self,
        *,
        question: str | None,
        client_ids: list[str],
        output_dir: str = "runs",
        session_id: str = "biobank-study",
        initial_proposal: dict[str, Any] | None = None,
        codex_binary: str = "codex",
        codex_model: str | None = None,
        task_timeout: int = 600,
        approval_timeout: int = 86_400,
        approval_poll_seconds: float = 2.0,
        initial_prompt_timeout: int = 86_400,
    ) -> None:
        super().__init__()
        if not client_ids:
            raise ValueError("client_ids must not be empty")
        self.question = question
        self.client_ids = client_ids
        self.output_dir = output_dir
        self.session_id = session_id
        self.initial_proposal = initial_proposal
        self.codex_binary = codex_binary
        self.codex_model = codex_model
        self.task_timeout = task_timeout
        self.approval_timeout = approval_timeout
        self.approval_poll_seconds = approval_poll_seconds
        self.initial_prompt_timeout = initial_prompt_timeout

    def start_controller(self, fl_ctx: FLContext) -> None:
        self._runtime_gate = HumanApprovalGate(
            Path(self.output_dir) / self.session_id,
            poll_seconds=self.approval_poll_seconds,
            timeout_seconds=self.approval_timeout,
        )
        fl_ctx.get_engine().register_aux_message_handler(
            topic=PROGRESS_TOPIC,
            message_handle_func=self._handle_progress,
        )
        self.log_info(fl_ctx, "Biobank analysis Controller started")

    def stop_controller(self, fl_ctx: FLContext) -> None:
        self.log_info(fl_ctx, "Biobank analysis Controller stopped")

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        run_dir = Path(self.output_dir) / self.session_id
        gate = self._runtime_gate
        HumanApprovalGate._write(
            gate.tool_registry_path,
            {"schema_version": "biobank.tool_registry.v1", "tools": TOOL_MANIFESTS},
        )
        HumanApprovalGate._write(
            gate.participants_path,
            {"schema_version": "biobank.participants.v1", "clients": self.client_ids},
        )
        gate.record_event("startup", "NVFlare Controller started and is waiting for the study workflow.", status="active")
        question = self.question
        if not question:
            gate.timeout_seconds = self.initial_prompt_timeout
            try:
                request = gate.wait_for_question(aborted=lambda: abort_signal.triggered)
                question = str(request["question"])
                gate.record_event(
                    "question",
                    "Research question received; catalog-only Codex planning consent was recorded.",
                    actor="researcher",
                    status="completed",
                )
            except (ApprovalRejected, ApprovalExpired, ValueError) as exc:
                self.log_warning(fl_ctx, f"Biobank workflow stopped before discovery: {exc}")
                return
            finally:
                gate.timeout_seconds = self.approval_timeout

        gate.set_state("DISCOVERING", patient_data_accessed=False, question_received=True)
        gate.record_event(
            "catalog_dispatch",
            f"Dispatched catalog discovery to {len(self.client_ids)} clients.",
            status="active",
        )
        try:
            catalogs = self._discover_catalogs(question, abort_signal, fl_ctx, gate)
        except Exception as exc:
            self._fail(gate, fl_ctx, "catalog discovery", exc)
            return
        if abort_signal.triggered:
            gate.set_state("ABORTED", reason="Run aborted during catalog discovery")
            return

        gate.set_state("PLANNING", patient_data_accessed=False)
        gate.record_event(
            "planning",
            "Catalog metadata was sent to Codex to draft a tool-bounded analysis contract.",
            status="active",
        )
        try:
            proposal = self._create_proposal(question, catalogs, run_dir)
            feasibility = assess_feasibility(proposal, catalogs)
            gate.record_event(
                "review_synthesis",
                "Server agent is synthesizing the client proposals into concise human decisions.",
                status="active",
            )
            feasibility["server_agent_review"] = CodexPlanner(
                binary=self.codex_binary,
                model=self.codex_model,
            ).synthesize_human_review(
                question,
                proposal,
                feasibility,
                run_dir / "server" / "codex_review",
            )
            gate.publish(proposal, feasibility)
            gate.record_event(
                "planning",
                "Server contract and concise server-agent recommendation are ready for human review.",
                status="completed",
            )
        except Exception as exc:
            self._fail(gate, fl_ctx, "contract planning", exc)
            return
        self.log_info(
            fl_ctx,
            "Human approval required. Review "
            f"{gate.proposal_path} and {gate.feasibility_path}; write the decision to {gate.approval_path}.",
        )
        gate.record_event(
            "human_approval",
            "Waiting for the researcher to approve or reject the exact proposed contract.",
            actor="researcher",
            status="waiting",
        )
        try:
            approved = gate.wait(proposal, aborted=lambda: abort_signal.triggered)
        except ApprovalRejected as exc:
            self.log_warning(fl_ctx, f"Biobank analysis stopped at human gate: {exc}")
            if gate.new_question_path.exists() and not abort_signal.triggered:
                archived = gate.reset_for_new_question()
                self.log_info(fl_ctx, f"Archived study attempt at {archived}; reopening question gate")
                self.question = None
                self.control_flow(abort_signal, fl_ctx)
            return
        except ApprovalExpired as exc:
            self.log_warning(fl_ctx, f"Biobank analysis approval expired: {exc}")
            return
        gate.record_event(
            "human_approval",
            "Researcher approval validated against the proposal digest.",
            actor="researcher",
            status="completed",
        )

        # This is the first point at which a task capable of reading data.csv is dispatched.
        gate.set_state(
            "DISPATCHING_ANALYSIS",
            contract_digest=approved.digest,
            target_clients=self.client_ids,
            patient_data_accessed=True,
        )
        gate.record_event(
            "analysis_dispatch",
            f"Dispatched the approved contract to {len(self.client_ids)} clients.",
            status="active",
            patient_data_accessed=True,
        )
        try:
            site_results = self._dispatch_analysis(approved, abort_signal, fl_ctx, gate)
        except Exception as exc:
            self._fail(gate, fl_ctx, "approved analysis dispatch", exc)
            return
        if abort_signal.triggered:
            gate.set_state("ABORTED", reason="Run aborted during site analysis")
            return
        try:
            gate.record_event(
                "aggregation",
                "Server is pooling aggregate-only responses and rendering the report.",
                status="active",
                patient_data_accessed=True,
            )
            aggregate = aggregate_site_results(site_results, approved.payload)
            aggregate["contract_digest"] = approved.digest
            HumanApprovalGate._write(run_dir / "server" / "server_aggregate.json", aggregate)
            artifacts = render_report(aggregate, approved.payload, run_dir / "server" / "report")
        except Exception as exc:
            self._fail(gate, fl_ctx, "server aggregation and reporting", exc)
            return
        gate.set_state(
            "COMPLETED",
            contract_digest=approved.digest,
            site_count=len(site_results),
            patient_rows_received=aggregate["patient_rows_received"],
            artifacts=[str(path) for path in artifacts],
        )
        gate.record_event(
            "complete",
            "Federated analysis completed; aggregate report artifacts are ready.",
            status="completed",
            patient_data_accessed=True,
        )

    def _handle_progress(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """Persist a safe client-originated progress code for the live UI."""
        sender = request.get_peer_prop(ReservedKey.IDENTITY_NAME, default=None)
        if sender not in self.client_ids:
            return make_reply(ReturnCode.BAD_PEER_CONTEXT)
        event = request.get("event")
        definitions = {
            "client_connected": (
                "client_connected",
                "Client connected to the NVFlare run and is ready for tasks.",
                "completed",
                False,
            ),
            "client_disconnected": (
                "client_disconnected",
                "Client disconnected from the NVFlare run.",
                "completed",
                False,
            ),
            "catalog_started": (
                "catalog_local",
                "Client is inspecting its local catalog, mappings, and CSV header only.",
                "active",
                False,
            ),
            "catalog_completed": (
                "catalog_local",
                "Client finished catalog inspection and is responding with metadata only.",
                "completed",
                False,
            ),
            "site_planning_started": (
                "site_planning",
                "Local Codex site agent is interpreting the question against this site's metadata.",
                "active",
                False,
            ),
            "site_planning_completed": (
                "site_planning",
                "Local Codex site agent returned its capability and harmonization proposal.",
                "completed",
                False,
            ),
            "analysis_started": (
                "analysis_local",
                "Client is reading local patient data and executing only approved tools.",
                "active",
                True,
            ),
            "analysis_completed": (
                "analysis_local",
                "Client finished approved local analysis and is responding with aggregates.",
                "completed",
                True,
            ),
        }
        definition = definitions.get(event)
        if definition is None:
            return make_reply(ReturnCode.BAD_REQUEST_DATA)
        phase, message, status, patient_data_accessed = definition
        self._runtime_gate.record_event(
            phase,
            message,
            actor="client",
            site_id=sender,
            status=status,
            patient_data_accessed=patient_data_accessed,
        )
        return make_reply(ReturnCode.OK)

    def _fail(self, gate: HumanApprovalGate, fl_ctx: FLContext, phase: str, exc: Exception) -> None:
        reason = f"{phase} failed: {type(exc).__name__}: {str(exc)[:500]}"
        gate.set_state("FAILED", reason=reason)
        gate.record_event(phase.replace(" ", "_"), reason, status="failed")
        self.log_exception(fl_ctx, reason)
        self.system_panic(reason, fl_ctx)

    def _discover_catalogs(
        self,
        question: str,
        abort_signal: Signal,
        fl_ctx: FLContext,
        gate: HumanApprovalGate,
    ) -> list[dict[str, Any]]:
        request = Shareable(
            {
                PAYLOAD_KEY: {
                    "schema_version": "biobank.catalog_request.v1",
                    "requested_fields": [
                        "clinical_features",
                        "genomics",
                        "allowed_tools",
                        "schema_fields",
                        "declared_mappings_yaml",
                    ],
                    "patient_rows_allowed": False,
                    "question": question,
                }
            }
        )
        task = Task(name=CATALOG_TASK, data=request, timeout=self.task_timeout)
        self.broadcast_and_wait(
            task=task,
            targets=self.client_ids,
            min_responses=len(self.client_ids),
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )
        catalogs = []
        for client_task in task.client_tasks:
            result = client_task.result
            if not isinstance(result, Shareable) or result.get_return_code() not in {None, ReturnCode.OK}:
                raise RuntimeError(f"Catalog discovery failed for {client_task.client.name}")
            payload = result.get(PAYLOAD_KEY)
            if not isinstance(payload, dict) or not isinstance(payload.get("catalog"), dict):
                raise RuntimeError(f"Malformed catalog response from {client_task.client.name}")
            catalog = payload["catalog"]
            if catalog.get("site_id") != client_task.client.name:
                raise RuntimeError(f"Catalog identity mismatch for {client_task.client.name}")
            catalogs.append(catalog)
            gate.record_event(
                "catalog_response",
                "Client returned schema and mapping metadata; no patient rows or values were returned.",
                actor="client",
                site_id=client_task.client.name,
                status="completed",
            )
        if len(catalogs) != len(self.client_ids):
            raise RuntimeError("Did not receive every required site catalog")
        return catalogs

    def _create_proposal(
        self, question: str, catalogs: list[dict[str, Any]], run_dir: Path
    ) -> AnalysisContract:
        if self.initial_proposal is not None:
            proposal = AnalysisContract.parse(json.loads(json.dumps(self.initial_proposal)))
            if proposal.is_approved:
                raise ValueError("Configured initial_proposal must be unapproved")
            if proposal.payload["question"].strip() != question.strip():
                raise ValueError("Configured initial_proposal does not match the submitted research question")
            return proposal
        return CodexPlanner(binary=self.codex_binary, model=self.codex_model).plan(
            question,
            catalogs,
            run_dir / "server" / "codex_planning",
        )

    def _dispatch_analysis(
        self,
        contract: AnalysisContract,
        abort_signal: Signal,
        fl_ctx: FLContext,
        gate: HumanApprovalGate,
    ) -> list[dict[str, Any]]:
        contract.require_approval()
        request = Shareable(
            {
                PAYLOAD_KEY: {
                    "schema_version": "biobank.analysis_request.v1",
                    "contract": contract.payload,
                    "contract_digest": contract.digest,
                }
            }
        )
        task = Task(name=ANALYSIS_TASK, data=request, timeout=self.task_timeout)
        self.broadcast_and_wait(
            task=task,
            targets=self.client_ids,
            min_responses=len(self.client_ids),
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )
        results = []
        for client_task in task.client_tasks:
            result = client_task.result
            if not isinstance(result, Shareable) or result.get_return_code() not in {None, ReturnCode.OK}:
                raise RuntimeError(f"Analysis failed for {client_task.client.name}")
            payload = result.get(PAYLOAD_KEY)
            if not isinstance(payload, dict) or payload.get("contract_digest") != contract.digest:
                raise RuntimeError(f"Malformed or wrong-contract result from {client_task.client.name}")
            if payload.get("site_id") != client_task.client.name:
                raise RuntimeError(f"Analysis identity mismatch for {client_task.client.name}")
            if payload.get("patient_rows_exported") != 0:
                raise RuntimeError(f"Client {client_task.client.name} violated the aggregate-only boundary")
            results.append(payload)
            gate.record_event(
                "analysis_response",
                "Client returned approved aggregates; zero patient rows were exported.",
                actor="client",
                site_id=client_task.client.name,
                status="completed",
                patient_data_accessed=True,
            )
        if len(results) != len(self.client_ids):
            raise RuntimeError("Did not receive every required site result")
        return results
