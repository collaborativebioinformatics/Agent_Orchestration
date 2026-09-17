"""Durable human-approval state owned by the server workflow."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from biobank_agent.contracts import AnalysisContract


class ApprovalRejected(RuntimeError):
    pass


class ApprovalExpired(TimeoutError):
    pass


class HumanApprovalGate:
    """Publish a proposal and wait for an exact, digest-bound human decision."""

    def __init__(self, run_dir: Path, poll_seconds: float = 2.0, timeout_seconds: float = 86_400.0) -> None:
        self.run_dir = run_dir
        self.server_dir = run_dir / "server"
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.proposal_path = self.server_dir / "analysis_contract.proposed.json"
        self.request_path = self.server_dir / "user_request.json"
        self.feasibility_path = self.server_dir / "feasibility_report.json"
        self.tool_registry_path = self.server_dir / "tool_registry.json"
        self.approval_path = self.server_dir / "human_approval.json"
        self.approved_contract_path = self.server_dir / "analysis_contract.approved.json"
        self.state_path = self.server_dir / "workflow_state.json"
        self.events_path = self.server_dir / "workflow_events.json"
        self.participants_path = self.server_dir / "participants.json"
        self.new_question_path = self.server_dir / "new_question_request.json"
        self._events_lock = threading.Lock()

    def wait_for_question(self, aborted: Callable[[], bool]) -> dict[str, Any]:
        """Wait for the UI to submit the initial human research question."""
        self.server_dir.mkdir(parents=True, exist_ok=True)
        self.set_state(
            "WAITING_FOR_QUESTION",
            request_path=str(self.request_path),
            patient_data_accessed=False,
        )
        started = time.monotonic()
        while True:
            if aborted():
                self.set_state("ABORTED", reason="NVFlare abort signal triggered while awaiting a question")
                raise ApprovalRejected("NVFlare run aborted while awaiting the initial question")
            if self.request_path.exists():
                request = json.loads(self.request_path.read_text(encoding="utf-8"))
                if request.get("schema_version") != "biobank.user_request.v1":
                    raise ValueError("Malformed initial user-request schema")
                question = str(request.get("question", "")).strip()
                if not question:
                    raise ValueError("Initial user request contains an empty question")
                if request.get("catalog_metadata_codex_consent") is not True:
                    raise ValueError("Catalog-metadata consent for Codex planning was not granted")
                return request
            if time.monotonic() - started >= self.timeout_seconds:
                self.set_state("EXPIRED", reason="Initial-question timeout elapsed")
                raise ApprovalExpired("A research question was not received before the timeout")
            time.sleep(self.poll_seconds)

    def publish(self, contract: AnalysisContract, feasibility: dict[str, Any]) -> None:
        if contract.is_approved:
            raise ValueError("Controller must publish a proposal, not an already approved contract")
        self.server_dir.mkdir(parents=True, exist_ok=True)
        self._write(self.proposal_path, contract.payload)
        self._write(self.feasibility_path, feasibility)
        self.set_state(
            "WAITING_FOR_APPROVAL",
            proposal_digest=contract.approval_digest,
            review_bundle_digest=_review_bundle_digest(contract.payload, feasibility),
            approval_path=str(self.approval_path),
            patient_data_accessed=False,
        )

    def wait(self, contract: AnalysisContract, aborted: Callable[[], bool]) -> AnalysisContract:
        started = time.monotonic()
        while True:
            if aborted():
                self.set_state("ABORTED", reason="NVFlare abort signal triggered while awaiting approval")
                raise ApprovalRejected("NVFlare run aborted while awaiting human approval")
            if self.approval_path.exists():
                return self._consume_decision(contract)
            if time.monotonic() - started >= self.timeout_seconds:
                self.set_state("EXPIRED", reason="Human approval timeout elapsed")
                raise ApprovalExpired("Human approval was not received before the timeout")
            time.sleep(self.poll_seconds)

    def _consume_decision(self, contract: AnalysisContract) -> AnalysisContract:
        decision = json.loads(self.approval_path.read_text(encoding="utf-8"))
        if decision.get("schema_version") != "biobank.human_approval.v1":
            self.set_state("REJECTED", reason="Malformed human approval schema")
            raise ApprovalRejected("Malformed human approval schema")
        if decision.get("proposal_digest") != contract.approval_digest:
            self.set_state("REJECTED", reason="Approval digest does not match published proposal")
            raise ApprovalRejected("Approval does not match the published proposal")
        feasibility = json.loads(self.feasibility_path.read_text(encoding="utf-8"))
        expected_review_digest = _review_bundle_digest(contract.payload, feasibility)
        if decision.get("review_bundle_digest") != expected_review_digest:
            self.set_state("REJECTED", reason="Approval does not match the published client and server review bundle")
            raise ApprovalRejected("Approval does not match the published client and server review bundle")
        if decision.get("decision") != "approve":
            self.set_state("REJECTED", reason=str(decision.get("reason") or "Researcher rejected proposal"))
            raise ApprovalRejected("Researcher rejected the analysis proposal")
        if not contract.payload.get("analyses"):
            self.set_state("REJECTED", reason="Proposal contains no executable analyses")
            raise ApprovalRejected("A proposal with no executable analyses cannot be approved")
        approved_payload = json.loads(json.dumps(contract.payload))
        approved_payload["approval"] = {
            "status": "approved",
            "approved_by": str(decision.get("approved_by") or "researcher"),
            "approved_at": str(decision.get("decided_at") or _timestamp()),
            "contract_digest": contract.approval_digest,
            "source": "nvflare_controller_human_approval_gate",
        }
        approved = AnalysisContract.parse(approved_payload)
        approved.require_approval()
        self._write(self.approved_contract_path, approved.payload)
        self.set_state("APPROVED", proposal_digest=contract.approval_digest, approved_by=decision.get("approved_by"))
        return approved

    def set_state(self, status: str, **details: Any) -> None:
        self._write(
            self.state_path,
            {
                "schema_version": "biobank.nvflare_workflow_state.v1",
                "status": status,
                "updated_at": _timestamp(),
                **details,
            },
        )

    def record_event(
        self,
        phase: str,
        message: str,
        *,
        actor: str = "server",
        site_id: str | None = None,
        status: str = "info",
        patient_data_accessed: bool = False,
    ) -> None:
        """Append a UI-safe lifecycle event without including patient data."""
        with self._events_lock:
            events: list[dict[str, Any]] = []
            if self.events_path.exists():
                loaded = json.loads(self.events_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    events = loaded
            events.append(
                {
                    "sequence": len(events) + 1,
                    "timestamp": _timestamp(),
                    "phase": phase,
                    "actor": actor,
                    "site_id": site_id,
                    "status": status,
                    "message": message,
                    "patient_data_accessed": patient_data_accessed,
                }
            )
            self._write(self.events_path, events)

    def reset_for_new_question(self) -> Path:
        """Archive the current study attempt and reopen the question gate."""
        restart_request: dict[str, Any] = {}
        if self.new_question_path.exists():
            loaded = json.loads(self.new_question_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                restart_request = loaded
        attempts_dir = self.server_dir / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir = attempts_dir / f"attempt-{len(list(attempts_dir.glob('attempt-*'))) + 1:03d}"
        attempt_dir.mkdir()
        artifacts = (
            self.request_path,
            self.proposal_path,
            self.feasibility_path,
            self.approval_path,
            self.approved_contract_path,
            self.state_path,
            self.server_dir / "server_aggregate.json",
            self.server_dir / "codex_planning",
            self.server_dir / "report",
        )
        for path in artifacts:
            if path.exists():
                shutil.move(str(path), str(attempt_dir / path.name))
        self.new_question_path.unlink(missing_ok=True)
        next_request = restart_request.get("next_request")
        if isinstance(next_request, dict):
            self._write(self.request_path, next_request)
        self.record_event(
            "new_question",
            (
                "Current proposal was archived; revision guidance was submitted for replanning."
                if isinstance(next_request, dict)
                else "Current proposal was archived; the workflow is ready for a new research question."
            ),
            actor="researcher",
            status="completed",
        )
        return attempt_dir

    @staticmethod
    def _write(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)


def create_human_decision(
    proposal_path: Path,
    *,
    decision: str,
    approved_by: str,
    reason: str | None = None,
) -> dict[str, Any]:
    contract = AnalysisContract.parse(json.loads(proposal_path.read_text(encoding="utf-8")))
    feasibility_path = proposal_path.with_name("feasibility_report.json")
    if not feasibility_path.exists():
        raise FileNotFoundError("The client feasibility and proposal report is unavailable")
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be 'approve' or 'reject'")
    return {
        "schema_version": "biobank.human_approval.v1",
        "decision": decision,
        "proposal_digest": contract.approval_digest,
        "review_bundle_digest": _review_bundle_digest(contract.payload, feasibility),
        "approved_by": approved_by,
        "decided_at": _timestamp(),
        "reason": reason,
    }


def _review_bundle_digest(proposal: dict[str, Any], feasibility: dict[str, Any]) -> str:
    """Bind the human decision to the server contract and client-agent proposals."""
    payload = {"proposal": proposal, "feasibility": feasibility}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def create_user_request(
    question: str,
    *,
    submitted_by: str,
    catalog_metadata_codex_consent: bool = False,
) -> dict[str, Any]:
    normalized = question.strip()
    if not normalized:
        raise ValueError("question must not be empty")
    return {
        "schema_version": "biobank.user_request.v1",
        "question": normalized,
        "submitted_by": submitted_by.strip() or "researcher",
        "submitted_at": _timestamp(),
        "catalog_metadata_codex_consent": catalog_metadata_codex_consent is True,
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
