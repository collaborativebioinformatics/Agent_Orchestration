"""Local web console for initial questions and digest-bound study approval."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from biobank_agent.flare.approval import HumanApprovalGate, create_human_decision, create_user_request

MAX_REQUEST_BYTES = 65_536


class StudyConsoleStore:
    def __init__(
        self,
        run_dir: Path,
        *,
        sites_root: Path | None = None,
        workspace_root: Path | None = None,
        output_root: Path | None = None,
    ) -> None:
        self.gate = HumanApprovalGate(run_dir)
        self.sites_root = sites_root
        self.workspace_root = workspace_root
        self.output_root = output_root or run_dir.parent

    def snapshot(self) -> dict[str, Any]:
        events = self._read(self.gate.events_path, [])
        participants = self._read(self.gate.participants_path, {"clients": []})
        feasibility = self._read(self.gate.feasibility_path)
        if isinstance(feasibility, dict):
            review = feasibility.get("server_agent_review")
            if isinstance(review, dict):
                public_review = dict(review)
                public_review.pop("full_revision_guidance", None)
                feasibility = dict(feasibility)
                feasibility["server_agent_review"] = public_review
        connections = {
            client_id: {"site_id": client_id, "connected": False, "updated_at": None}
            for client_id in participants.get("clients", [])
        }
        for event in events:
            site_id = event.get("site_id")
            if site_id not in connections:
                continue
            if event.get("phase") == "client_connected":
                connections[site_id] = {
                    "site_id": site_id,
                    "connected": True,
                    "updated_at": event.get("timestamp"),
                }
            elif event.get("phase") == "client_disconnected":
                connections[site_id] = {
                    "site_id": site_id,
                    "connected": False,
                    "updated_at": event.get("timestamp"),
                }
        state = self._read(self.gate.state_path, {"status": "STARTING"})
        return {
            "state": state,
            "request": self._read(self.gate.request_path),
            "proposal": self._read(self.gate.proposal_path),
            "feasibility": feasibility,
            "tool_registry": self._read(self.gate.tool_registry_path),
            "decision_recorded": self.gate.approval_path.exists(),
            "approved_contract_available": self.gate.approved_contract_path.exists(),
            "events": events,
            "clients": list(connections.values()),
            "final_result": self._final_result() if state.get("status") == "COMPLETED" else None,
            "can_start_new_session": self.sites_root is not None and self.workspace_root is not None,
        }

    def _final_result(self) -> dict[str, Any] | None:
        result_root = self.gate.server_dir
        candidates = [result_root / "report", result_root / "recovered-report", result_root]
        result_dir = next(
            (path for path in candidates if path.is_dir() and (path / "report.md").exists() and list(path.glob("*.svg"))),
            next((path for path in candidates if (path / "report.md").exists()), None),
        )
        if result_dir is None:
            return None
        aggregate_path = result_root / "server_aggregate.json"
        recovered_aggregate = result_dir / "server_aggregate.json"
        if recovered_aggregate.exists():
            aggregate_path = recovered_aggregate
        artifacts = sorted(path.name for path in result_dir.iterdir() if path.suffix.lower() in {".svg", ".png"})
        return {
            "report": (result_dir / "report.md").read_text(encoding="utf-8"),
            "aggregate": self._read(aggregate_path),
            "artifacts": artifacts,
            "artifact_base": result_dir.name,
        }

    def result_artifact(self, directory_name: str, filename: str) -> Path:
        if Path(directory_name).name != directory_name or Path(filename).name != filename:
            raise FileNotFoundError("invalid artifact path")
        allowed_directories = {"report", "recovered-report", self.gate.server_dir.name}
        if directory_name not in allowed_directories:
            raise FileNotFoundError("unknown result directory")
        directory = self.gate.server_dir if directory_name == self.gate.server_dir.name else self.gate.server_dir / directory_name
        path = directory / filename
        if path.suffix.lower() not in {".svg", ".png", ".md"} or not path.is_file():
            raise FileNotFoundError("unknown result artifact")
        return path

    def submit_question(self, question: str, submitted_by: str, catalog_metadata_codex_consent: bool) -> dict[str, Any]:
        state = self._read(self.gate.state_path, {})
        if state.get("status") != "WAITING_FOR_QUESTION":
            raise RuntimeError(f"Workflow is not accepting a question (status={state.get('status')})")
        if self.gate.request_path.exists():
            raise RuntimeError("A research question has already been submitted")
        if catalog_metadata_codex_consent is not True:
            raise ValueError("Consent is required to send schema and mapping metadata to Codex for planning")
        request = create_user_request(
            question,
            submitted_by=submitted_by,
            catalog_metadata_codex_consent=catalog_metadata_codex_consent,
        )
        HumanApprovalGate._write(self.gate.request_path, request)
        return request

    def decide(self, decision: str, approved_by: str, confirmation: str, reason: str | None) -> dict[str, Any]:
        state = self._read(self.gate.state_path, {})
        if state.get("status") != "WAITING_FOR_APPROVAL":
            raise RuntimeError(f"Workflow is not accepting a decision (status={state.get('status')})")
        expected = "APPROVE" if decision == "approve" else "REJECT"
        if confirmation.strip() != expected:
            raise ValueError(f"Type {expected} to confirm this decision")
        if self.gate.approval_path.exists():
            raise RuntimeError("A human decision has already been recorded")
        payload = create_human_decision(
            self.gate.proposal_path,
            decision=decision,
            approved_by=approved_by,
            reason=reason,
        )
        HumanApprovalGate._write(self.gate.approval_path, payload)
        return payload

    def start_new_question(self, requested_by: str) -> dict[str, Any]:
        state = self._read(self.gate.state_path, {})
        if state.get("status") != "WAITING_FOR_APPROVAL":
            raise RuntimeError(f"A new question can be started only during review (status={state.get('status')})")
        if self.gate.approval_path.exists():
            raise RuntimeError("A human decision has already been recorded")
        request = {
            "schema_version": "biobank.new_question_request.v1",
            "requested_by": requested_by.strip() or "researcher",
        }
        HumanApprovalGate._write(self.gate.new_question_path, request)
        decision = create_human_decision(
            self.gate.proposal_path,
            decision="reject",
            approved_by=request["requested_by"],
            reason="Researcher requested a new question",
        )
        HumanApprovalGate._write(self.gate.approval_path, decision)
        return request

    def start_new_session(self, requested_by: str) -> dict[str, Any]:
        state = self._read(self.gate.state_path, {})
        if state.get("status") != "COMPLETED":
            raise RuntimeError(f"A new session can be started only after completion (status={state.get('status')})")
        if self.sites_root is None or self.workspace_root is None:
            raise RuntimeError("This console was not configured with the data-site and NVFlare workspace locations")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        session_id = f"ui-{stamp}-{secrets.token_hex(2)}"
        run_dir = self.output_root / session_id
        next_gate = HumanApprovalGate(run_dir)
        next_gate.set_state(
            "STARTING",
            requested_by=requested_by.strip() or "researcher",
            patient_data_accessed=False,
        )
        log_path = next_gate.server_dir / "session_launch.log"
        command = [
            sys.executable,
            "-m",
            "biobank_agent",
            "run-simulation",
            "--sites-root",
            str(self.sites_root),
            "--output",
            str(self.output_root),
            "--workspace",
            str(self.workspace_root),
            "--session-id",
            session_id,
        ]
        with log_path.open("ab") as stream:
            subprocess.Popen(
                command,
                cwd=Path.cwd(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.gate = next_gate
        return {"schema_version": "biobank.new_session.v1", "session_id": session_id}

    def revise_proposal(self, recommendation_digest: str, requested_by: str) -> dict[str, Any]:
        state = self._read(self.gate.state_path, {})
        if state.get("status") != "WAITING_FOR_APPROVAL":
            raise RuntimeError(f"Revision guidance is accepted only during review (status={state.get('status')})")
        if self.gate.approval_path.exists():
            raise RuntimeError("A human decision has already been recorded")
        feasibility = self._read(self.gate.feasibility_path, {})
        review = feasibility.get("server_agent_review", {}) if isinstance(feasibility, dict) else {}
        if not isinstance(review, dict) or review.get("action") != "revise":
            raise RuntimeError("The server agent has not recommended a revision")
        guidance = str(review.get("full_revision_guidance", "")).strip()
        expected_digest = hashlib.sha256(guidance.encode()).hexdigest()
        if not guidance or recommendation_digest != expected_digest or review.get("revision_digest") != expected_digest:
            raise ValueError("The server-agent revision recommendation is missing or stale")
        current = self._read(self.gate.request_path, {})
        if current.get("catalog_metadata_codex_consent") is not True:
            raise RuntimeError("The original request did not authorize catalog metadata for Codex replanning")
        original_question = str(current.get("question", "")).strip()
        if not original_question:
            raise RuntimeError("The original research question is unavailable")
        next_request = create_user_request(
            f"{original_question}\n\nResearcher-confirmed server-agent revision:\n{guidance}",
            submitted_by=requested_by,
            catalog_metadata_codex_consent=True,
        )
        next_request["revision_guidance"] = guidance
        next_request["revision_recommendation_digest"] = expected_digest
        request = {
            "schema_version": "biobank.new_question_request.v1",
            "action": "revise_proposal",
            "requested_by": requested_by.strip() or "researcher",
            "next_request": next_request,
        }
        HumanApprovalGate._write(self.gate.new_question_path, request)
        decision = create_human_decision(
            self.gate.proposal_path,
            decision="reject",
            approved_by=request["requested_by"],
            reason="Researcher requested proposal revision with additional guidance",
        )
        HumanApprovalGate._write(self.gate.approval_path, decision)
        return {"action": request["action"], "revision_recommendation_digest": expected_digest}

    @staticmethod
    def _read(path: Path, default: Any = None) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))


def serve(
    run_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    sites_root: Path | None = None,
    workspace_root: Path | None = None,
    output_root: Path | None = None,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The study console may bind only to a loopback address")
    store = StudyConsoleStore(
        run_dir,
        sites_root=sites_root,
        workspace_root=workspace_root,
        output_root=output_root,
    )
    csrf_token = secrets.token_urlsafe(24)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlparse(self.path).path
            if path == "/":
                self._send(HTTPStatus.OK, HTML.replace("__CSRF_TOKEN__", csrf_token), "text/html; charset=utf-8")
            elif path == "/api/study":
                self._json(HTTPStatus.OK, store.snapshot())
            elif path.startswith("/results/"):
                parts = path.split("/")
                try:
                    artifact = store.result_artifact(parts[2], parts[3]) if len(parts) == 4 else None
                    if artifact is None:
                        raise FileNotFoundError("invalid artifact path")
                    content_type = mimetypes.guess_type(artifact.name)[0] or "application/octet-stream"
                    self._send_bytes(HTTPStatus.OK, artifact.read_bytes(), content_type)
                except FileNotFoundError:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "result artifact not found"})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.headers.get("X-Study-Token") != csrf_token:
                self._json(HTTPStatus.FORBIDDEN, {"error": "invalid study-console token"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("invalid request size")
                body = json.loads(self.rfile.read(length))
                path = urlparse(self.path).path
                if path == "/api/question":
                    result = store.submit_question(
                        str(body.get("question", "")),
                        str(body.get("submitted_by", "")),
                        body.get("catalog_metadata_codex_consent") is True,
                    )
                elif path == "/api/decision":
                    result = store.decide(
                        str(body.get("decision", "")),
                        str(body.get("approved_by", "")),
                        str(body.get("confirmation", "")),
                        str(body["reason"]) if body.get("reason") else None,
                    )
                elif path == "/api/new-question":
                    result = store.start_new_question(str(body.get("requested_by", "")))
                elif path == "/api/start-new-session":
                    result = store.start_new_session(str(body.get("requested_by", "")))
                elif path == "/api/revise":
                    result = store.revise_proposal(
                        str(body.get("recommendation_digest", "")),
                        str(body.get("requested_by", "")),
                    )
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                self._json(HTTPStatus.OK, result)
            except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.CONFLICT, {"error": str(exc)})

        def _json(self, status: HTTPStatus, payload: Any) -> None:
            self._send(status, json.dumps(payload), "application/json; charset=utf-8")

        def _send(self, status: HTTPStatus, body: str, content_type: str) -> None:
            self._send_bytes(status, body.encode("utf-8"), content_type)

        def _send_bytes(self, status: HTTPStatus, encoded: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Study console: http://{host}:{port}")
    print(f"Run directory: {run_dir.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agentic Federated Analysis on Biobanks</title>
<style>
:root{color-scheme:dark;--bg:#090b0c;--panel:#151819;--line:#303536;--text:#f4f6f6;--muted:#a7afaf;--green:#76b900;--teal:#22d3c5;--warn:#f4b942;--bad:#ff7066}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 85% 0,#18251d 0,transparent 34%),var(--bg);color:var(--text);font:15px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}.shell{max-width:1440px;margin:auto;padding:36px 24px 80px}header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:28px}h1{font-size:30px;margin:0 0 6px}h2{font-size:20px;margin:0 0 14px}h3{font-size:15px;margin:0 0 8px}.muted{color:var(--muted)}.badge{border:1px solid var(--line);border-radius:999px;padding:7px 12px;font:600 12px ui-monospace,monospace;letter-spacing:.04em}.workspace-layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(320px,1fr);gap:16px;align-items:start}.primary,.sidebar{display:grid;gap:16px;min-width:0}.sidebar{position:sticky;top:16px;max-height:calc(100vh - 32px);overflow:auto;padding-bottom:2px}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}.card{grid-column:span 12;background:linear-gradient(145deg,#171a1b,#121415);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 16px 45px #0005}.half{grid-column:span 6}.third{grid-column:span 4}textarea,input{width:100%;border:1px solid #3a4141;border-radius:9px;background:#0c0e0f;color:var(--text);padding:12px;font:inherit}input[type=checkbox]{width:auto;accent-color:var(--green);margin:0 9px 0 0}.consent{display:flex;align-items:flex-start;color:var(--text);border:1px solid var(--line);padding:12px;border-radius:9px;background:#101213}textarea{min-height:150px;resize:vertical}label{display:block;margin:12px 0 6px;color:var(--muted);font-size:13px}button{border:0;border-radius:9px;padding:11px 16px;font-weight:700;cursor:pointer;background:var(--green);color:#0a1100}button.secondary{background:#343a3b;color:white}button.danger{background:var(--bad);color:#180000}button:disabled{opacity:.4;cursor:not-allowed}.actions{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap}.tools,.cohorts,.unavailable{display:grid;gap:10px}.item{border:1px solid var(--line);border-radius:9px;padding:12px;background:#101213}.item strong{color:var(--teal)}.clients{display:grid;grid-template-columns:1fr;gap:10px}.client{border:1px solid var(--line);border-radius:9px;padding:12px;background:#101213}.client-head{display:flex;align-items:center;gap:9px;font-weight:650}.presence{width:10px;height:10px;border-radius:50%;background:#606869;box-shadow:0 0 0 3px #252a2b}.client.connected .presence{background:var(--green);box-shadow:0 0 0 3px #263b10}.client-state{margin-top:6px;color:var(--muted);font:12px ui-monospace,monospace}.timeline{display:grid;gap:0}.event{display:grid;grid-template-columns:72px 14px 1fr;gap:10px;min-height:62px}.event-time{color:var(--muted);font:12px ui-monospace,monospace;padding-top:2px}.event-rail{position:relative}.event-rail:before{content:'';position:absolute;left:6px;top:13px;bottom:-10px;border-left:1px solid #3a4141}.event:last-child .event-rail:before{display:none}.event-dot{position:absolute;top:5px;left:1px;width:11px;height:11px;border-radius:50%;background:var(--teal);box-shadow:0 0 0 3px #153532}.event.failed .event-dot{background:var(--bad);box-shadow:0 0 0 3px #3d1917}.event.waiting .event-dot{background:var(--warn);box-shadow:0 0 0 3px #413515}.event.completed .event-dot{background:var(--green);box-shadow:0 0 0 3px #263b10}.event-title{font-weight:650}.event-meta{font:12px ui-monospace,monospace;color:var(--muted);margin-top:3px}.privacy-flag{color:var(--warn)}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:9px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-size:12px;text-transform:uppercase}.ok{color:var(--green)}.no{color:var(--bad)}.hidden{display:none!important}.notice{border-left:3px solid var(--warn);padding:10px 13px;background:#241f12;color:#f6dfaa;border-radius:4px}.error{border-left-color:var(--bad);background:#281412;color:#ffd1cd}code{color:var(--teal)}@media(max-width:920px){.workspace-layout{grid-template-columns:1fr}.sidebar{position:static;max-height:none;overflow:visible}.half,.third{grid-column:span 12}header{display:block}.badge{display:inline-block;margin-top:12px}}
.timeline-wide{margin-bottom:16px}.timeline-wide .timeline{grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.timeline-wide .event{border:1px solid var(--line);border-radius:9px;padding:10px;min-height:76px;background:#101213}.timeline-wide .event-rail:before{display:none}@media(max-width:920px){.timeline-wide .timeline{grid-template-columns:1fr}}
.primary>.card,.sidebar>.card{grid-column:auto;min-width:0}.sidebar>.card{width:100%;overflow:hidden}.client-head span:last-child{min-width:0;overflow-wrap:anywhere;word-break:break-word}.workspace-layout>*{min-width:0}
.result-figures{display:grid;gap:16px}.result-figure{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#0b0d0e}.result-figure img{display:block;width:100%;height:auto}.result-summary{white-space:pre-wrap;border:1px solid var(--line);border-radius:9px;padding:14px;background:#101213;color:var(--muted);max-height:320px;overflow:auto}.result-links a{color:var(--teal)}
</style></head><body><div class="shell"><header><div><h1>Agentic Federated Analysis on Biobanks</h1><div class="muted">One console from research question to reviewed analysis contract.</div></div><div id="status" class="badge">STARTING</div></header><div id="message" class="notice hidden"></div>
<section id="timeline-card" class="card timeline-wide"><h2>Live execution timeline</h2><p class="muted">Client entries originate from NVFlare auxiliary messages. This feed contains status only—never patient values.</p><div id="timeline" class="timeline"><div class="muted">Waiting for events…</div></div></section>
<main class="workspace-layout"><div class="primary"><section id="question-card" class="card hidden"><h2>1. Research question</h2><p class="muted">Describe the scientific question. The agent will inspect schema metadata, select general-purpose tools, and return a proposal for review. No patient rows are read at this stage.</p><label for="researcher">Researcher</label><input id="researcher" value="Cecilie"><label for="question">Question</label><textarea id="question" placeholder="What would you like to compare or estimate across the participating sites?"></textarea><label class="consent"><input id="catalog-consent" type="checkbox"><span>I authorize sending site schema names and declared harmonization mappings to the signed-in Codex backend for contract planning. No patient rows or values are included.</span></label><div class="actions"><button id="submit-question">Submit question</button></div></section>
<section id="waiting-card" class="card hidden"><h2>Workflow progress</h2><p id="waiting-text" class="muted"></p></section>
<section id="study-summary" class="card hidden"><h2>Study summary</h2><p id="study-total" class="muted"></p><div style="overflow:auto"><table><thead><tr><th>Site</th><th>Rows used</th><th>Total local rows</th><th>Included</th><th>Client-agent reason</th></tr></thead><tbody id="study-sites"></tbody></table></div></section>
<section id="final-result" class="card hidden"><h2>Final Result</h2><p class="muted">Disclosure-controlled aggregates returned by the participating sites. No patient-level rows were received by the server.</p><div id="result-figures" class="result-figures"></div><h3 style="margin-top:18px">Report</h3><div id="result-summary" class="result-summary"></div><div id="result-links" class="result-links actions"></div><div class="actions"><button id="start-new-completed">Start New</button></div></section>
<section id="review" class="card hidden"><h2>2. Review proposed analysis</h2><p id="original-question"></p><div class="grid"><div class="half"><h3>Cohorts</h3><div id="cohorts" class="cohorts"></div></div><div class="half"><h3>Privacy</h3><div id="privacy" class="item"></div></div><div class="card" style="grid-column:span 12;padding:14px"><h3>Site harmonization</h3><div id="harmonization" class="tools"></div></div><div class="card" style="grid-column:span 12;padding:14px"><h3>Selected tools and analyses</h3><div id="tools" class="tools"></div></div><div class="card" style="grid-column:span 12;padding:14px"><h3>Site feasibility</h3><div style="overflow:auto"><table><thead><tr><th>Site</th><th>Adapter</th><th>Supported</th><th>Unsupported</th><th>Status</th></tr></thead><tbody id="sites"></tbody></table></div></div><div id="unavailable-wrap" class="card hidden" style="grid-column:span 12;padding:14px"><h3>Unsupported requests</h3><div id="unavailable" class="unavailable"></div></div></div></section>
<section id="server-review" class="card hidden"><h2>Server-agent recommendation</h2><p id="server-review-summary"></p><h3>Key points to confirm</h3><div id="revision-key-points" class="tools"></div></section>
<section id="decision" class="card hidden"><h2>3. Human decision</h2><p id="decision-notice" class="notice">Review the server agent's concise recommendation before continuing.</p><label for="approver">Decision by</label><input id="approver" value="Cecilie"><div class="actions"><button id="revise" class="secondary">Confirm revision and replan</button><button id="new-question" class="secondary">Start new question</button></div><div id="approval-controls"><label for="reason">Reason or note</label><input id="reason" placeholder="Optional for approval; recommended for rejection"><label for="confirmation">Confirmation</label><input id="confirmation" placeholder="Type APPROVE or REJECT"><div class="actions"><button id="approve">Approve contract</button><button id="reject" class="danger">Reject contract</button></div></div></section></div>
<aside class="sidebar"><section id="clients-card" class="card"><h2>Connected clients <span id="client-count" class="muted"></span></h2><div id="clients" class="clients"><div class="muted">Waiting for client presence events…</div></div></section></aside></main></div>
<script>
const TOKEN='__CSRF_TOKEN__';const $=id=>document.getElementById(id);const esc=v=>String(v??'');let currentAgentReview={};
function show(id,on=true){$(id).classList.toggle('hidden',!on)}function message(text,bad=false){$('message').textContent=text;show('message',!!text);$('message').classList.toggle('error',bad)}
function predicate(p){if(!p)return '—';const [op,v]=Object.entries(p)[0];if(op==='all'||op==='any')return '('+v.map(predicate).join(op==='all'?' AND ':' OR ')+')';if(op==='not')return 'NOT '+predicate(v);return `${v.field} ${op} ${v.value??(v.values||[]).join(', ')}`}
async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Study-Token':TOKEN},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.error||'Request failed');return data}
function renderFinal(result,canStart){const ready=!!result;show('study-summary',ready);show('final-result',ready);$('start-new-completed').disabled=!canStart;if(!ready)return;const aggregate=result.aggregate||{},sites=aggregate.site_row_summaries||[];let total=0,used=0,complete=true;const rows=sites.map(site=>{const counts=Object.values(site.analysis_row_counts||{});const chosen=counts.find(x=>Number.isFinite(x.rows_used));const siteUsed=chosen?.rows_used,siteTotal=site.total_rows;if(!Number.isFinite(siteUsed)||!Number.isFinite(siteTotal))complete=false;else{used+=siteUsed;total+=siteTotal}const tr=document.createElement('tr');const included=Number.isFinite(siteUsed)&&Number.isFinite(siteTotal)&&siteTotal?`${(100*siteUsed/siteTotal).toFixed(1)}%`:'Not available';[site.site_id,siteUsed??'Suppressed',siteTotal??'Suppressed',included,site.exclusion_reason||'Client-agent explanation unavailable for this run.'].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});return tr});$('study-sites').replaceChildren(...rows);$('study-total').textContent=complete&&sites.length?`${used.toLocaleString()} of ${total.toLocaleString()} local rows were included across ${sites.length} sites (${(100*used/total).toFixed(1)}%). Counts are site-produced aggregates.`:'Local row-use counts were unavailable or disclosure-suppressed for this result.';const base=encodeURIComponent(result.artifact_base);const artifacts=result.artifacts||[];$('result-figures').replaceChildren(...artifacts.map(name=>{const figure=document.createElement('figure');figure.className='result-figure';const img=document.createElement('img');img.src=`/results/${base}/${encodeURIComponent(name)}`;img.alt=name.replace(/[_-]+/g,' ').replace(/\.[^.]+$/,'');figure.append(img);return figure}));$('result-summary').textContent=result.report||'The aggregate report is unavailable.';const report=document.createElement('a');report.href=`/results/${base}/report.md`;report.target='_blank';report.rel='noopener';report.textContent='Open report';$('result-links').replaceChildren(report)}
function render(s){const state=s.state||{};const status=state.status||'STARTING';$('status').textContent=status;show('question-card',status==='WAITING_FOR_QUESTION');show('waiting-card',['STARTING','DISCOVERING','PLANNING','APPROVED','DISPATCHING_ANALYSIS'].includes(status));$('waiting-text').textContent={STARTING:'Waiting for the NVFlare Controller.',DISCOVERING:'Sites are returning schema and mapping metadata; no row values are being read.',PLANNING:'Codex is translating the question into a typed contract using the registered tools.',APPROVED:'Approval validated. Preparing dispatch.',DISPATCHING_ANALYSIS:'Approved analysis is executing behind each site boundary.'}[status]||'';
const clients=s.clients||[];const connected=clients.filter(c=>c.connected).length;$('client-count').textContent=`${connected}/${clients.length}`;$('clients').replaceChildren(...(clients.length?clients.map(c=>{const d=document.createElement('div');d.className=`client ${c.connected?'connected':''}`;const head=document.createElement('div');head.className='client-head';const dot=document.createElement('span');dot.className='presence';const name=document.createElement('span');name.textContent=c.site_id;head.append(dot,name);const detail=document.createElement('div');detail.className='client-state';const seen=c.updated_at?new Date(c.updated_at).toLocaleTimeString():'not seen';detail.textContent=`${c.connected?'CONNECTED':'DISCONNECTED'} · ${seen}`;d.append(head,detail);return d}):[Object.assign(document.createElement('div'),{className:'muted',textContent:'Waiting for client presence events…'})]));
const events=(s.events||[]).slice(-3);$('timeline').replaceChildren(...(events.length?events.map(e=>{const row=document.createElement('div');row.className=`event ${e.status||''}`;const time=document.createElement('div');time.className='event-time';const d=new Date(e.timestamp);time.textContent=Number.isNaN(d.getTime())?'':d.toLocaleTimeString();const rail=document.createElement('div');rail.className='event-rail';const dot=document.createElement('span');dot.className='event-dot';rail.append(dot);const body=document.createElement('div');const title=document.createElement('div');title.className='event-title';title.textContent=e.message;const meta=document.createElement('div');meta.className='event-meta';meta.textContent=[e.actor,e.site_id,e.phase,e.patient_data_accessed?'local patient data accessed':'no patient data accessed'].filter(Boolean).join(' · ');if(e.patient_data_accessed)meta.classList.add('privacy-flag');body.append(title,meta);row.append(time,rail,body);return row}):[Object.assign(document.createElement('div'),{className:'muted',textContent:'Waiting for events…'})]));
if(['REJECTED','EXPIRED','ABORTED','FAILED'].includes(status))message(state.reason||`Workflow ${status.toLowerCase()}.`,true);else if(status==='COMPLETED')message('Study completed. Final aggregate results are ready.');
const completed=status==='COMPLETED';renderFinal(completed?s.final_result:null,s.can_start_new_session===true);const review=!completed&&(status==='WAITING_FOR_APPROVAL'||s.proposal);show('review',!!review);show('server-review',false);show('decision',false);if(completed||!s.proposal)return;const p=s.proposal,f=s.feasibility||{},registry=(s.tool_registry||{}).tools||{},agentReview=f.server_agent_review||{},recommendationReady=!!agentReview.summary&&(agentReview.confirmation_items||[]).length>0;currentAgentReview=agentReview;show('server-review',recommendationReady);show('decision',status==='WAITING_FOR_APPROVAL'&&!s.decision_recorded&&recommendationReady);if(status==='WAITING_FOR_APPROVAL'&&!recommendationReady){show('waiting-card',true);$('waiting-text').textContent='The server agent is synthesizing the completed client responses into a concise recommendation.'}$('original-question').textContent=p.question;
const executable=(p.analyses||[]).length>0;$('approve').disabled=!executable;show('revise',agentReview.action==='revise');show('approval-controls',agentReview.action!=='revise');$('decision-notice').textContent=agentReview.action==='revise'?'Confirming sends the server agent\'s complete digest-bound revision instructions to all site agents and replans.':executable?'Approval authorizes the exact reviewed contract. The complete client/server review bundle is digest-bound.':'No executable analysis is available and no server-agent revision has been prepared.';
$('cohorts').replaceChildren(...(p.cohorts||[]).map(c=>{const d=document.createElement('div');d.className='item';const b=document.createElement('strong');b.textContent=c.name;const q=document.createElement('div');q.className='muted';q.textContent=predicate(c.predicate);d.append(b,q);return d}));
$('privacy').textContent=`Minimum cell count: ${p.privacy?.min_cell_count}. Forbidden outputs: ${(p.privacy?.forbidden_outputs||[]).join(', ')}. Training allowed: ${p.training_allowed}.`;
$('harmonization').replaceChildren(...Object.entries(p.site_harmonization||{}).map(([site,config])=>{const d=document.createElement('div');d.className='item';const b=document.createElement('strong');b.textContent=site;const q=document.createElement('div');q.className='muted';q.textContent=Object.entries(config.fields||{}).map(([canonical,spec])=>`${canonical} ← ${spec.source||canonical}${spec.multiply!==undefined?` × ${spec.multiply}`:''}${spec.value_map?' · value map':''}`).join(' | ')||'No transformations';d.append(b,q);return d}));
$('tools').replaceChildren(...(p.analyses||[]).map(a=>{const d=document.createElement('div');d.className='item';const b=document.createElement('strong');b.textContent=`${a.analysis_id||'analysis'} · ${a.tool}`;const q=document.createElement('div');q.textContent=registry[a.tool]?.description||'';const pre=document.createElement('div');pre.className='muted';pre.textContent=Object.entries(a).filter(([k])=>!['analysis_id','tool'].includes(k)).map(([k,v])=>`${k}: ${JSON.stringify(v)}`).join(' · ');d.append(b,q,pre);return d}));
$('server-review-summary').textContent=agentReview.summary||'';$('revision-key-points').replaceChildren(...(agentReview.confirmation_items||[]).map((text,index)=>{const d=document.createElement('div');d.className='item';const b=document.createElement('strong');b.textContent=`${index+1}`;const span=document.createElement('span');span.textContent=`  ${text}`;d.append(b,span);return d}));
$('sites').replaceChildren(...(f.sites||[]).map(site=>{const tr=document.createElement('tr');const adapter=site.data_adapter||{};const adapterText=adapter.status?`${adapter.status} · ${(adapter.digest||'').slice(0,10)}`:'unavailable';[site.site_id,adapterText,site.supported?.length||0,site.unsupported?.map(x=>x.analysis_id).join(', ')||'0',site.all_requested_analyses_supported?'Supported':'Partial'].forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;if(i===4)td.className=site.all_requested_analyses_supported?'ok':'no';tr.append(td)});return tr}));
const unavailable=p.unavailable_requests||[];show('unavailable-wrap',unavailable.length>0);$('unavailable').replaceChildren(...unavailable.map(x=>{const d=document.createElement('div');d.className='item';d.textContent=`${x.concept}: ${x.reason}`;return d}));}
async function refresh(){try{const r=await fetch('/api/study',{cache:'no-store'});render(await r.json())}catch(e){message(e.message,true)}}
$('submit-question').onclick=async()=>{try{await post('/api/question',{question:$('question').value,submitted_by:$('researcher').value,catalog_metadata_codex_consent:$('catalog-consent').checked});message('Question submitted. The Controller will begin schema discovery.');refresh()}catch(e){message(e.message,true)}};
async function decide(decision){try{await post('/api/decision',{decision,approved_by:$('approver').value,confirmation:$('confirmation').value,reason:$('reason').value});message(`Decision recorded: ${decision}.`);refresh()}catch(e){message(e.message,true)}}$('approve').onclick=()=>decide('approve');$('reject').onclick=()=>decide('reject');
$('new-question').onclick=async()=>{if(!window.confirm('Archive this proposal and start a new research question?'))return;try{await post('/api/new-question',{requested_by:$('approver').value});message('Current proposal archived. Reopening the question form…');refresh()}catch(e){message(e.message,true)}};
$('start-new-completed').onclick=async()=>{if(!window.confirm('Start a fresh federated study session? The completed result will remain saved.'))return;try{const result=await post('/api/start-new-session',{requested_by:'Cecilie'});message(`Started ${result.session_id}. Waiting for the new Controller…`);refresh()}catch(e){message(e.message,true)}};
$('revise').onclick=async()=>{if(!window.confirm('Confirm these revision points and ask the server agent to replan?'))return;try{await post('/api/revise',{recommendation_digest:currentAgentReview.revision_digest,requested_by:$('approver').value});message('Revision confirmed. The complete server-agent instructions were submitted and planning will rerun…');refresh()}catch(e){message(e.message,true)}};refresh();setInterval(refresh,1000);
</script></body></html>'''
