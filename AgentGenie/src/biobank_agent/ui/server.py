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
    ACTIVE_SESSION_SCHEMA = "biobank.active_session.v1"

    def __init__(
        self,
        run_dir: Path,
        *,
        sites_root: Path | None = None,
        workspace_root: Path | None = None,
        output_root: Path | None = None,
    ) -> None:
        self.output_root = (output_root or run_dir.parent).resolve()
        active_run_dir = self._active_run_dir(run_dir.resolve())
        self.gate = HumanApprovalGate(active_run_dir)
        self.sites_root = sites_root
        self.workspace_root = workspace_root

    @property
    def active_session_path(self) -> Path:
        return self.output_root / ".active_session.json"

    def _active_run_dir(self, fallback: Path) -> Path:
        pointer = self._read(self.active_session_path, {})
        if not isinstance(pointer, dict):
            return fallback
        session_id = pointer.get("session_id")
        if pointer.get("schema_version") != self.ACTIVE_SESSION_SCHEMA or not isinstance(session_id, str):
            return fallback
        candidate = (self.output_root / session_id).resolve()
        if candidate.parent != self.output_root or not candidate.is_dir():
            return fallback
        return candidate

    def _remember_active_session(self, run_dir: Path) -> None:
        HumanApprovalGate._write(
            self.active_session_path,
            {
                "schema_version": self.ACTIVE_SESSION_SCHEMA,
                "session_id": run_dir.name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

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
            "session_id": self.gate.run_dir.name,
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
            "benchmark_bundle_available": (result_root / "benchmark_bundle.zip").is_file(),
        }

    def benchmark_bundle(self) -> Path:
        path = self.gate.server_dir / "benchmark_bundle.zip"
        if not path.is_file():
            raise FileNotFoundError("benchmark bundle is unavailable")
        return path

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
        self._remember_active_session(run_dir)
        self.gate = next_gate
        return {"schema_version": "biobank.new_session.v1", "session_id": session_id}

    def revise_proposal(
        self,
        recommendation_digest: str,
        requested_by: str,
        manual_guidance: str = "",
    ) -> dict[str, Any]:
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
        manual_guidance = manual_guidance.strip()
        if len(manual_guidance) > 4_000:
            raise ValueError("Additional revision guidance must be 4,000 characters or fewer")
        revision_sections = [f"Researcher-confirmed server-agent revision:\n{guidance}"]
        if manual_guidance:
            revision_sections.append(f"Additional guidance supplied by the researcher:\n{manual_guidance}")
        next_request = create_user_request(
            f"{original_question}\n\n" + "\n\n".join(revision_sections),
            submitted_by=requested_by,
            catalog_metadata_codex_consent=True,
        )
        next_request["revision_guidance"] = guidance
        next_request["manual_revision_guidance"] = manual_guidance
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
            elif path == "/benchmark-bundle.zip":
                try:
                    bundle = store.benchmark_bundle()
                    self._send_bytes(HTTPStatus.OK, bundle.read_bytes(), "application/zip")
                except FileNotFoundError:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "benchmark bundle not found"})
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
                        str(body.get("manual_guidance", "")),
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
:root{color-scheme:dark;--bg:#090b0c;--panel:#151819;--line:#303536;--text:#f4f6f6;--muted:#a7afaf;--green:#76b900;--teal:#22d3c5;--warn:#f4b942;--bad:#ff7066}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 85% 0,#18251d 0,transparent 34%),var(--bg);color:var(--text);font:15px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}.shell{max-width:1440px;margin:auto;padding:36px 24px 80px}header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:28px}h1{font-size:30px;margin:0 0 6px}h2{font-size:20px;margin:0 0 14px}h3{font-size:15px;margin:0 0 8px}.muted{color:var(--muted)}.badge{border:1px solid var(--line);border-radius:999px;padding:7px 12px;font:600 12px ui-monospace,monospace;letter-spacing:.04em}.workspace-layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(320px,1fr);gap:16px;align-items:start}.primary,.sidebar{display:grid;gap:16px;min-width:0}.sidebar{position:sticky;top:16px;max-height:calc(100vh - 32px);overflow:auto;padding-bottom:2px}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}.card{grid-column:span 12;background:linear-gradient(145deg,#171a1b,#121415);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 16px 45px #0005}.half{grid-column:span 6}.third{grid-column:span 4}textarea,input{width:100%;border:1px solid #3a4141;border-radius:9px;background:#0c0e0f;color:var(--text);padding:12px;font:inherit}input[type=checkbox]{width:auto;accent-color:var(--green);margin:0 9px 0 0}.consent{display:flex;align-items:flex-start;color:var(--text);border:1px solid var(--line);padding:12px;border-radius:9px;background:#101213}textarea{min-height:150px;resize:vertical}label{display:block;margin:12px 0 6px;color:var(--muted);font-size:13px}button{border:0;border-radius:9px;padding:11px 16px;font-weight:700;cursor:pointer;background:var(--green);color:#0a1100}button.secondary{background:#343a3b;color:white}button.danger{background:var(--bad);color:#180000}button:disabled{opacity:.4;cursor:not-allowed}.actions{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap}.tools,.cohorts,.unavailable{display:grid;gap:10px}.item{border:1px solid var(--line);border-radius:9px;padding:12px;background:#101213}.item strong{color:var(--teal)}.clients{display:grid;grid-template-columns:1fr;gap:10px}.client{border:1px solid var(--line);border-radius:9px;padding:12px;background:#101213}.client-head{display:flex;align-items:center;gap:9px;font-weight:650}.presence{width:10px;height:10px;border-radius:50%;background:#606869;box-shadow:0 0 0 3px #252a2b}.client.connected .presence{background:var(--green);box-shadow:0 0 0 3px #263b10}.client-state{margin-top:6px;color:var(--muted);font:12px ui-monospace,monospace}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:9px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-size:12px;text-transform:uppercase}.ok{color:var(--green)}.no{color:var(--bad)}.hidden{display:none!important}.notice{border-left:3px solid var(--warn);padding:10px 13px;background:#241f12;color:#f6dfaa;border-radius:4px}.error{border-left-color:var(--bad);background:#281412;color:#ffd1cd}code{color:var(--teal)}@media(max-width:920px){.workspace-layout{grid-template-columns:1fr}.sidebar{position:static;max-height:none;overflow:visible}.half,.third{grid-column:span 12}header{display:block}.badge{display:inline-block;margin-top:12px}}
.execution-wide{margin-bottom:16px}.graph-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:18px}.graph-legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px}.legend-working:before,.legend-connected:before{content:'';display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}.legend-working:before{background:var(--teal);box-shadow:0 0 10px var(--teal)}.legend-connected:before{background:var(--green)}.agent-graph{width:100%;overflow-x:auto;margin-top:8px}.agent-graph svg{display:block;width:100%;min-width:720px;height:280px}.graph-node rect{fill:#101314;stroke:#495052;stroke-width:1.5}.graph-node.connected rect{stroke:var(--green)}.graph-node.working rect{stroke:var(--teal);stroke-width:2.5;filter:drop-shadow(0 0 8px #22d3c588);animation:agent-pulse 1.5s ease-in-out infinite}.graph-node.done rect{stroke:var(--green);fill:#11200d}.graph-node.failed rect{stroke:var(--bad)}.graph-node .node-title{fill:var(--text);font-size:17px;font-weight:700}.graph-node.client-node .node-title{font-size:14px}.graph-node .node-phase{fill:var(--muted);font-size:12px}.graph-node .node-dot{fill:#606869}.graph-node.connected .node-dot,.graph-node.done .node-dot{fill:var(--green)}.graph-node.working .node-dot{fill:var(--teal)}.graph-node.failed .node-dot{fill:var(--bad)}.flow-line{fill:none;stroke:#485052;stroke-width:2}.flow-line.active{stroke:var(--teal);stroke-width:3;stroke-dasharray:8 7;animation:flow-move 1s linear infinite}.flow-arrow{fill:#485052}.flow-arrow.active{fill:var(--teal);filter:drop-shadow(0 0 5px #22d3c5aa)}.flow-label{fill:var(--muted);font-size:11px;text-anchor:middle}.flow-label.active{fill:var(--teal)}@keyframes agent-pulse{50%{filter:drop-shadow(0 0 15px #22d3c5cc)}}@keyframes flow-move{to{stroke-dashoffset:-15}}@media(prefers-reduced-motion:reduce){.graph-node.working rect,.flow-line.active{animation:none}}@media(max-width:920px){.graph-heading{display:block}.graph-legend{margin-top:8px}.agent-graph svg{height:250px}}
.primary>.card,.sidebar>.card{grid-column:auto;min-width:0}.sidebar>.card{width:100%;overflow:hidden}.client-head span:last-child{min-width:0;overflow-wrap:anywhere;word-break:break-word}.workspace-layout>*{min-width:0}
.result-figures{display:grid;gap:16px}.result-figure{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#0b0d0e}.result-figure img{display:block;width:100%;height:auto}.result-summary{white-space:pre-wrap;border:1px solid var(--line);border-radius:9px;padding:14px;background:#101213;color:var(--muted);max-height:320px;overflow:auto}.result-links a{color:var(--teal)}
</style></head><body><div class="shell"><header><div><h1>Agentic Federated Analysis on Biobanks</h1><div class="muted">One console from research question to reviewed analysis contract.</div></div><div id="status" class="badge">STARTING</div></header><div id="message" class="notice hidden"></div>
<section id="execution-card" class="card execution-wide"><div class="graph-heading"><div><h2>Live agent workflow</h2><p class="muted">Status-only NVFlare events illuminate the agents and direction of work; patient values never enter this view.</p></div><div class="graph-legend"><span class="legend-working">Working now</span><span class="legend-connected">Connected or complete</span></div></div><div id="agent-graph" class="agent-graph" aria-live="polite"><div class="muted">Waiting for agents…</div></div></section>
<main class="workspace-layout"><div class="primary"><section id="question-card" class="card hidden"><h2>1. Research question</h2><p class="muted">Describe the scientific question. The agent will inspect schema metadata, select general-purpose tools, and return a proposal for review. No patient rows are read at this stage.</p><label for="researcher">Researcher</label><input id="researcher" value="Cecilie"><label for="question">Question</label><textarea id="question" placeholder="What would you like to compare or estimate across the participating sites?"></textarea><label class="consent"><input id="catalog-consent" type="checkbox"><span>I authorize sending site schema names and declared harmonization mappings to the signed-in Codex backend for contract planning. No patient rows or values are included.</span></label><div class="actions"><button id="submit-question">Submit question</button></div></section>
<section id="waiting-card" class="card hidden"><h2>Workflow progress</h2><p id="waiting-text" class="muted"></p></section>
<section id="study-summary" class="card hidden"><h2>Study summary</h2><p id="study-total" class="muted"></p><div style="overflow:auto"><table><thead><tr><th>Site</th><th>Rows used</th><th>Total local rows</th><th>Included</th><th>Client-agent reason</th></tr></thead><tbody id="study-sites"></tbody></table></div></section>
<section id="final-result" class="card hidden"><h2>Final Result</h2><p class="muted">Disclosure-controlled aggregates returned by the participating sites. No patient-level rows were received by the server.</p><div id="result-figures" class="result-figures"></div><h3 style="margin-top:18px">Report</h3><div id="result-summary" class="result-summary"></div><div id="result-links" class="result-links actions"></div></section>
<section id="review" class="card hidden"><h2>2. Review proposed analysis</h2><p id="original-question"></p><div class="grid"><div class="half"><h3>Cohorts</h3><div id="cohorts" class="cohorts"></div></div><div class="half"><h3>Privacy</h3><div id="privacy" class="item"></div></div><div class="card" style="grid-column:span 12;padding:14px"><h3>Site harmonization</h3><div id="harmonization" class="tools"></div></div><div class="card" style="grid-column:span 12;padding:14px"><h3>Selected tools and analyses</h3><div id="tools" class="tools"></div></div><div class="card" style="grid-column:span 12;padding:14px"><h3>Site feasibility</h3><div style="overflow:auto"><table><thead><tr><th>Site</th><th>Adapter</th><th>Supported</th><th>Unsupported</th><th>Status</th></tr></thead><tbody id="sites"></tbody></table></div></div><div id="unavailable-wrap" class="card hidden" style="grid-column:span 12;padding:14px"><h3>Unsupported requests</h3><div id="unavailable" class="unavailable"></div></div></div></section>
<section id="server-review" class="card hidden"><h2>Server-agent recommendation</h2><p id="server-review-summary"></p><h3>Key points to confirm</h3><div id="revision-key-points" class="tools"></div></section>
<section id="decision" class="card hidden"><h2>3. Human decision</h2><p id="decision-notice" class="notice">Review the server agent's concise recommendation before continuing.</p><label for="approver">Decision by</label><input id="approver" value="Cecilie"><div id="revision-guidance-wrap" class="hidden"><label for="revision-guidance">Additional revision guidance</label><textarea id="revision-guidance" maxlength="4000" placeholder="Optional: add scientific assumptions, exclusions, endpoint definitions, grouping rules, or presentation requests for the next planning round."></textarea><p class="muted">This is appended to the server agent's complete recommendation and recorded in the next request.</p></div><div class="actions"><button id="revise" class="secondary">Confirm revision and replan</button></div><div id="approval-controls"><label for="reason">Reason or note</label><input id="reason" placeholder="Optional for approval; recommended for rejection"><label for="confirmation">Confirmation</label><input id="confirmation" placeholder="Type APPROVE or REJECT"><div class="actions"><button id="approve">Approve contract</button><button id="reject" class="danger">Reject contract</button></div></div></section></div>
<aside class="sidebar"><section id="clients-card" class="card"><h2>Connected clients <span id="client-count" class="muted"></span></h2><div id="clients" class="clients"><div class="muted">Waiting for client presence events…</div></div></section><section id="completed-actions" class="card"><h2>New study</h2><p id="start-new-help" class="muted">Archive the current attempt and return to a blank research question.</p><button id="start-new-completed">Start New</button></section></aside></main></div>
<script>
const TOKEN='__CSRF_TOKEN__';const $=id=>document.getElementById(id);const esc=v=>String(v??'');let currentAgentReview={},currentStatus='STARTING',renderedFinalKey='';
function show(id,on=true){$(id).classList.toggle('hidden',!on)}function message(text,bad=false){$('message').textContent=text;show('message',!!text);$('message').classList.toggle('error',bad)}
function predicate(p){if(!p)return '—';const [op,v]=Object.entries(p)[0];if(op==='all'||op==='any')return '('+v.map(predicate).join(op==='all'?' AND ':' OR ')+')';if(op==='not')return 'NOT '+predicate(v);return `${v.field} ${op} ${v.value??(v.values||[]).join(', ')}`}
async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Study-Token':TOKEN},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.error||'Request failed');return data}
function svgElement(tag,attrs={},text=''){const node=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([key,value])=>node.setAttribute(key,String(value)));if(text)node.textContent=text;return node}
function addGraphNode(svg,x,y,width,height,title,phase,state,isClient=false){const group=svgElement('g',{class:`graph-node ${state}${isClient?' client-node':''}`});group.append(svgElement('rect',{x,y,width,height,rx:12}),svgElement('circle',{class:'node-dot',cx:x+22,cy:y+25,r:6}),svgElement('text',{class:'node-title',x:x+38,y:y+31},title),svgElement('text',{class:'node-phase',x:x+22,y:y+58},phase));svg.append(group)}
function renderAgentGraph(status,clients,events){const host=$('agent-graph');if(!clients.length){host.replaceChildren(Object.assign(document.createElement('div'),{className:'muted',textContent:'Waiting for agents…'}));return}const latest={};events.forEach(event=>{if(event.site_id)latest[event.site_id]=event});const phaseLabels={client_connected:'Connected · ready',catalog_local:'Inspecting local metadata',site_planning:'Planning local adapter',catalog_response:'Metadata returned',analysis_local:'Running approved analysis',analysis_response:'Aggregates returned',client_disconnected:'Run complete · disconnected'};const serverLabels={STARTING:'Starting controller',WAITING_FOR_QUESTION:'Awaiting research question',DISCOVERING:'Coordinating discovery',PLANNING:'Building analysis contract',WAITING_FOR_APPROVAL:'Awaiting researcher decision',APPROVED:'Preparing approved dispatch',DISPATCHING_ANALYSIS:'Coordinating local analysis',COMPLETED:'Study complete',FAILED:'Workflow failed',REJECTED:'Proposal rejected',ABORTED:'Workflow aborted',EXPIRED:'Decision expired'};const failed=['FAILED','REJECTED','ABORTED','EXPIRED'].includes(status);const height=Math.max(220,50+clients.length*78);const serverCenter=height/2;const svg=svgElement('svg',{viewBox:`0 0 1000 ${height}`,role:'img','aria-label':`Live federated workflow: server ${serverLabels[status]||status}; ${clients.length} clients`});svg.append(svgElement('title',{},'Live server and client-agent workflow'),svgElement('desc',{},'Bidirectional links show task dispatch and aggregate responses. Actively working site agents pulse in teal.'));addGraphNode(svg,45,serverCenter-36,250,72,'Server agent',serverLabels[status]||status,failed?'failed':status==='COMPLETED'?'done':'');clients.forEach((client,index)=>{const y=45+index*78;const event=latest[client.site_id]||{};const working=event.status==='active'&&['catalog_local','site_planning','analysis_local'].includes(event.phase);const done=status==='COMPLETED';const nodeState=failed?'failed':working?'working':done?'done':client.connected?'connected':'';const outbound=(status==='DISCOVERING'&&!['catalog_response'].includes(event.phase))||(status==='DISPATCHING_ANALYSIS'&&!['analysis_response'].includes(event.phase));const inbound=(status==='DISCOVERING'&&event.phase==='catalog_response')||(status==='DISPATCHING_ANALYSIS'&&event.phase==='analysis_response');const outboundClass=`flow-line${outbound?' active':''}`,inboundClass=`flow-line${inbound?' active':''}`;const middleY=(serverCenter+y)/2;svg.append(svgElement('path',{class:outboundClass,d:`M 295 ${serverCenter-6} Q 500 ${middleY-14} 705 ${y-6}`}),svgElement('polygon',{class:`flow-arrow${outbound?' active':''}`,points:`705,${y-6} 692,${y-13} 692,${y+1}`}),svgElement('path',{class:inboundClass,d:`M 705 ${y+6} Q 500 ${middleY+14} 295 ${serverCenter+6}`}),svgElement('polygon',{class:`flow-arrow${inbound?' active':''}`,points:`295,${serverCenter+6} 308,${serverCenter-1} 308,${serverCenter+13}`}));const label=working?(phaseLabels[event.phase]||'Working locally'):inbound?'Returning aggregates':outbound?'Task dispatched':phaseLabels[event.phase]||(client.connected?'Connected · idle':'Not connected');svg.append(svgElement('text',{class:`flow-label${working||inbound||outbound?' active':''}`,x:515,y:middleY-10},label));addGraphNode(svg,705,y-32,250,64,client.site_id.replaceAll('_',' '),phaseLabels[event.phase]||(client.connected?'Connected · idle':'Not connected'),nodeState,true)});host.replaceChildren(svg)}
function renderFinal(result){const ready=!!result;show('study-summary',ready);show('final-result',ready);if(!ready){renderedFinalKey='';return}const aggregate=result.aggregate||{},artifacts=result.artifacts||[],resultKey=`${aggregate.contract_digest||''}:${result.artifact_base||''}:${artifacts.join('|')}:${result.benchmark_bundle_available===true}`;if(resultKey===renderedFinalKey)return;renderedFinalKey=resultKey;const sites=aggregate.site_row_summaries||[];let total=0,used=0,complete=true;const rows=sites.map(site=>{const counts=Object.values(site.analysis_row_counts||{});const chosen=counts.find(x=>Number.isFinite(x.rows_used));const ranged=counts.find(x=>Number.isFinite(x.rows_used_range?.minimum)&&Number.isFinite(x.rows_used_range?.maximum));const suppressed=counts.some(x=>Object.hasOwn(x,'rows_used')&&x.rows_used===null);const siteUsed=chosen?.rows_used,siteTotal=site.total_rows,range=ranged?.rows_used_range;if(!Number.isFinite(siteUsed)||!Number.isFinite(siteTotal))complete=false;else{used+=siteUsed;total+=siteTotal}const tr=document.createElement('tr');const minimal=!!range;const included=Number.isFinite(siteUsed)&&Number.isFinite(siteTotal)&&siteTotal?`${(100*siteUsed/siteTotal).toFixed(1)}%`:minimal?'Minimal':suppressed?'Privacy-protected':'Not reported by this run';const usedLabel=Number.isFinite(siteUsed)?siteUsed:(minimal?'Minimal':suppressed?'Privacy-protected':'Not reported');[site.site_id,usedLabel,siteTotal??'Privacy-protected',included,site.exclusion_reason||'Client-agent explanation unavailable for this run.'].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});return tr});const pooledKm=(aggregate.analyses||[]).find(a=>a.tool==='federated_kaplan_meier'&&a.output?.groups);const pooledUsed=pooledKm?Object.values(pooledKm.output.groups).reduce((sum,g)=>sum+(Number.isFinite(g.n)?g.n:0),0):null;const pooledTotal=sites.every(s=>Number.isFinite(s.total_rows))?sites.reduce((sum,s)=>sum+s.total_rows,0):null;$('study-sites').replaceChildren(...rows);$('study-total').textContent=complete&&sites.length?`${used.toLocaleString()} of ${total.toLocaleString()} local rows were included across ${sites.length} sites (${(100*used/total).toFixed(1)}%). Counts are site-produced aggregates.`:Number.isFinite(pooledUsed)&&Number.isFinite(pooledTotal)&&pooledTotal?`${pooledUsed.toLocaleString()} of ${pooledTotal.toLocaleString()} local rows were feasible across ${sites.length} sites (${(100*pooledUsed/pooledTotal).toFixed(1)}%); site-level exclusions below the privacy threshold are marked Minimal.`:'Local row-use counts were not reported by this run or were privacy-protected.';const base=encodeURIComponent(result.artifact_base);$('result-figures').replaceChildren(...artifacts.map(name=>{const figure=document.createElement('figure');figure.className='result-figure';const img=document.createElement('img');img.src=`/results/${base}/${encodeURIComponent(name)}`;img.alt=name.replace(/[_-]+/g,' ').replace(/\.[^.]+$/,'');figure.append(img);return figure}));$('result-summary').textContent=result.report||'The aggregate report is unavailable.';const report=document.createElement('a');report.href=`/results/${base}/report.md`;report.target='_blank';report.rel='noopener';report.textContent='Open report';const links=[report];if(result.benchmark_bundle_available){const bundle=document.createElement('a');bundle.href='/benchmark-bundle.zip';bundle.textContent='Download implementation benchmark bundle';bundle.download='agentgenie-benchmark-bundle.zip';links.push(bundle)}$('result-links').replaceChildren(...links)}
function render(s){const state=s.state||{};const status=state.status||'STARTING';currentStatus=status;$('status').textContent=status;show('question-card',status==='WAITING_FOR_QUESTION');show('waiting-card',['STARTING','DISCOVERING','PLANNING','APPROVED','DISPATCHING_ANALYSIS'].includes(status));$('waiting-text').textContent={STARTING:'Waiting for the NVFlare Controller.',DISCOVERING:'Sites are returning schema and mapping metadata; no row values are being read.',PLANNING:'Codex is translating the question into a typed contract using the registered tools.',APPROVED:'Approval validated. Preparing dispatch.',DISPATCHING_ANALYSIS:'Approved analysis is executing behind each site boundary.'}[status]||'';
const clients=s.clients||[];const connected=clients.filter(c=>c.connected).length;$('client-count').textContent=`${connected}/${clients.length}`;$('clients').replaceChildren(...(clients.length?clients.map(c=>{const d=document.createElement('div');d.className=`client ${c.connected?'connected':''}`;const head=document.createElement('div');head.className='client-head';const dot=document.createElement('span');dot.className='presence';const name=document.createElement('span');name.textContent=c.site_id;head.append(dot,name);const detail=document.createElement('div');detail.className='client-state';const seen=c.updated_at?new Date(c.updated_at).toLocaleTimeString():'not seen';detail.textContent=`${c.connected?'CONNECTED':'DISCONNECTED'} · ${seen}`;d.append(head,detail);return d}):[Object.assign(document.createElement('div'),{className:'muted',textContent:'Waiting for client presence events…'})]));
const events=s.events||[];renderAgentGraph(status,clients,events);
if(['REJECTED','EXPIRED','ABORTED','FAILED'].includes(status))message(state.reason||`Workflow ${status.toLowerCase()}.`,true);else if(status==='COMPLETED')message('Study completed. Final aggregate results are ready.');else if($('message').textContent.startsWith('Study completed.'))message('');
const completed=status==='COMPLETED';const canStartNew=['WAITING_FOR_QUESTION','WAITING_FOR_APPROVAL','COMPLETED'].includes(status);$('start-new-completed').disabled=!canStartNew;$('start-new-help').textContent=completed?'The completed result remains saved when a fresh session starts.':status==='WAITING_FOR_APPROVAL'?'Archive this proposal and return to a blank research question.':status==='WAITING_FOR_QUESTION'?'A blank research question is ready.':'Start New becomes available at the next safe human checkpoint.';renderFinal(completed?s.final_result:null);const review=!completed&&(status==='WAITING_FOR_APPROVAL'||s.proposal);show('review',!!review);show('server-review',false);show('decision',false);if(completed||!s.proposal)return;const p=s.proposal,f=s.feasibility||{},registry=(s.tool_registry||{}).tools||{},agentReview=f.server_agent_review||{},recommendationReady=!!agentReview.summary&&(agentReview.confirmation_items||[]).length>0;currentAgentReview=agentReview;show('server-review',recommendationReady&&agentReview.action==='revise');show('decision',status==='WAITING_FOR_APPROVAL'&&!s.decision_recorded&&recommendationReady);if(status==='WAITING_FOR_APPROVAL'&&!recommendationReady){show('waiting-card',true);$('waiting-text').textContent='The server agent is synthesizing the completed client responses into a concise recommendation.'}$('original-question').textContent=p.question;
const executable=(p.analyses||[]).length>0;$('approve').disabled=!executable;show('revise',agentReview.action==='revise');show('revision-guidance-wrap',agentReview.action==='revise');show('approval-controls',agentReview.action!=='revise');$('decision-notice').textContent=agentReview.action==='revise'?'Confirming sends the server agent\'s complete digest-bound revision instructions plus any additional guidance below to all site agents and replans.':executable?'Approval authorizes the exact reviewed contract. The complete client/server review bundle is digest-bound.':'No executable analysis is available and no server-agent revision has been prepared.';
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
$('start-new-completed').onclick=async()=>{if(currentStatus==='WAITING_FOR_QUESTION'){message('A blank research question is already ready.');$('question').focus();return}if(!window.confirm('Start a fresh study? The current attempt will remain archived.'))return;try{if(currentStatus==='COMPLETED'){const result=await post('/api/start-new-session',{requested_by:'Cecilie'});message(`Started ${result.session_id}. Waiting for the new Controller…`)}else if(currentStatus==='WAITING_FOR_APPROVAL'){await post('/api/new-question',{requested_by:$('approver').value});message('Current proposal archived. Reopening the question form…')}else throw Error('Start New is available at the next safe human checkpoint.');refresh()}catch(e){message(e.message,true)}};
$('revise').onclick=async()=>{if(!window.confirm('Confirm these revision points and ask the server agent to replan?'))return;try{await post('/api/revise',{recommendation_digest:currentAgentReview.revision_digest,requested_by:$('approver').value,manual_guidance:$('revision-guidance').value});message('Revision confirmed. The server recommendation and your additional guidance were submitted; planning will rerun…');refresh()}catch(e){message(e.message,true)}};refresh();setInterval(refresh,1000);
</script></body></html>'''
