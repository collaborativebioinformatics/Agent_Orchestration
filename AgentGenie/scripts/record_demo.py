#!/usr/bin/env python3
"""Record a timing-compressed, clearly labelled replay of the study console."""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, Route, sync_playwright

from biobank_agent.contracts import AnalysisContract
from biobank_agent.site import SiteExecutor


QUESTION = (
    "Across all breast cancer patients, compare overall survival between the available "
    "breast-surgery categories. Recognize the survival time and capture critical event."
)
MANUAL_REVISION = (
    "Exclude missing surgery category, missing event, and invalid survival time, report the "
    "number of feasible cases to be used for further analysis"
)
CLIENTS = [
    "site_a_university_hospital",
    "site_b_cancer_centre",
    "site_c_regional_hospital",
]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class Replay:
    def __init__(self, source_run: Path, sites_root: Path) -> None:
        self.source_run = source_run
        self.server_dir = source_run / "server"
        self.stage = "initial"
        self.stage_started = time.monotonic()
        self.proposal = _load(self.server_dir / "analysis_contract.proposed.json")
        self.proposal["question"] = QUESTION
        self.feasibility = _load(self.server_dir / "feasibility_report.json")
        self.registry = _load(self.server_dir / "tool_registry.json")
        self.final_result = self._final_result(sites_root)

    def _final_result(self, sites_root: Path) -> dict[str, Any]:
        aggregate = _load(self.server_dir / "server_aggregate.json")
        contract = AnalysisContract.parse(_load(self.server_dir / "analysis_contract.approved.json"))
        summaries = []
        for site_dir in sorted(path for path in sites_root.iterdir() if (path / "data.csv").exists()):
            result = SiteExecutor(site_dir).execute(contract)
            summaries.append(
                {
                    "site_id": result["site_id"],
                    "total_rows": result["local_total_rows"],
                    "analysis_row_counts": result["analysis_row_counts"],
                    "exclusion_reason": result["row_exclusion_reason"],
                }
            )
        aggregate["site_row_summaries"] = summaries
        report_dir = self.server_dir / "report"
        return {
            "report": (report_dir / "report.md").read_text(encoding="utf-8"),
            "aggregate": aggregate,
            "artifacts": sorted(path.name for path in report_dir.glob("*.svg")),
            "artifact_base": "report",
        }

    def advance(self, stage: str) -> None:
        self.stage = stage
        self.stage_started = time.monotonic()

    def handle(self, route: Route) -> None:
        request = route.request
        path = request.url.split("?", 1)[0]
        if path.endswith("/api/study"):
            route.fulfill(status=200, content_type="application/json", body=json.dumps(self.snapshot()))
            return
        if path.endswith("/api/question"):
            self.advance("first_work")
        elif path.endswith("/api/revise"):
            self.advance("second_work")
        elif path.endswith("/api/decision"):
            self.advance("analysis")
        route.fulfill(status=200, content_type="application/json", body=json.dumps({"ok": True}))

    def handle_artifact(self, route: Route) -> None:
        filename = route.request.url.split("?", 1)[0].rsplit("/", 1)[-1]
        if Path(filename).name != filename:
            route.fulfill(status=404, body="not found")
            return
        artifact = self.server_dir / "report" / filename
        if not artifact.is_file() or artifact.suffix.lower() not in {".svg", ".png", ".md"}:
            route.fulfill(status=404, body="not found")
            return
        content_type = {
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".md": "text/markdown; charset=utf-8",
        }[artifact.suffix.lower()]
        route.fulfill(status=200, content_type=content_type, body=artifact.read_bytes())

    def snapshot(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self.stage_started
        if self.stage == "first_work" and elapsed >= 6.0:
            self.advance("first_review")
            elapsed = 0
        elif self.stage == "second_work" and elapsed >= 6.0:
            self.advance("second_review")
            elapsed = 0
        elif self.stage == "analysis" and elapsed >= 6.0:
            self.advance("complete")
            elapsed = 0

        status = "WAITING_FOR_QUESTION"
        proposal = None
        feasibility = None
        final_result = None
        events = self._connected_events()
        if self.stage in {"first_work", "second_work"}:
            if elapsed < 3.2:
                status = "DISCOVERING"
                events += self._client_events("site_planning", "Local agent is interpreting the question.", "active")
            else:
                status = "PLANNING"
                events.append(self._event("planning", "Server agent is preparing the analysis contract.", "active"))
        elif self.stage == "first_review":
            status = "WAITING_FOR_APPROVAL"
            proposal = self._proposal(False)
            feasibility = self._review(False)
        elif self.stage == "second_review":
            status = "WAITING_FOR_APPROVAL"
            proposal = self._proposal(True)
            feasibility = self._review(True)
        elif self.stage == "analysis":
            if elapsed < 1.2:
                status = "APPROVED"
            else:
                status = "DISPATCHING_ANALYSIS"
                events += self._client_events("analysis_local", "Running approved aggregate analysis.", "active")
        elif self.stage == "complete":
            status = "COMPLETED"
            final_result = self.final_result
            events += self._client_events("analysis_response", "Disclosure-controlled aggregates returned.", "completed")

        return {
            "session_id": "recorded-demo-replay",
            "state": {"status": status, "updated_at": _timestamp(), "patient_data_accessed": status in {"DISPATCHING_ANALYSIS", "COMPLETED"}},
            "request": None if self.stage == "initial" else {"question": QUESTION},
            "proposal": proposal,
            "feasibility": feasibility,
            "tool_registry": self.registry,
            "decision_recorded": False,
            "approved_contract_available": status in {"APPROVED", "DISPATCHING_ANALYSIS", "COMPLETED"},
            "events": events,
            "clients": [{"site_id": site, "connected": True, "updated_at": _timestamp()} for site in CLIENTS],
            "final_result": final_result,
            "can_start_new_session": True,
        }

    def _review(self, supported: bool) -> dict[str, Any]:
        value = copy.deepcopy(self.feasibility)
        if supported:
            value["unavailable_requests"] = []
        for site in value.get("sites", []):
            site["all_requested_analyses_supported"] = supported
            if supported:
                site["unsupported"] = []
                proposal = site.get("client_agent_proposal")
                if isinstance(proposal, dict):
                    proposal["unavailable_concepts"] = []
            else:
                site["supported"] = []
                site["unsupported"] = [
                    {"analysis_id": "complete_case_survival", "missing_fields": ["explicit eligibility rules"]}
                ]
        if supported:
            value["server_agent_review"] = {
                "schema_version": "biobank.server_agent_review.v1",
                "action": "approve",
                "summary": "All three sites support the revised complete-case survival analysis. Confirm the analysis contract before execution.",
                "confirmation_items": [
                    "Exclude missing surgery category, missing event, and invalid survival time.",
                    "Report disclosure-controlled feasible-case counts before the survival comparison.",
                    "Run federated Kaplan–Meier by the available breast-surgery categories.",
                ],
                "revision_digest": "",
            }
        else:
            guidance = "Define complete-case eligibility and require disclosure-controlled feasible-case counts at every site."
            value["server_agent_review"] = {
                "schema_version": "biobank.server_agent_review.v1",
                "action": "revise",
                "summary": "The sites can perform the analysis, but the first contract needs explicit complete-case exclusions and feasible-case reporting.",
                "confirmation_items": [
                    "Define how missing surgery category and event values are handled.",
                    "Exclude invalid survival times rather than treating them as valid follow-up.",
                    "Return disclosure-controlled feasible-case counts with the analysis.",
                ],
                "revision_digest": "demo-revision-digest",
                "full_revision_guidance": guidance,
            }
        return value

    def _proposal(self, supported: bool) -> dict[str, Any]:
        proposal = copy.deepcopy(self.proposal)
        if supported:
            proposal["unavailable_requests"] = []
        return proposal

    @staticmethod
    def _event(phase: str, message: str, status: str, site_id: str | None = None) -> dict[str, Any]:
        return {
            "timestamp": _timestamp(),
            "phase": phase,
            "actor": "client" if site_id else "server",
            "site_id": site_id,
            "status": status,
            "message": message,
            "patient_data_accessed": phase == "analysis_local",
        }

    def _connected_events(self) -> list[dict[str, Any]]:
        return [self._event("client_connected", "Client connected.", "completed", site) for site in CLIENTS]

    def _client_events(self, phase: str, message: str, status: str) -> list[dict[str, Any]]:
        return [self._event(phase, message, status, site) for site in CLIENTS]


def slow_scroll(page: Page, target: str, pause: float = 0.14) -> None:
    target_y = page.locator(target).evaluate("element => element.getBoundingClientRect().top + window.scrollY - 24")
    current = page.evaluate("window.scrollY")
    step = 45 if target_y >= current else -45
    while (step > 0 and current < target_y) or (step < 0 and current > target_y):
        current = current + step
        if step > 0:
            current = min(current, target_y)
        else:
            current = max(current, target_y)
        page.evaluate("y => window.scrollTo(0, y)", current)
        page.wait_for_timeout(int(pause * 1000))


def wait_for_image(page: Page, selector: str, timeout_ms: int = 10_000) -> None:
    image = page.locator(selector).first
    image.wait_for(state="visible", timeout=timeout_ms)
    deadline = time.monotonic() + timeout_ms / 1_000
    while time.monotonic() < deadline:
        if image.evaluate("element => element.complete && element.naturalWidth > 0"):
            return
        page.wait_for_timeout(100)
    raise TimeoutError(f"Image did not render: {selector}")


def record(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    replay = Replay(args.source_run.resolve(), args.sites_root.resolve())
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(output_dir),
            record_video_size={"width": 1440, "height": 900},
            color_scheme="dark",
        )
        page = context.new_page()
        page.route("**/api/**", replay.handle)
        page.route("**/results/**", replay.handle_artifact)
        page.on("dialog", lambda dialog: dialog.accept())
        page.goto(args.url, wait_until="networkidle")
        page.evaluate(
            """() => { const badge=document.createElement('div'); badge.textContent='RECORDED WORKFLOW REPLAY · TIMING COMPRESSED';
            Object.assign(badge.style,{position:'fixed',top:'12px',right:'18px',zIndex:9999,padding:'8px 12px',
            border:'1px solid #22d3c5',borderRadius:'999px',background:'#071313ee',color:'#22d3c5',
            font:'700 11px ui-monospace,monospace',letterSpacing:'.05em'}); document.body.append(badge); }"""
        )
        page.wait_for_timeout(1800)
        page.locator("#question").type(QUESTION, delay=18)
        page.locator("#catalog-consent").check()
        page.wait_for_timeout(700)
        page.locator("#submit-question").click()
        slow_scroll(page, "#execution-card", 0.12)
        page.wait_for_timeout(6500)

        slow_scroll(page, "#review", 0.15)
        slow_scroll(page, "#sites", 0.15)
        page.wait_for_timeout(3200)
        slow_scroll(page, "#server-review", 0.15)
        page.wait_for_timeout(2600)
        slow_scroll(page, "#decision", 0.15)
        page.wait_for_timeout(1800)
        page.locator("#revision-guidance").type(MANUAL_REVISION, delay=22)
        page.wait_for_timeout(900)
        page.locator("#revise").click()
        slow_scroll(page, "#execution-card", 0.12)
        page.wait_for_timeout(6500)

        slow_scroll(page, "#review", 0.14)
        slow_scroll(page, "#sites", 0.14)
        page.wait_for_timeout(3500)
        slow_scroll(page, "#decision", 0.14)
        page.wait_for_timeout(1800)
        page.locator("#confirmation").fill("APPROVE")
        page.wait_for_timeout(900)
        page.locator("#approve").click()
        slow_scroll(page, "#execution-card", 0.12)
        page.wait_for_timeout(6500)

        slow_scroll(page, "#study-summary", 0.16)
        page.wait_for_timeout(4500)
        slow_scroll(page, "#final-result", 0.16)
        wait_for_image(page, "#result-figures img")
        page.wait_for_timeout(1800)
        slow_scroll(page, "#result-figures", 0.16)
        page.wait_for_timeout(10_000)
        video = page.video
        page.close()
        context.close()
        browser.close()
        return Path(video.path())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--source-run", type=Path, default=Path("../runs/ui-20260917-150739-453c"))
    parser.add_argument("--sites-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("demo-video"))
    args = parser.parse_args()
    path = record(args)
    print(path)


if __name__ == "__main__":
    main()
