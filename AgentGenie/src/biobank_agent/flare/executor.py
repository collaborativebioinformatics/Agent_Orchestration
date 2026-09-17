"""NVFlare site Executor for catalog discovery and approved analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from biobank_agent.contracts import AnalysisContract
from biobank_agent.flare import ANALYSIS_TASK, CATALOG_TASK, PROGRESS_TOPIC
from biobank_agent.site_agent import CodexSiteAgent
from biobank_agent.site import SiteExecutor

from nvflare.apis.executor import Executor
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

PAYLOAD_KEY = "biobank_payload"


class BiobankSiteExecutor(Executor):
    """Keep local CSV access behind the client execution boundary."""

    def __init__(self, site_dir: str) -> None:
        super().__init__()
        # Keep constructor state JSON-serializable. FedJob derives component
        # configuration from instance attributes; storing a Path here causes it
        # to be emitted as an empty PosixPath constructor (that is, ".").
        self.site_dir = site_dir

    def handle_event(self, event_type: str, fl_ctx: FLContext) -> None:
        if event_type == EventType.START_RUN:
            self._emit_progress(fl_ctx, "client_connected")
        elif event_type == EventType.END_RUN:
            self._emit_progress(fl_ctx, "client_disconnected")

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        try:
            site_dir = Path(self.site_dir)
            if task_name == CATALOG_TASK:
                self._emit_progress(fl_ctx, "catalog_started")
                # Read schema and declared mapping metadata, never patient rows or values.
                catalog = json.loads((site_dir / "catalog.json").read_text(encoding="utf-8"))
                catalog["schema_fields"] = list(pd.read_csv(site_dir / "data.csv", nrows=0).columns)
                mappings_path = site_dir / "mappings.yaml"
                catalog["declared_mappings_yaml"] = (
                    mappings_path.read_text(encoding="utf-8") if mappings_path.exists() else ""
                )
                request = shareable.get(PAYLOAD_KEY) or {}
                question = str(request.get("question", "")).strip()
                if not question:
                    return make_reply(ReturnCode.BAD_TASK_DATA)
                self._emit_progress(fl_ctx, "site_planning_started")
                catalog["site_agent_assessment"] = CodexSiteAgent().assess(
                    question=question,
                    catalog=catalog,
                    mappings_yaml=catalog["declared_mappings_yaml"],
                )
                self._emit_progress(fl_ctx, "site_planning_completed")
                self._emit_progress(fl_ctx, "catalog_completed")
                return Shareable({PAYLOAD_KEY: {"schema_version": "biobank.catalog_response.v1", "catalog": catalog}})
            if task_name == ANALYSIS_TASK:
                payload = shareable.get(PAYLOAD_KEY)
                if not isinstance(payload, dict) or not isinstance(payload.get("contract"), dict):
                    return make_reply(ReturnCode.BAD_TASK_DATA)
                contract = AnalysisContract.parse(payload["contract"])
                contract.require_approval()
                if payload.get("contract_digest") != contract.digest:
                    return make_reply(ReturnCode.BAD_TASK_DATA)
                self._emit_progress(fl_ctx, "analysis_started")
                aggregate = SiteExecutor(site_dir).execute(contract)
                self._emit_progress(fl_ctx, "analysis_completed")
                return Shareable({PAYLOAD_KEY: aggregate})
            return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as exc:  # NVFlare executors must report local failure as a reply.
            self.log_exception(fl_ctx, f"Biobank site execution failed: {exc}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    @staticmethod
    def _emit_progress(fl_ctx: FLContext, event: str) -> None:
        """Send non-sensitive, best-effort client progress to the server."""
        fl_ctx.get_engine().send_aux_request(
            targets=None,
            topic=PROGRESS_TOPIC,
            request=Shareable({"event": event}),
            timeout=0,
            fl_ctx=fl_ctx,
            optional=True,
        )
