from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from biobank_agent.aggregate import aggregate_site_results
from biobank_agent.benchmark import create_benchmark_bundle
from biobank_agent.codex import CodexPlanner
from biobank_agent.contracts import AnalysisContract
from biobank_agent.feasibility import assess_feasibility, promote_verified_site_adapters
from biobank_agent.local_profile import build_local_profile, verify_adapter_locally
from biobank_agent.flare.approval import ApprovalRejected, HumanApprovalGate, create_human_decision
from biobank_agent.render import _km_svg
from biobank_agent.site import SiteExecutor
from biobank_agent.site_agent import CodexSiteAgent
from biobank_agent.tools.harmonize import harmonize
from biobank_agent.tools.local import survival_histograms
from biobank_agent.ui.server import HTML, StudyConsoleStore


def _contract(approved: bool) -> AnalysisContract:
    payload = {
        "schema_version": "biobank.analysis_contract.v2",
        "study_id": "general-tools-test",
        "question": "Compare two declared cohorts",
        "training_allowed": False,
        "cohort_definition": "Contract-defined synthetic groups",
        "cohorts": [
            {"name": "group-a", "predicate": {"eq_ci": {"field": "canonical_group", "value": "a"}}},
            {"name": "group-b", "predicate": {"eq_ci": {"field": "canonical_group", "value": "b"}}},
        ],
        "site_harmonization": {"site_one": {"fields": {"canonical_group": {"source": "local_group"}}}},
        "approved_tools": ["federated_histogram", "categorical_contingency"],
        "privacy": {"min_cell_count": 5, "forbidden_outputs": ["row_level"]},
        "analyses": [
            {
                "analysis_id": "numeric_distribution",
                "tool": "federated_histogram",
                "field": "measurement",
                "cohorts": ["group-a", "group-b"],
                "bins": [0, 10, 20, 30, 40],
            },
            {
                "analysis_id": "category_association",
                "tool": "categorical_contingency",
                "field": "category",
                "cohorts": ["group-a", "group-b"],
            },
        ],
        "approval": {"status": "proposed"},
    }
    contract = AnalysisContract.parse(payload)
    if approved:
        payload["approval"] = {"status": "approved", "contract_digest": contract.approval_digest}
        contract = AnalysisContract.parse(payload)
    return contract


def _site(tmp_path: Path) -> Path:
    site = tmp_path / "site_one"
    site.mkdir()
    rows = [
        {
            "record_id": index,
            "measurement": index + 1,
            "local_group": "a" if index < 12 else "b",
            "category": "x" if index % 2 else "y",
        }
        for index in range(24)
    ]
    pd.DataFrame(rows).to_csv(site / "data.csv", index=False)
    (site / "catalog.json").write_text(json.dumps({"site_id": "site_one"}), encoding="utf-8")
    return site


def test_execution_requires_explicit_approval(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        SiteExecutor(_site(tmp_path)).execute(_contract(approved=False))


def test_general_tools_export_only_aggregates(tmp_path: Path) -> None:
    contract = _contract(approved=True)
    result = SiteExecutor(_site(tmp_path)).execute(contract)
    assert result["patient_rows_exported"] == 0
    assert "record_id" not in json.dumps(result)
    pooled = aggregate_site_results([result], contract.payload)
    assert pooled["patient_rows_received"] == 0
    histogram = next(item for item in pooled["analyses"] if item["tool"] == "federated_histogram")
    assert histogram["output"]["groups"]["group-a"]["n"] == 12


def test_training_contract_is_rejected() -> None:
    payload = _contract(approved=False).payload.copy()
    payload["training_allowed"] = True
    with pytest.raises(ValueError, match="training_allowed=false"):
        AnalysisContract.parse(payload)


def test_harmonization_preserves_categories_and_maps_other_non_missing_values() -> None:
    data = pd.DataFrame(
        {
            "local_surgery": ["Lumpectomy", "Mastectomy", None],
            "local_status": ["Died of Disease", "Living", None],
            "survival_years": [1.0, 2.0, 3.0],
        }
    )
    config = {
        "site_one": {
            "fields": {
                "surgery": {"source": "local_surgery", "value_map": {}, "multiply": 1.0},
                "event": {
                    "source": "local_status",
                    "value_map": {"Died of Disease": 1, "__OTHER_NON_MISSING__": 0},
                    "multiply": 1.0,
                },
                "survival_months": {"source": "survival_years", "value_map": {}, "multiply": 12.0},
            }
        }
    }

    result, _ = harmonize(data, "site_one", config)

    assert result["surgery"].tolist()[:2] == ["Lumpectomy", "Mastectomy"]
    assert result["event"].tolist()[:2] == [1.0, 0.0]
    assert pd.isna(result["event"].iloc[2])
    assert result["survival_months"].tolist() == [12.0, 24.0, 36.0]


def test_kaplan_meier_matches_integer_event_contract_to_float_data() -> None:
    data = pd.DataFrame(
        {
            "time": [3.0, 8.0, 15.0, 18.0],
            "event": [1.0, 0.0, 1.0, 0.0],
            "group": ["a", "a", "a", "a"],
        }
    )
    result = survival_histograms(
        data,
        {"all": pd.Series(True, index=data.index)},
        {
            "time_field": "time",
            "event": {"field": "event", "values": [1]},
            "group_by": "group",
            "subset_cohort": "all",
            "time_bins": [0, 12, 24],
        },
        min_cell=1,
    )

    assert result["groups"]["a"]["events"] == [1, 1]
    assert result["groups"]["a"]["censored"] == [1, 1]


def test_kaplan_meier_excludes_missing_event_group_and_invalid_time() -> None:
    data = pd.DataFrame(
        {
            "time": [3.0, 8.0, -1.0, 14.0, 16.0],
            "event": [1.0, None, 0.0, 0.0, 1.0],
            "group": ["a", "a", "a", None, "a"],
        }
    )

    result = survival_histograms(
        data,
        {"all": pd.Series(True, index=data.index)},
        {
            "time_field": "time",
            "event": {"field": "event", "values": [1]},
            "group_by": "group",
            "subset_cohort": "all",
            "time_bins": [0, 12, 24],
        },
        min_cell=1,
    )

    assert result["groups"]["a"]["n"] == 2
    assert result["groups"]["a"]["events"] == [1, 1]
    assert result["groups"]["a"]["censored"] == [0, 0]


def test_site_reports_kaplan_meier_complete_case_rows_without_subset_cohort(tmp_path: Path) -> None:
    site = tmp_path / "site_one"
    site.mkdir()
    pd.DataFrame(
        {
            "time": [3.0, 8.0, -1.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0],
            "event": [1.0, None, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, None, 0.0],
            "group": ["a", "a", "a", None, "a", "a", "a", "a", "a", None],
        }
    ).to_csv(site / "data.csv", index=False)
    (site / "catalog.json").write_text(json.dumps({"site_id": "site_one"}), encoding="utf-8")
    payload = {
        "schema_version": "biobank.analysis_contract.v2",
        "study_id": "row-count-test",
        "question": "Compare survival",
        "training_allowed": False,
        "cohort_definition": "All records with a group value",
        "cohorts": [{"name": "all", "predicate": {"not": {"is_missing": {"field": "group"}}}}],
        "site_harmonization": {},
        "approved_tools": ["federated_kaplan_meier"],
        "privacy": {"min_cell_count": 5, "forbidden_outputs": ["row_level"]},
        "analyses": [
            {
                "analysis_id": "survival",
                "tool": "federated_kaplan_meier",
                "cohorts": ["all"],
                "time_field": "time",
                "event": {"field": "event", "values": [1]},
                "group_by": "group",
                "time_bins": [0, 12, 24],
            }
        ],
        "approval": {"status": "proposed"},
    }
    proposed = AnalysisContract.parse(payload)
    payload["approval"] = {"status": "approved", "contract_digest": proposed.approval_digest}

    result = SiteExecutor(site).execute(AnalysisContract.parse(payload))

    assert result["analysis_row_counts"]["survival"]["rows_used"] == 5
    assert result["analysis_row_counts"]["survival"]["selection"].startswith("complete cases")
    assert result["row_exclusion_reason"].startswith("5 local rows were excluded")


def test_edit_after_approval_invalidates_it() -> None:
    contract = _contract(approved=True)
    assert contract.is_approved
    contract.payload["privacy"]["min_cell_count"] = 6
    assert not contract.is_approved


def test_controller_gate_consumes_matching_human_decision(tmp_path: Path) -> None:
    proposal = _contract(approved=False)
    gate = HumanApprovalGate(tmp_path)
    gate.publish(proposal, {"schema_version": "biobank.feasibility_report.v2"})
    decision = create_human_decision(gate.proposal_path, decision="approve", approved_by="tester")
    gate._write(gate.approval_path, decision)
    approved = gate._consume_decision(proposal)
    assert approved.is_approved
    assert json.loads(gate.state_path.read_text())["status"] == "APPROVED"


def test_controller_gate_rejects_wrong_digest(tmp_path: Path) -> None:
    proposal = _contract(approved=False)
    gate = HumanApprovalGate(tmp_path)
    gate.publish(proposal, {"schema_version": "biobank.feasibility_report.v2"})
    gate._write(
        gate.approval_path,
        {
            "schema_version": "biobank.human_approval.v1",
            "decision": "approve",
            "proposal_digest": "wrong",
            "approved_by": "tester",
        },
    )
    with pytest.raises(ApprovalRejected, match="does not match"):
        gate._consume_decision(proposal)


def test_controller_gate_binds_approval_to_client_review_bundle(tmp_path: Path) -> None:
    proposal = _contract(approved=False)
    gate = HumanApprovalGate(tmp_path)
    feasibility = {"schema_version": "biobank.feasibility_report.v2", "sites": [{"site_id": "site_one"}]}
    gate.publish(proposal, feasibility)
    decision = create_human_decision(gate.proposal_path, decision="approve", approved_by="tester")
    gate._write(gate.approval_path, decision)
    gate._write(
        gate.feasibility_path,
        {"schema_version": "biobank.feasibility_report.v2", "sites": [{"site_id": "changed"}]},
    )
    with pytest.raises(ApprovalRejected, match="client and server review bundle"):
        gate._consume_decision(proposal)


def test_feasibility_exposes_client_agent_proposal_for_human_review() -> None:
    proposal = _contract(approved=False)
    catalog = {
        "site_id": "site_one",
        "schema_fields": ["local_group", "measurement", "category"],
        "site_agent_assessment": {
            "supported_concepts": [
                {"concept": "study group", "source_fields": ["local_group"], "evidence": "CSV header"}
            ],
            "unavailable_concepts": [{"concept": "endpoint", "reason": "not requested"}],
            "data_adapter": {
                "status": "ready",
                "digest": "adapter-digest",
                "fields": {"canonical_group": {"source": "local_group"}},
                "unresolved": [],
            },
            "analysis_proposals": [
                {"tool": "federated_histogram", "parameters": {"field": "measurement"}, "caveats": []}
            ],
        },
    }

    report = assess_feasibility(proposal, [catalog])
    site = report["sites"][0]

    assert site["data_adapter"]["fields"]["canonical_group"]["source"] == "local_group"
    assert site["client_agent_proposal"]["analysis_proposals"][0]["tool"] == "federated_histogram"
    assert report["federation_ready"] is True
    assert site["fedready"] is True


def test_server_promotes_ready_verified_site_adapter_before_human_review() -> None:
    proposal = _contract(approved=False)
    catalog = {
        "site_id": "site_one",
        "site_agent_assessment": {
            "data_adapter": {
                "status": "ready",
                "unresolved": [],
                "fields": {
                    "canonical_group": {
                        "source": "local_group",
                        "multiply": 1.0,
                        "value_map": {"__OTHER_NON_MISSING__": "other"},
                    }
                },
            }
        },
    }

    reconciled = promote_verified_site_adapters(proposal, [catalog])

    mapping = reconciled.payload["site_harmonization"]["site_one"]["fields"]["canonical_group"]
    assert mapping["value_map"] == {"__OTHER_NON_MISSING__": "other"}


def test_ui_handles_question_and_approval_without_bypassing_controller(tmp_path: Path) -> None:
    store = StudyConsoleStore(tmp_path)
    store.gate.set_state("WAITING_FOR_QUESTION")
    request = store.submit_question("A new scientific question", "researcher", True)
    assert request["question"] == "A new scientific question"
    assert request["catalog_metadata_codex_consent"] is True
    proposal = _contract(approved=False)
    store.gate.publish(proposal, {"schema_version": "biobank.feasibility_report.v2", "sites": []})
    decision = store.decide("approve", "researcher", "APPROVE", None)
    assert decision["proposal_digest"] == proposal.approval_digest
    assert not store.gate.approved_contract_path.exists()


def test_ui_appends_manual_guidance_to_digest_bound_revision(tmp_path: Path) -> None:
    store = StudyConsoleStore(tmp_path)
    store.gate.set_state("WAITING_FOR_QUESTION")
    store.submit_question("Compare survival", "Cecilie", True)
    proposal = _contract(approved=False)
    server_guidance = "Use a common endpoint and shared time bins."
    digest = hashlib.sha256(server_guidance.encode()).hexdigest()
    store.gate.publish(
        proposal,
        {
            "schema_version": "biobank.feasibility_report.v2",
            "sites": [],
            "server_agent_review": {
                "schema_version": "biobank.server_agent_review.v1",
                "action": "revise",
                "summary": "Revision is required.",
                "confirmation_items": ["Confirm endpoint."],
                "full_revision_guidance": server_guidance,
                "revision_digest": digest,
            },
        },
    )

    store.revise_proposal(digest, "Cecilie", "Exclude negative follow-up times.")

    restart = json.loads(store.gate.new_question_path.read_text())
    next_request = restart["next_request"]
    assert next_request["revision_guidance"] == server_guidance
    assert next_request["manual_revision_guidance"] == "Exclude negative follow-up times."
    assert "Additional guidance supplied by the researcher" in next_request["question"]
    assert "Exclude negative follow-up times." in next_request["question"]


def test_completed_ui_exposes_final_result_and_disclosure_controlled_row_summary(tmp_path: Path) -> None:
    store = StudyConsoleStore(tmp_path)
    store.gate.server_dir.mkdir(parents=True)
    store.gate.set_state("COMPLETED")
    report_dir = store.gate.server_dir / "report"
    report_dir.mkdir()
    (report_dir / "report.md").write_text("# Result\n", encoding="utf-8")
    (report_dir / "survival.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    store.gate._write(
        store.gate.server_dir / "server_aggregate.json",
        {
            "site_row_summaries": [
                {
                    "site_id": "site_one",
                    "total_rows": 24,
                    "analysis_row_counts": {"survival": {"rows_used": 20, "selection": "eligible"}},
                }
            ]
        },
    )

    result = store.snapshot()["final_result"]

    assert result["artifacts"] == ["survival.svg"]
    assert result["aggregate"]["site_row_summaries"][0]["analysis_row_counts"]["survival"]["rows_used"] == 20
    assert store.result_artifact("report", "survival.svg").name == "survival.svg"


def test_kaplan_meier_svg_includes_numeric_axes_and_grid(tmp_path: Path) -> None:
    destination = tmp_path / "curve.svg"
    result = {
        "bins": [0, 12, 24, 36, 60, 120, 180, 240, 360],
        "groups": {
            "group_a": {
                "n": 42,
                "curve": [
                    {"time": 0, "survival": 1.0},
                    {"time": 120, "survival": 0.75},
                    {"time": 360, "survival": 0.5},
                ],
            }
        },
    }

    _km_svg(result, destination, "Overall survival", "Time (months)")

    svg = destination.read_text(encoding="utf-8")
    assert ">0.00</text>" in svg
    assert ">0.25</text>" in svg
    assert ">1.00</text>" in svg
    assert ">12</text>" in svg
    assert ">360</text>" in svg
    assert "Time (months)" in svg
    assert 'stroke="#303536"' in svg


def test_completed_run_creates_privacy_safe_implementation_benchmark_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "study-one"
    gate = HumanApprovalGate(run_dir)
    contract = _contract(approved=True)
    gate._write(gate.approved_contract_path, contract.payload)
    gate._write(
        gate.feasibility_path,
        {
            "schema_version": "biobank.feasibility_report.v2",
            "sites": [
                {
                    "site_id": "site_one",
                    "data_adapter": {
                        "status": "ready",
                        "digest": "adapter-digest",
                        "fields": {"canonical_group": {"source": "local_group"}},
                    },
                    "client_agent_proposal": {"analysis_proposals": []},
                    "all_requested_analyses_supported": True,
                }
            ],
        },
    )
    gate._write(gate.approval_path, {"schema_version": "biobank.human_approval.v1"})
    gate._write(gate.tool_registry_path, {"schema_version": "biobank.tool_registry.v1", "tools": {}})
    gate._write(gate.server_dir / "server_aggregate.json", {"schema_version": "biobank.server_aggregate.v2"})
    report = gate.server_dir / "report"
    report.mkdir(parents=True)
    (report / "report.md").write_text("# Aggregate result\n", encoding="utf-8")
    (report / "curve.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    gate.set_state("COMPLETED", contract_digest=contract.digest)

    workspace = tmp_path / "workspace" / "biobank_agentic_analysis"
    server_job = workspace / "server" / "simulate_job"
    gate._write(server_job / "meta.json", {"name": "biobank_agentic_analysis"})
    gate._write(server_job / "app_server" / "config" / "config_fed_server.json", {"workflows": []})
    server_code = server_job / "app_server" / "custom" / "biobank_agent"
    server_code.mkdir(parents=True)
    (server_code / "controller.py").write_text("# frozen server implementation\n", encoding="utf-8")
    client_job = workspace / "site_one" / "simulate_job" / "app_site_one"
    gate._write(client_job / "config" / "config_fed_client.json", {"executors": []})
    client_code = client_job / "custom" / "biobank_agent"
    client_code.mkdir(parents=True)
    (client_code / "site.py").write_text("# frozen client implementation\n", encoding="utf-8")
    (client_job / "data.csv").write_text("patient_id,value\n1,secret\n", encoding="utf-8")

    archive = create_benchmark_bundle(run_dir, workspace_root=tmp_path / "workspace")

    assert archive == gate.server_dir / "benchmark_bundle.zip"
    adapter = json.loads((gate.server_dir / "benchmark_bundle" / "site_adapters" / "site_one.json").read_text())
    assert adapter["agent_proposed_adapter"]["digest"] == "adapter-digest"
    assert adapter["executed_harmonization"] == contract.payload["site_harmonization"]["site_one"]
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert "site_adapters/site_one.json" in names
    assert "nvflare_job/server/deployed_biobank_agent/controller.py" in names
    assert "nvflare_job/clients/site_one/deployed_biobank_agent/site.py" in names
    assert not any(name.endswith("data.csv") for name in names)
    store = StudyConsoleStore(run_dir)
    assert store.snapshot()["final_result"]["benchmark_bundle_available"] is True
    assert store.benchmark_bundle() == archive


def test_ui_reopens_persisted_active_session_after_restart(tmp_path: Path) -> None:
    old_run = tmp_path / "completed-session"
    active_run = tmp_path / "new-session"
    HumanApprovalGate(old_run).set_state("COMPLETED")
    HumanApprovalGate(active_run).set_state("WAITING_FOR_APPROVAL")
    HumanApprovalGate._write(
        tmp_path / ".active_session.json",
        {
            "schema_version": StudyConsoleStore.ACTIVE_SESSION_SCHEMA,
            "session_id": active_run.name,
        },
    )

    restarted = StudyConsoleStore(old_run, output_root=tmp_path)

    snapshot = restarted.snapshot()
    assert restarted.gate.run_dir == active_run.resolve()
    assert snapshot["session_id"] == "new-session"
    assert snapshot["state"]["status"] == "WAITING_FOR_APPROVAL"


def test_ui_ignores_active_session_pointer_outside_output_root(tmp_path: Path) -> None:
    fallback = tmp_path / "fallback"
    HumanApprovalGate(fallback).set_state("WAITING_FOR_QUESTION")
    HumanApprovalGate._write(
        tmp_path / ".active_session.json",
        {
            "schema_version": StudyConsoleStore.ACTIVE_SESSION_SCHEMA,
            "session_id": "../outside",
        },
    )

    restarted = StudyConsoleStore(fallback, output_root=tmp_path)

    assert restarted.gate.run_dir == fallback.resolve()


def test_ui_can_start_fresh_session_after_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    failed_run = tmp_path / "failed-session"
    HumanApprovalGate(failed_run).set_state("FAILED", reason="contract planning failed")
    launched: list[list[str]] = []

    class FakeProcess:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            launched.append(command)

    monkeypatch.setattr("biobank_agent.ui.server.subprocess.Popen", FakeProcess)
    store = StudyConsoleStore(
        failed_run,
        sites_root=tmp_path / "sites",
        workspace_root=tmp_path / "workspace",
        output_root=tmp_path,
    )

    result = store.start_new_session("Cecilie")

    assert result["session_id"].startswith("ui-")
    assert store.snapshot()["state"]["status"] == "STARTING"
    assert launched and "run-simulation" in launched[0]
    assert HumanApprovalGate(failed_run).state_path.exists()


def test_ui_uses_live_agent_graph_and_places_completed_action_in_sidebar() -> None:
    assert 'id="execution-card"' in HTML
    assert 'id="agent-graph"' in HTML
    assert 'id="timeline-card"' not in HTML
    clients = HTML.index('id="clients-card"')
    action = HTML.index('id="completed-actions"')
    sidebar_end = HTML.index("</aside>", clients)
    assert clients < action < sidebar_end
    assert "'COMPLETED','FAILED'" in HTML


def test_site_agent_retries_adapter_that_uses_a_derived_field_as_source() -> None:
    class FakePlanner:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def run_json(self, prompt: str, work_dir: Path) -> dict:
            self.prompts.append(prompt)
            source = "time_months" if len(self.prompts) == 1 else "overall_survival_years"
            return {
                "schema_version": "biobank.site_agent_assessment.v1",
                "site_id": "site_one",
                "supported_concepts": [],
                "unavailable_concepts": [],
                "data_adapter": {
                    "schema_version": "biobank.declarative_data_adapter.v1",
                    "status": "ready",
                    "fields": {"time_months": {"source": source, "multiply": 12.0}},
                    "unresolved": [],
                },
                "analysis_proposals": [],
            }

    agent = CodexSiteAgent()
    planner = FakePlanner()
    agent.planner = planner
    result = agent.assess(
        question="Compare survival",
        catalog={"site_id": "site_one", "schema_fields": ["overall_survival_years"]},
        mappings_yaml="outcome:\n  time: time_months\n",
    )

    assert len(planner.prompts) == 2
    assert "absolute source field allowlist" in planner.prompts[1]
    assert result["data_adapter"]["fields"]["time_months"]["source"] == "overall_survival_years"
    assert result["data_adapter"]["digest"]


def test_site_agent_canonicalizes_default_nonmissing_value_map_sentinel() -> None:
    result = {
        "schema_version": "biobank.site_agent_assessment.v1",
        "site_id": "site_one",
        "data_adapter": {
            "schema_version": "biobank.declarative_data_adapter.v1",
            "status": "ready",
            "fields": {
                "event": {
                    "source": "status",
                    "value_map": {"Died": 1, "__OTHER_NONMISSING__": 0, "__MISSING__": None},
                }
            },
            "unresolved": [],
        },
        "analysis_proposals": [],
    }

    CodexSiteAgent._validate(result, {"site_id": "site_one", "schema_fields": ["status"]})

    assert result["data_adapter"]["fields"]["event"]["value_map"] == {
        "Died": 1,
        "__OTHER_NON_MISSING__": 0,
    }


def test_site_agent_returns_incomplete_assessment_after_two_invalid_adapters() -> None:
    class InvalidPlanner:
        def run_json(self, prompt: str, work_dir: Path) -> dict:
            return {
                "schema_version": "biobank.site_agent_assessment.v1",
                "site_id": "site_one",
                "supported_concepts": [],
                "unavailable_concepts": [],
                "data_adapter": {
                    "schema_version": "biobank.declarative_data_adapter.v1",
                    "status": "ready",
                    "fields": {"time_months": {"source": "invented"}},
                    "unresolved": [],
                },
                "analysis_proposals": [],
            }

    agent = CodexSiteAgent()
    agent.planner = InvalidPlanner()
    result = agent.assess(
        question="Compare survival",
        catalog={"site_id": "site_one", "schema_fields": ["overall_survival_months"]},
        mappings_yaml="",
    )

    assert result["data_adapter"]["status"] == "incomplete"
    assert result["data_adapter"]["fields"] == {}
    assert "failed boundary verification" in result["unavailable_concepts"][0]["reason"]


def test_local_profile_suppresses_rare_values_and_verifies_generated_adapter() -> None:
    data = pd.DataFrame(
        {
            "survival_years": [float(index) for index in range(20)],
            "vital_status": ["Living"] * 12 + ["Dead"] * 7 + ["Rare status"],
        }
    )
    catalog = {
        "clinical_features": {"survival_years": {}, "vital_status": {}},
        "schema_fields": list(data.columns),
    }
    profile = build_local_profile(data, catalog, "Compare overall survival", min_cell_count=5)

    categories = profile["fields"]["vital_status"]["observed_categories"]
    assert not any(item["value"] == "Rare status" for item in categories)
    assert any(item["value"] == "__SUPPRESSED_CATEGORIES__" for item in categories)
    adapter = {
        "fields": {
            "time_months": {"source": "survival_years", "value_map": {}, "multiply": 12.0},
            "event_death": {
                "source": "vital_status",
                "value_map": {"Dead": 1, "__OTHER_NON_MISSING__": 0},
                "multiply": 1.0,
            },
        }
    }
    verify_adapter_locally(adapter, data, profile["digest"])
    assert adapter["verification"]["status"] == "verified_against_local_profile"


def test_site_scoped_analysis_is_only_checked_executed_and_pooled_at_target(tmp_path: Path) -> None:
    payload = _contract(approved=False).payload
    payload["analyses"] = [{**payload["analyses"][0], "site_id": "site_one"}]
    payload["approved_tools"] = ["federated_histogram"]
    proposal = AnalysisContract.parse(payload)
    other_catalog = {"site_id": "site_two", "schema_fields": []}
    report = assess_feasibility(proposal, [other_catalog])
    assert report["sites"][0]["targeted_analysis_count"] == 0
    assert report["sites"][0]["support_status"] == "not_targeted"
    assert report["sites"][0]["all_requested_analyses_supported"] is True
    assert report["federation_ready"] is False
    assert report["sites"][0]["fedready"] is False

    payload["approval"] = {"status": "approved", "contract_digest": proposal.approval_digest}
    result = SiteExecutor(_site(tmp_path)).execute(AnalysisContract.parse(payload))
    assert [item["analysis_id"] for item in result["analyses"]] == ["numeric_distribution"]
    pooled = aggregate_site_results([result, {"site_id": "site_two", "analyses": []}], payload)
    assert len(pooled["analyses"]) == 1


def test_human_approved_dynamic_pairwise_algorithm_executes_locally_and_pools(tmp_path: Path) -> None:
    local_code = '''def run_local(data, analysis, min_cell_count):
    fields = analysis["fields"]
    pairs = []
    for i in range(len(fields)):
        for j in range(i, len(fields)):
            x = fields[i]
            y = fields[j]
            left = pd.to_numeric(data[x], errors="coerce")
            right = pd.to_numeric(data[y], errors="coerce")
            frame = pd.DataFrame({"x": left, "y": right}).dropna()
            n = int(len(frame))
            if n < min_cell_count:
                pairs.append({"x": x, "y": y, "status": "suppressed", "n": None})
            else:
                pairs.append({"x": x, "y": y, "status": "ok", "n": n, "sum_x": float(frame["x"].sum()), "sum_y": float(frame["y"].sum()), "sum_x2": float((frame["x"] * frame["x"]).sum()), "sum_y2": float((frame["y"] * frame["y"]).sum()), "sum_xy": float((frame["x"] * frame["y"]).sum())})
    return {"schema_version": "biobank.dynamic_local_output.v1", "status": "ok", "pairs": pairs}
'''
    server_code = '''def run_server(outputs, analysis):
    fields = analysis["fields"]
    matrices = {}
    pooled = {}
    for output in outputs:
        matrix = {}
        for field in fields:
            matrix[field] = {}
        for item in output.get("pairs", []):
            if item.get("status") != "ok":
                continue
            n = item["n"]
            numerator = n * item["sum_xy"] - item["sum_x"] * item["sum_y"]
            left = n * item["sum_x2"] - item["sum_x"] * item["sum_x"]
            right = n * item["sum_y2"] - item["sum_y"] * item["sum_y"]
            correlation = numerator / math.sqrt(left * right) if left > 0 and right > 0 else 1.0 if item["x"] == item["y"] else None
            matrix[item["x"]][item["y"]] = correlation
            matrix[item["y"]][item["x"]] = correlation
            key = item["x"] + "|" + item["y"]
            if key not in pooled:
                pooled[key] = {"x": item["x"], "y": item["y"], "n": 0, "sum_x": 0.0, "sum_y": 0.0, "sum_x2": 0.0, "sum_y2": 0.0, "sum_xy": 0.0}
            for name in ["n", "sum_x", "sum_y", "sum_x2", "sum_y2", "sum_xy"]:
                pooled[key][name] = pooled[key][name] + item[name]
        matrices[output["site_id"]] = matrix
    matrix = {}
    for field in fields:
        matrix[field] = {}
    for item in pooled.values():
        n = item["n"]
        numerator = n * item["sum_xy"] - item["sum_x"] * item["sum_y"]
        left = n * item["sum_x2"] - item["sum_x"] * item["sum_x"]
        right = n * item["sum_y2"] - item["sum_y"] * item["sum_y"]
        correlation = numerator / math.sqrt(left * right) if left > 0 and right > 0 else 1.0 if item["x"] == item["y"] else None
        matrix[item["x"]][item["y"]] = correlation
        matrix[item["y"]][item["x"]] = correlation
    matrices["federated"] = matrix
    return {"schema_version": "biobank.dynamic_server_output.v1", "status": "ok", "matrices": matrices, "method": "Pearson correlation from pooled pairwise sufficient statistics"}
'''
    payload = {
        "schema_version": "biobank.analysis_contract.v2",
        "study_id": "dynamic-correlation",
        "question": "Calculate a correlation matrix",
        "training_allowed": False,
        "cohorts": [],
        "site_harmonization": {},
        "dynamic_tools": [{
            "name": "dynamic_pairwise_correlation",
            "version": 1,
            "description": "Pairwise-complete Pearson correlation",
            "required_parameters": ["fields"],
            "local_output": "pairwise sufficient statistics",
            "server_operation": "sum pairwise moments and calculate correlations",
            "local_code": local_code,
            "server_code": server_code,
        }],
        "approved_tools": ["dynamic_pairwise_correlation"],
        "privacy": {"min_cell_count": 5, "forbidden_outputs": ["row_level"]},
        "analyses": [{
            "analysis_id": "correlations",
            "tool": "dynamic_pairwise_correlation",
            "fields": ["record_id", "measurement"],
        }],
        "unavailable_requests": [],
        "approval": {"status": "proposed"},
    }
    proposal = AnalysisContract.parse(payload)
    payload["approval"] = {"status": "approved", "contract_digest": proposal.approval_digest}
    approved = AnalysisContract.parse(payload)

    local = SiteExecutor(_site(tmp_path)).execute(approved)
    pooled = aggregate_site_results([local], approved.payload)

    matrix = pooled["analyses"][0]["output"]["matrices"]["federated"]
    assert matrix["record_id"]["measurement"] == pytest.approx(1.0)
    assert local["patient_rows_exported"] == 0


def test_dynamic_tool_rejects_imports() -> None:
    payload = _contract(approved=False).payload
    payload["dynamic_tools"] = [{
        "name": "dynamic_bad",
        "description": "unsafe",
        "required_parameters": ["fields"],
        "local_code": "import os\ndef run_local(data, analysis, min_cell_count):\n    return {}",
        "server_code": "def run_server(outputs, analysis):\n    return {}",
    }]
    with pytest.raises(ValueError, match="only def run_local|forbidden syntax"):
        AnalysisContract.parse(payload)


def test_server_planner_allows_two_schema_repair_attempts(tmp_path: Path) -> None:
    invalid = {
        "schema_version": "biobank.analysis_contract.v2",
        "study_id": "repair-km",
        "question": "Compare survival",
        "training_allowed": False,
        "cohorts": [{"name": "all", "predicate": {"not": {"is_missing": {"field": "group"}}}}],
        "site_harmonization": {},
        "approved_tools": ["federated_kaplan_meier"],
        "privacy": {"min_cell_count": 5, "forbidden_outputs": ["row_level"]},
        "analyses": [{
            "analysis_id": "survival",
            "tool": "federated_kaplan_meier",
            "time_field": "time",
            "event": "event",
            "group_by": "group",
            "time_bins": [0, 12, 24],
        }],
        "unavailable_requests": [],
        "approval": {"status": "proposed"},
    }
    valid = json.loads(json.dumps(invalid))
    valid["analyses"][0]["event"] = {"field": "event", "values": [1]}

    class RepairingPlanner(CodexPlanner):
        def __init__(self) -> None:
            self.calls = 0

        def run_json(self, prompt: str, work_dir: Path) -> dict:
            self.calls += 1
            return invalid if self.calls < 3 else valid

    planner = RepairingPlanner()
    contract = planner.plan("Compare survival", [], tmp_path)

    assert planner.calls == 3
    assert contract.payload["analyses"][0]["event"]["values"] == [1]
