"""Create a privacy-safe, run-scoped bundle for human implementation review."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUNDLE_SCHEMA = "biobank.implementation_benchmark_bundle.v1"


def create_benchmark_bundle(
    run_dir: str | Path,
    *,
    workspace_root: str | Path,
    job_name: str = "biobank_agentic_analysis",
) -> Path:
    """Package adapters and the executed NVFlare job without patient-level data."""
    run_dir = Path(run_dir).resolve()
    server_dir = run_dir / "server"
    state = _read_json(server_dir / "workflow_state.json")
    if state.get("status") != "COMPLETED":
        raise RuntimeError("A benchmark bundle may be created only for a completed study")

    required = {
        "analysis_contract.approved.json": server_dir / "analysis_contract.approved.json",
        "feasibility_report.json": server_dir / "feasibility_report.json",
        "human_approval.json": server_dir / "human_approval.json",
        "tool_registry.json": server_dir / "tool_registry.json",
    }
    aggregate_path = server_dir / "server_aggregate.json"
    missing = [name for name, path in required.items() if not path.is_file()]
    if not aggregate_path.is_file():
        missing.append("server_aggregate.json")
    if missing:
        raise FileNotFoundError(f"Completed run is missing benchmark inputs: {', '.join(missing)}")

    job_root = Path(workspace_root).resolve() / job_name
    server_job = job_root / "server" / "simulate_job"
    if not server_job.is_dir():
        raise FileNotFoundError(f"Generated NVFlare job is unavailable at {server_job}")

    temporary = server_dir / ".benchmark_bundle.tmp"
    bundle = server_dir / "benchmark_bundle"
    archive = server_dir / "benchmark_bundle.zip"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    contract = _read_json(required["analysis_contract.approved.json"])
    feasibility = _read_json(required["feasibility_report.json"])
    sites = []
    adapters_dir = temporary / "site_adapters"
    adapters_dir.mkdir()
    harmonization = contract.get("site_harmonization", {})
    for site in feasibility.get("sites", []):
        site_id = str(site.get("site_id", "")).strip()
        if not site_id or Path(site_id).name != site_id:
            raise ValueError(f"Invalid site identifier in feasibility report: {site_id!r}")
        sites.append(site_id)
        adapter = {
            "schema_version": "biobank.benchmark_site_adapter.v1",
            "site_id": site_id,
            "agent_proposed_adapter": site.get("data_adapter", {}),
            "agent_analysis_proposal": site.get("client_agent_proposal", {}),
            "executed_harmonization": harmonization.get(site_id, {}),
            "all_requested_analyses_supported": site.get("all_requested_analyses_supported"),
        }
        _write_json(adapters_dir / f"{site_id}.json", adapter)

    inputs_dir = temporary / "approved_study"
    inputs_dir.mkdir()
    for name, source in required.items():
        shutil.copy2(source, inputs_dir / name)

    results_dir = temporary / "results"
    results_dir.mkdir()
    shutil.copy2(aggregate_path, results_dir / "server_aggregate.json")
    report_dir = server_dir / "report"
    if not report_dir.is_dir():
        raise FileNotFoundError("Completed run has no report directory")
    shutil.copytree(report_dir, results_dir / "report")

    nvflare_dir = temporary / "nvflare_job"
    nvflare_dir.mkdir()
    shutil.copy2(server_job / "meta.json", nvflare_dir / "meta.json")
    _copy_if_present(
        server_job / "app_server" / "config" / "config_fed_server.json",
        nvflare_dir / "server" / "config_fed_server.json",
    )
    _copy_tree_if_present(
        server_job / "app_server" / "custom" / "biobank_agent",
        nvflare_dir / "server" / "deployed_biobank_agent",
    )
    for site_id in sites:
        client_job = job_root / site_id / "simulate_job" / f"app_{site_id}"
        _copy_if_present(
            client_job / "config" / "config_fed_client.json",
            nvflare_dir / "clients" / site_id / "config_fed_client.json",
        )
        _copy_tree_if_present(
            client_job / "custom" / "biobank_agent",
            nvflare_dir / "clients" / site_id / "deployed_biobank_agent",
        )

    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "session_id": run_dir.name,
        "study_id": contract.get("study_id"),
        "contract_digest": state.get("contract_digest"),
        "sites": sites,
        "contents": {
            "site_adapters": "Agent-proposed adapters and contract-authorized harmonization, one file per site.",
            "approved_study": "Approved contract, feasibility, human decision, and tool registry.",
            "nvflare_job": "Generated job metadata, configs, and exact deployed server/client implementation.",
            "results": "Aggregate-only output, report, and figures.",
        },
        "privacy": {
            "patient_level_data_included": False,
            "site_csv_included": False,
            "catalog_snapshot_included": False,
        },
    }
    _write_json(temporary / "manifest.json", manifest)
    (temporary / "README.md").write_text(_bundle_readme(), encoding="utf-8")
    _write_json(temporary / "checksums.sha256.json", _checksums(temporary))

    if bundle.exists():
        shutil.rmtree(bundle)
    temporary.rename(bundle)
    archive.unlink(missing_ok=True)
    created_archive = Path(shutil.make_archive(str(archive.with_suffix("")), "zip", bundle))
    return created_archive


def _copy_if_present(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Generated NVFlare asset is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree_if_present(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Generated NVFlare implementation is missing: {source}")
    shutil.copytree(source, destination)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checksums(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256.json"
    }


def _bundle_readme() -> str:
    return """# AgentGenie implementation benchmark bundle

This bundle captures the two implementation surfaces intended for comparison
with a human data scientist:

1. `site_adapters/` contains each site agent's local cleaning/harmonization
   proposal and the mapping authorized in the approved contract.
2. `nvflare_job/` contains the generated NVFlare metadata, configuration, and
   exact deployed server/client analysis implementation.

`approved_study/` provides the task and approval context. `results/` contains
aggregate-only outputs and the final report. No site CSV or patient-level data
is included. Verify file integrity with `checksums.sha256.json`.
"""
