"""Command line workflow with a mandatory user-approval boundary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from biobank_agent.aggregate import aggregate_site_results
from biobank_agent.codex import CodexPlanner
from biobank_agent.contracts import AnalysisContract
from biobank_agent.feasibility import assess_feasibility
from biobank_agent.render import render_report
from biobank_agent.site import SiteExecutor


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _site_dirs(root: Path) -> list[Path]:
    paths = sorted(path.parent for path in root.glob("*/catalog.json"))
    if not paths:
        raise FileNotFoundError(f"No site catalog.json files found under {root}")
    return paths


def _catalogs(root: Path) -> list[dict[str, Any]]:
    catalogs = []
    for path in _site_dirs(root):
        catalog = _read_json(path / "catalog.json")
        catalog["schema_fields"] = list(pd.read_csv(path / "data.csv", nrows=0).columns)
        mapping_path = path / "mappings.yaml"
        catalog["declared_mappings_yaml"] = mapping_path.read_text(encoding="utf-8") if mapping_path.exists() else ""
        catalogs.append(catalog)
    return catalogs


def command_plan(args: argparse.Namespace) -> None:
    sites_root = Path(args.sites_root)
    output = Path(args.output)
    catalogs = _catalogs(sites_root)
    planner = CodexPlanner(binary=args.codex_binary, model=args.model)
    contract = planner.plan(args.question, catalogs, output.parent / ".codex-plan")
    feasibility = assess_feasibility(contract, catalogs)
    _write_json(output, contract.payload)
    _write_json(output.with_suffix(".feasibility.json"), feasibility)
    print_review(contract, feasibility)
    print(f"\nProposal written to {output}. No patient data was accessed and no analysis was run.")
    print(f"Review it, then approve with: biobank-agent approve {output}")


def command_review(args: argparse.Namespace) -> None:
    contract = AnalysisContract.parse(_read_json(Path(args.contract)))
    feasibility = assess_feasibility(contract, _catalogs(Path(args.sites_root)))
    print_review(contract, feasibility)


def print_review(contract: AnalysisContract, feasibility: dict[str, Any]) -> None:
    print("\n=== USER CONFIRMATION REQUIRED ===")
    print(f"Study: {contract.study_id}")
    print(f"Cohort: {contract.payload.get('cohort_definition')}")
    print("Proposed tools:")
    for tool in contract.payload["approved_tools"]:
        print(f"  - {tool}")
    print("Data support:")
    for site in feasibility["sites"]:
        print(f"  - {site['site_id']}: {len(site['supported'])} supported, {len(site['unsupported'])} unsupported")
    for item in feasibility.get("unavailable_requests", []):
        if isinstance(item, dict):
            print(f"  - unavailable: {item.get('concept')} ({item.get('reason')})")
        else:
            print(f"  - unavailable: {item}")
    print("Nothing executes until this exact contract is approved.")


def command_approve(args: argparse.Namespace) -> None:
    source = Path(args.contract)
    payload = _read_json(source)
    contract = AnalysisContract.parse(payload)
    if contract.is_approved:
        raise ValueError("Contract is already approved")
    payload["approval"] = {
        "status": "approved",
        "approved_by": args.by,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "confirmation": "I approve the cohort definition, tools, fields, and disclosure policy in this contract.",
        "contract_digest": contract.approval_digest,
    }
    approved = AnalysisContract.parse(payload)
    destination = Path(args.output) if args.output else source.with_name(source.stem + ".approved.json")
    _write_json(destination, approved.payload)
    print(f"Approved immutable-input contract written to {destination}")


def command_run(args: argparse.Namespace) -> None:
    contract = AnalysisContract.parse(_read_json(Path(args.contract)))
    contract.require_approval()
    site_dirs = _site_dirs(Path(args.sites_root))
    run_dir = Path(args.output)
    site_results = []
    for site_dir in site_dirs:
        result = SiteExecutor(site_dir).execute(contract)
        _write_json(run_dir / "site_aggregates" / f"{site_dir.name}.json", result)
        site_results.append(result)
    aggregate = aggregate_site_results(site_results, contract.payload)
    aggregate["contract_digest"] = contract.digest
    _write_json(run_dir / "server_aggregate.json", aggregate)
    artifacts = render_report(aggregate, contract.payload, run_dir)
    print(f"Completed aggregate-only analysis across {len(site_dirs)} sites.")
    print(f"Patient rows received by server: {aggregate['patient_rows_received']}")
    print("Artifacts:")
    for artifact in artifacts:
        print(f"  - {artifact}")


def command_approve_run(args: argparse.Namespace) -> None:
    from biobank_agent.flare.approval import HumanApprovalGate, create_human_decision

    run_dir = Path(args.run_dir)
    gate = HumanApprovalGate(run_dir)
    if not gate.proposal_path.exists() or not gate.feasibility_path.exists():
        raise FileNotFoundError("The Controller has not published a proposal and feasibility report")
    state = _read_json(gate.state_path)
    if state.get("status") != "WAITING_FOR_APPROVAL":
        raise RuntimeError(f"Workflow is not waiting for approval (status={state.get('status')})")
    if gate.approval_path.exists():
        raise FileExistsError(f"A human decision already exists at {gate.approval_path}")
    contract = AnalysisContract.parse(_read_json(gate.proposal_path))
    feasibility = _read_json(gate.feasibility_path)
    print_review(contract, feasibility)
    if args.decision == "approve" and not args.yes:
        confirmation = input("Type APPROVE to authorize NVFlare analysis dispatch: ").strip()
        if confirmation != "APPROVE":
            raise RuntimeError("Approval was not confirmed")
    decision = create_human_decision(
        gate.proposal_path,
        decision=args.decision,
        approved_by=args.by,
        reason=args.reason,
    )
    HumanApprovalGate._write(gate.approval_path, decision)
    print(f"Human decision written to {gate.approval_path}; the Controller will validate its digest.")


def command_serve_ui(args: argparse.Namespace) -> None:
    from biobank_agent.ui.server import serve

    serve(
        Path(args.run_dir),
        host=args.host,
        port=args.port,
        sites_root=Path(args.sites_root) if args.sites_root else None,
        workspace_root=Path(args.workspace) if args.workspace else None,
        output_root=Path(args.output_root) if args.output_root else None,
    )


def command_run_simulation(args: argparse.Namespace) -> None:
    from biobank_agent.flare.job import run_biobank_simulation

    result = run_biobank_simulation(
        workspace_root=args.workspace,
        sites_root=args.sites_root,
        question=args.question,
        output_dir=args.output,
        session_id=args.session_id,
        proposal_path=args.proposal,
        threads=args.threads,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    plan = sub.add_parser("plan", help="ask Codex for a proposal; does not access patient data")
    plan.add_argument("--question", required=True)
    plan.add_argument("--sites-root", required=True)
    plan.add_argument("--output", default="runs/proposed-contract.json")
    plan.add_argument("--codex-binary", default="codex")
    plan.add_argument("--model")
    plan.set_defaults(func=command_plan)
    review = sub.add_parser("review", help="show tools and catalog-only feasibility")
    review.add_argument("contract")
    review.add_argument("--sites-root", required=True)
    review.set_defaults(func=command_review)
    approve = sub.add_parser("approve", help="record explicit user approval as a new contract")
    approve.add_argument("contract")
    approve.add_argument("--by", default="researcher")
    approve.add_argument("--output")
    approve.set_defaults(func=command_approve)
    run = sub.add_parser("run", help="execute a previously approved contract")
    run.add_argument("--contract", required=True)
    run.add_argument("--sites-root", required=True)
    run.add_argument("--output", default="runs/latest")
    run.set_defaults(func=command_run)
    approve_run = sub.add_parser("approve-run", help="approve or reject a waiting NVFlare Controller run")
    approve_run.add_argument("run_dir")
    approve_run.add_argument("--decision", choices=["approve", "reject"], required=True)
    approve_run.add_argument("--by", default="researcher")
    approve_run.add_argument("--reason")
    approve_run.add_argument("--yes", action="store_true", help="non-interactive explicit confirmation")
    approve_run.set_defaults(func=command_approve_run)
    ui = sub.add_parser("serve-ui", help="serve the local question-and-approval study console")
    ui.add_argument("run_dir")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--sites-root", help="site directory used when Start New launches another simulation")
    ui.add_argument("--workspace", help="NVFlare workspace used when Start New launches another simulation")
    ui.add_argument("--output-root", help="parent directory for new study sessions; defaults to the current run parent")
    ui.set_defaults(func=command_serve_ui)
    simulation = sub.add_parser("run-simulation", help="launch the NVFlare Controller and site Executors")
    simulation.add_argument("--sites-root", required=True)
    simulation.add_argument("--output", default="runs")
    simulation.add_argument("--workspace", default="workspace")
    simulation.add_argument("--session-id", required=True)
    simulation.add_argument("--question", help="omit to collect the question through the study console")
    simulation.add_argument("--proposal", help="optional reproducible unapproved proposal matching the question")
    simulation.add_argument("--threads", type=int)
    simulation.set_defaults(func=command_run_simulation)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
