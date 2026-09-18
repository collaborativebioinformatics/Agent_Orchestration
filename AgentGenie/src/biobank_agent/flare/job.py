"""Build the NVFlare FedJob for the human-approved biobank workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from biobank_agent.benchmark import create_benchmark_bundle
from biobank_agent.flare import ANALYSIS_TASK, CATALOG_TASK
from biobank_agent.flare.controller import BiobankAnalysisController
from biobank_agent.flare.executor import BiobankSiteExecutor

from nvflare.job_config.api import FedJob
from nvflare.recipe import SimEnv
from nvflare.recipe.spec import Recipe


class BiobankAnalysisRecipe(Recipe):
    @property
    def job(self) -> FedJob:
        return self._job


def build_biobank_job(
    *,
    sites_root: str | Path,
    question: str | None = None,
    output_dir: str | Path = "runs",
    session_id: str = "biobank-study",
    proposal_path: str | Path | None = None,
    job_name: str = "biobank_agentic_analysis",
    **controller_options: Any,
) -> tuple[FedJob, list[str]]:
    root = Path(sites_root).resolve()
    site_dirs = sorted(path.parent for path in root.glob("*/catalog.json"))
    client_ids = [path.name for path in site_dirs]
    if not client_ids:
        raise ValueError(f"No site catalogs found under {root}")
    job = FedJob(name=job_name, min_clients=len(client_ids), mandatory_clients=client_ids)
    controller = BiobankAnalysisController(
        question=question,
        client_ids=client_ids,
        output_dir=str(Path(output_dir).resolve()),
        session_id=session_id,
        initial_proposal=(
            json.loads(Path(proposal_path).read_text(encoding="utf-8")) if proposal_path is not None else None
        ),
        **controller_options,
    )
    job.to_server(controller, id="biobank_analysis_controller")
    for client_id, site_dir in zip(client_ids, site_dirs):
        job.to(
            BiobankSiteExecutor(site_dir=str(site_dir)),
            client_id,
            id="biobank_site_executor",
            tasks=[CATALOG_TASK, ANALYSIS_TASK],
        )
    return job, client_ids


def build_biobank_recipe(**kwargs: Any) -> tuple[BiobankAnalysisRecipe, list[str]]:
    job, client_ids = build_biobank_job(**kwargs)
    return BiobankAnalysisRecipe(job), client_ids


def run_biobank_simulation(
    *,
    workspace_root: str | Path,
    threads: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Launch the Controller/Executor workflow through Recipe + SimEnv."""
    recipe, client_ids = build_biobank_recipe(**kwargs)
    environment = SimEnv(
        clients=client_ids,
        num_threads=threads,
        workspace_root=str(Path(workspace_root).resolve()),
        log_config="concise",
    )
    run = recipe.execute(environment)
    workspace = run.get_result(clean_up=False)
    output_dir = Path(kwargs.get("output_dir", "runs")).resolve()
    session_id = str(kwargs.get("session_id", "biobank-study"))
    job_name = str(kwargs.get("job_name", "biobank_agentic_analysis"))
    run_dir = output_dir / session_id
    state_path = run_dir / "server" / "workflow_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    benchmark_archive = (
        create_benchmark_bundle(run_dir, workspace_root=workspace_root, job_name=job_name)
        if state.get("status") == "COMPLETED"
        else None
    )
    return {
        "schema_version": "biobank.nvflare_recipe_run.v1",
        "job_id": run.get_job_id(),
        "clients": client_ids,
        "workspace": workspace,
        "benchmark_bundle": str(benchmark_archive) if benchmark_archive is not None else None,
    }
