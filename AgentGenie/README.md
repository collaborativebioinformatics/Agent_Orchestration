# AgentGenie

## Federated biobank workflow

This design illustrates privacy-preserving, agentic analysis across federated biobanks. It separates central coordination from local data access so that multiple institutions can contribute to a shared analysis without exchanging patient-level records.

### How the workflow operates

1. **Research question:** The researcher submits a scientific question and consents to catalog-only agent planning. The published tool registry defines the only algorithms that may later be approved.
2. **Local discovery and proposal:** Each site agent inspects its redacted catalog, declared mappings, and CSV header—never row values—to propose a local data adapter and report which requested analyses it can support.
3. **Server orchestration:** The server agent combines the site proposals, drafts a typed analysis contract, and produces a concise recommendation for the researcher.
4. **Human confirmation:** The researcher either approves the exact contract, rejects it, or confirms the server agent's revision points and optionally adds scientific guidance. A revision starts another discovery and planning pass. Approval is required even when every site reports full support.
5. **Analysis contract:** Each site receives the immutable, digest-bound specification covering the cohort, variables, harmonization, model, output granularity, disclosure rules, and approved tools.
6. **Local harmonization and analysis:** Behind each biobank's firewall, the site agent applies the approved adapter, runs only allowlisted tools, and enforces disclosure controls before releasing aggregate output.
7. **Aggregate analysis and review:** The server agent pools the permitted aggregates, renders the report and figures, and returns provenance, feasible-row summaries, and brief exclusion explanations for researcher review.

### Trust and data boundaries

Solid green arrows represent tasks, contracts, and approved tools travelling from the coordinator to each site. Dashed arrows represent aggregate-only results returning to the server. Patient-level records remain inside their source biobank, and biobanks never communicate directly with one another.

## Analysis-only prototype

This repository includes a runnable adaptation of the medical-imaging FedReady
pattern for tabular clinical and gene data. Codex is the only generative-agent
backend. Client-side Codex agents propose local adapters and assess support from
site-visible metadata. A server-side Codex agent turns those proposals into a
typed analysis contract and a concise human recommendation. Deterministic,
allowlisted statistical tools execute only the approved contract. There is no
model training.

The workflow deliberately stops before touching patient rows. The researcher
first sees the proposed cohort and fields, local adapter status, selected tools,
disclosure rules, unavailable concepts, and the server agent's recommendation.
The recommendation either presents an approvable contract or concise revision
points. Only a separate, digest-bound human decision creates an executable
contract; the headless `run` command rejects a proposed contract.

### General-purpose server tool registry

The statistical runtime contains no breast-cancer, TNBC, receptor, treatment,
or fixed-column logic. The server publishes a versioned registry of statistical
primitives to Codex:

- `federated_histogram`: fixed-bin numeric distributions and binned median intervals;
- `categorical_contingency`: pooled category-by-cohort tables and chi-square tests;
- `federated_kaplan_meier`: pooled event/censor histograms and product-limit curves;
- `numeric_sufficient_statistics`: pooled count, sum, sum-of-squares, mean, and SD; and
- `missingness_summary`: pooled observed and missing counts.

The Kaplan–Meier primitive generalizes the time-binned design in NVFlare's
Kaplan–Meier example. It accepts contract-defined time fields, event encodings,
subsets, grouping variables, and time bins; it knows nothing about a particular
disease or endpoint.

For every new user question, Codex must create a proposed v2 contract containing:

- named cohorts expressed with the safe predicate DSL (`all`, `any`, `not`,
  equality, membership, numeric comparisons, and missingness);
- per-site canonical field mappings, categorical value maps, and unit multipliers;
- selected registry tools with all required parameters;
- disclosure controls and requested-but-unavailable concepts; and
- an unapproved status for human review.

Sites expose only catalog metadata, declared mapping metadata, CSV column
headers, and the site agent's structured proposal during discovery. No row
values are read before approval. The server agent validates and promotes the
site adapters into the proposed contract, assesses cross-site feasibility, and
summarizes only the key decisions for the researcher while retaining the full
digest-bound review as an audit artifact. Once approved, the same generic local
and server implementations execute the task-specific contract. Registry
definitions are in `src/biobank_agent/tools/registry.py`; cohort evaluation,
harmonization, local aggregation, and server pooling are separate modules in
the same directory.

### TNBC demonstration contract

The included proposal compares triple-negative breast cancer (TNBC) with
non-TNBC by age and menopause, then describes overall survival among TNBC
patients by surgery, chemotherapy, hormone therapy, and radiotherapy. It also
pools sufficient statistics for a small gene panel. TNBC is explicitly defined
as ER-negative, PR-negative, and HER2-negative.

The current data do **not** contain weight, ethnicity/race, or a
progression/recurrence endpoint. The workflow reports those requests as
unsupported. It does not infer ethnicity and does not relabel overall survival
as progression-free survival. Site B's survival years are converted to months,
and Site C's binary ER coding is mapped to the common receptor representation.
Site C has clinical metadata but no gene-expression panel, so it participates
in clinical analyses only.

The TNBC JSON is a task-specific example produced on top of the general
registry. It supplies receptor-based cohort predicates, site-specific coding
and survival-unit mappings, selected fields, and analysis parameters. None of
those choices are embedded in the tools. Survival comparisons are observational
and unadjusted; they cannot establish which treatment causes longer survival.

### Run the approval workflow

From the `AgentGenie` directory, install the Python 3.9+ package. An authenticated Codex CLI is needed when
planning a new free-form question.

```bash
python3 -m pip install -e .
```

#### Recommended: one study console for question and approval

The local web console is the primary human interface. Start an NVFlare run with
`question=None`; the Controller enters `WAITING_FOR_QUESTION` instead of reading
a prompt file or command-line string:

```python
from biobank_agent.flare.job import run_biobank_simulation

run_biobank_simulation(
    workspace_root="workspace",
    sites_root="/path/to/sites",
    question=None,
    output_dir="runs",
    session_id="study-001",
)
```

In another terminal, point the study console at that same run directory. Supplying
the site and workspace locations enables the completed screen's **Start New** button:

```bash
biobank-agent serve-ui runs/study-001 \
  --sites-root /path/to/sites \
  --workspace workspace \
  --output-root runs
```

Open `http://127.0.0.1:8765`. The console provides the complete human flow:

1. enter the initial scientific question and researcher identity;
2. see connected clients and follow discovery, local adapter planning, contract
   planning, dispatch, local analysis, and aggregate return in the live graph;
3. review readable cohort predicates, harmonization-dependent feasibility,
   selected general-purpose tools and parameters, privacy rules, and unavailable
   requests;
4. review the server agent's concise recommendation;
5. approve or reject an executable contract, or confirm a proposed revision and
   optionally add manual scientific guidance for the next planning pass;
6. review the final figures and aggregate report together with site-level total
   and feasible-row counts, disclosure-controlled values, and a one-sentence
   explanation of exclusions; and
7. download the implementation benchmark bundle or use **Start New** to archive
   the current session and launch a clean study.

The human gate is never skipped. Full site support removes the need for another
revision, but the researcher must still approve the final contract before any
task capable of reading patient rows is dispatched.

The browser never changes Controller state directly. It writes typed user-input
and decision artifacts; the Controller validates their schema, workflow state,
and proposal digest before proceeding. The server binds only to loopback,
requires a per-process request token, applies a restrictive content security
policy, and exposes no patient data. JSON remains the durable audit format but
is no longer the researcher-facing interface.

#### Live status reporting

Status reporting uses a separate, non-sensitive control path:

1. On run start and completion, and around catalog inspection, site-agent
   planning, and approved analysis, each NVFlare Executor sends a best-effort
   auxiliary message containing only an allowlisted progress code.
2. The Controller verifies that the sender is one of the configured clients,
   maps the code to a safe phase and message, and appends it to
   `server/workflow_events.json`. Server-side planning, review synthesis,
   approval, dispatch, aggregation, failure, and completion events are written
   through the same audit stream.
3. `server/participants.json` supplies the expected client identities. Connect
   and disconnect events determine the **Connected clients** count.
4. The browser polls `/api/study` once per second and renders a compact dynamic
   graph. Arrows show task dispatch or aggregate return, while only a site agent
   doing local work is illuminated. The server remains visually neutral because
   coordination is its default role.

These events contain phase, actor, site identifier, status, timestamp, and a
safe message. They never contain patient values, row payloads, or analysis
results. Auxiliary reporting is optional and must not block the federated task;
the durable Controller state remains authoritative.

The commands below remain useful as headless/automation alternatives.

Ask Codex to draft a proposal from catalogs only (no patient rows are opened):

```bash
biobank-agent plan \
  --question "Compare TNBC with non-TNBC and describe TNBC survival by treatment" \
  --sites-root /path/to/sites \
  --output runs/tnbc.proposed.json
```

Or inspect the supplied reproducible proposal:

```bash
biobank-agent review examples/tnbc_contract.proposed.json \
  --sites-root /path/to/sites
```

After the researcher agrees to the displayed tools and feasibility report,
record approval and run that contract:

```bash
biobank-agent approve examples/tnbc_contract.proposed.json \
  --by researcher \
  --output runs/tnbc.approved.json

biobank-agent run \
  --contract runs/tnbc.approved.json \
  --sites-root /path/to/sites \
  --output runs/tnbc
```

Output contains one aggregate-only JSON payload per site, a pooled server
payload, a Markdown report, and SVG figures. Raw rows and patient identifiers
are never written to the run directory.

### Implementation benchmark bundle

After a successful NVFlare study, AgentGenie automatically creates
`server/benchmark_bundle/` and `server/benchmark_bundle.zip` inside that
session's run directory. The completed-study UI exposes the ZIP as **Download
implementation benchmark bundle**.

Each bundle is scoped to one completed session and never combines artifacts
from older, failed, rejected, or active runs. For retro-inspection, use the
latest successful session's bundle.

The bundle is designed for comparison with an implementation prepared by a
human data scientist. It contains:

- one record per site with the agent-proposed local adapter and the
  contract-authorized cleaning/harmonization mapping;
- the approved analysis contract, feasibility report, human decision, and
  tool registry;
- the generated NVFlare job metadata, server/client configurations, and exact
  deployed analysis code;
- aggregate-only results, report, and figures; and
- a manifest plus SHA-256 checksums.

Site CSVs, patient-level data, and catalog snapshots are explicitly excluded.

### NVFlare Controller workflow

The production-shaped path lives under `biobank_agent.flare`. Its
`BiobankAnalysisController` owns the state transition rather than relying on a
wrapper script:

```text
STARTING -> WAITING_FOR_QUESTION -> DISCOVERING -> PLANNING
                                      ^              |
                                      |              v
                                      +---- REVISE -- WAITING_FOR_APPROVAL
                                                        |
                                                     APPROVED
                                                        |
                                                        v
                                             DISPATCHING_ANALYSIS
                                                        |
                                                        v
                                                     COMPLETED

Any active phase may terminate as FAILED, REJECTED, EXPIRED, or ABORTED.
```

During discovery the Controller broadcasts `biobank_catalog_discovery`; its
Executors read each site's redacted `catalog.json`, declared mappings, and only
the header of `data.csv`. A local Codex site agent proposes the adapter and
analysis support without reading row values. The Controller then asks the
server Codex agent for a contract (or loads a reproducible unapproved proposal),
validates the site proposals, writes the proposal and feasibility report, and
synthesizes a concise human recommendation before entering
`WAITING_FOR_APPROVAL`. No task capable of reading patient rows has been sent at
that point.

If revision is recommended, the UI shows only the concise key points. The full
server-authored guidance is retained under the hood and bound by its digest.
When the researcher confirms it, any additional manual guidance is appended,
the current attempt is archived, and all agents run another planning pass. If
the resulting proposal is supported, the workflow returns to the mandatory
approval gate without requesting an unnecessary further revision.

While the NVFlare run waits, the researcher reviews those artifacts in another
terminal and records a decision:

```bash
biobank-agent approve-run runs/<session-id> \
  --decision approve \
  --by researcher
```

The command displays the cohort, proposed tools, site support, and unavailable
requests, then requires the researcher to type `APPROVE`. `--yes` is available
for an explicitly authorized non-interactive workflow. Rejection uses
`--decision reject --reason "..."`.

The approval contains the published proposal digest. The Controller validates
that digest and the complete client/server review bundle, creates the immutable
approved contract, and only then broadcasts `biobank_approved_analysis`. Any
proposal edit, stale decision, timeout, rejection, or abort prevents analysis
dispatch. Site Executors independently validate the approval and contract
digest before opening their CSV and return only disclosure-controlled aggregate
payloads. The server pools those results, renders the final report and curves,
and records how many local rows were feasible; small site counts may be shown as
**Minimal** rather than disclosed exactly.

Build a job in Python using the current NVFlare `FedJob`/`Recipe` pattern:

```python
from biobank_agent.flare.job import build_biobank_recipe

recipe, clients = build_biobank_recipe(
    sites_root="/path/to/sites",
    question="Compare TNBC with non-TNBC and describe TNBC survival by treatment",
    output_dir="runs",
    session_id="tnbc-study-001",
    proposal_path="examples/tnbc_contract.proposed.json",  # omit to use Codex
)
```

Install against an NVFlare 2.9 environment with `pip install -e '.[nvflare]'`.
The Controller, client Executor, approval state machine, and job builder are in
`src/biobank_agent/flare/`.

### Timing-compressed workflow demo

[![Watch the AgentGenie workflow demo](docs/demo-preview.png)](demo-video/agentgenie-workflow-demo-clean-final.webm)

Click the preview to watch the AgentGenie workflow video. It replays a real
analysis contract and aggregate result with compressed agent wait times and is
visibly labelled as a recorded replay. The demonstration includes the initial
question, client feasibility review, researcher-supplied revision guidance, a
supported second pass, human approval, disclosure-controlled feasible-case
reporting, and the final Kaplan–Meier curve.
