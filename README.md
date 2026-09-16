# Agent Orchestration

## Federated biobank workflow

This design targets privacy-preserving, agentic analysis across federated biobanks. A researcher defines the scientific question and approves the available tools. The server agent then orchestrates the work, contracts each site-specific task, and analyzes the returned aggregates. Within each biobank boundary, a site agent discovers eligible data, harmonizes it, and runs only user-approved analysis tools. Patient-level records never leave their source biobank, and sites do not communicate directly with one another.

![Agentic data analysis across federated biobanks](assets/biobank-agentic-workflow.png)

### Initial prompt

> A workflow chart for an agentic data analysis workflow for a biobank task, similar to what we have in [this reference design](https://claude.ai/design/p/9888c087-a639-48eb-b042-e859507cc22a?via=share). The server agent will orchestrate, contract, and analyze the returned analysis results, while the site agent will do data discovery, data harmonization, and perform analysis with particular tools validated and approved by the user.

### Figure description

The chart is a 1660×1320 artboard: researcher and approved-tool registry on top; **server agent** (orchestrate → contract → analyze returns, with a re-contract feedback loop); three **site agents** (discovery → harmonization → analysis with allowlisted tools); and the biobanks they sit in front of. Solid green arrows show tasks and contracts travelling down, dashed arrows show aggregates only coming back up, and no-cross-site markers reinforce the separation between biobanks.
