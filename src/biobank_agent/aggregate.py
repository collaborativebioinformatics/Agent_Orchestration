"""Server dispatch into the general-purpose aggregate tool registry."""

from __future__ import annotations

from typing import Any

from biobank_agent.tools.server import aggregate_tool
from biobank_agent.tools.dynamic import validate_dynamic_tools


def aggregate_site_results(results: list[dict[str, Any]], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    if contract is None:
        raise ValueError("The approved contract is required to aggregate named analyses")
    dynamic_tools = validate_dynamic_tools(contract.get("dynamic_tools", []))
    pooled = []
    for index, analysis in enumerate(contract["analyses"]):
        analysis_id = analysis.get("analysis_id", f"analysis_{index + 1}")
        outputs = []
        target_site = analysis.get("site_id")
        expected_results = [result for result in results if not isinstance(target_site, str) or result.get("site_id") == target_site]
        if isinstance(target_site, str) and not expected_results:
            raise ValueError(f"Target site {target_site} did not return a result for {analysis_id}")
        for result in expected_results:
            matching = [item for item in result["analyses"] if item["analysis_id"] == analysis_id]
            if len(matching) != 1:
                raise ValueError(f"Site {result.get('site_id')} returned {len(matching)} outputs for {analysis_id}")
            outputs.append(matching[0]["output"])
        pooled.append(
            {
                "analysis_id": analysis_id,
                "tool": analysis["tool"],
                "specification": analysis,
                "output": aggregate_tool(analysis, outputs, dynamic_tools),
            }
        )
    return {
        "schema_version": "biobank.server_aggregate.v2",
        "site_count": len(results),
        "patient_rows_received": 0,
        "site_row_summaries": [
            {
                "site_id": result.get("site_id"),
                "total_rows": result.get("local_total_rows"),
                "analysis_row_counts": result.get("analysis_row_counts", {}),
                "exclusion_reason": result.get("row_exclusion_reason"),
            }
            for result in results
        ],
        "analyses": pooled,
    }
