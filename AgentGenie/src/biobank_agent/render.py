"""Task-neutral SVG and Markdown reporting for registered tool outputs."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

COLORS = ["#76b900", "#00c9b7", "#f2a900", "#9b7ede", "#e66b5b"]


def render_report(aggregate: dict[str, Any], contract: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []
    lines = [
        f"# Federated analysis: {contract['study_id']}",
        "",
        f"**Question:** {contract['question']}",
        "",
        f"Results combine disclosure-controlled aggregates from {aggregate['site_count']} sites; no patient rows were received.",
        "",
        "## Findings",
        "",
    ]
    for item in aggregate["analyses"]:
        tool, spec, result = item["tool"], item["specification"], item["output"]
        analysis_id = item["analysis_id"]
        title = str(spec.get("title") or analysis_id.replace("_", " ").title())
        lines.extend([f"### {title}", ""])
        if tool == "federated_histogram" and result.get("status") == "ok":
            path = output_dir / f"{analysis_id}.svg"
            _histogram_svg(result, path, title=title, x_label=spec["field"])
            figures.append(path)
            for name, values in result["groups"].items():
                lo, hi = values["median_interval"]
                lines.append(f"- {name}: n={values['n']}; binned median interval [{lo}, {hi}).")
            lines.extend(["", f"![{title}]({path.name})", ""])
        elif tool == "categorical_contingency" and result.get("status") == "ok":
            path = output_dir / f"{analysis_id}.svg"
            _bar_svg(result, path, title=title)
            figures.append(path)
            p_value = result["test"].get("p_value")
            lines.append(f"Pearson chi-square p={p_value:.4g}." if isinstance(p_value, float) else "Association test was not estimable after suppression.")
            lines.extend(["This is an unadjusted association, not evidence of causation.", "", f"![{title}]({path.name})", ""])
        elif tool == "federated_kaplan_meier" and result.get("status") == "ok":
            path = output_dir / f"{analysis_id}.svg"
            _km_svg(result, path, title=title, x_label=spec["time_field"])
            figures.append(path)
            lines.extend(
                [
                    "Curves use contract-defined time bins and pooled event/censor histograms. Group comparisons are descriptive and unadjusted.",
                    "",
                    f"![{title}]({path.name})",
                    "",
                ]
            )
        elif tool == "numeric_sufficient_statistics" and result.get("status") == "ok":
            for field, field_result in result["fields"].items():
                summaries = "; ".join(
                    f"{name}: mean={_number(values['mean'])}, SD={_number(values['standard_deviation'])}, n={values['n']}"
                    for name, values in field_result["groups"].items()
                )
                lines.append(f"- {field}: {summaries}")
            lines.append("")
        elif tool == "missingness_summary" and result.get("status") == "ok":
            for field, values in result["fields"].items():
                lines.append(f"- {field}: {_number(values['missing_fraction'])} missing ({values['missing']} of {values['missing'] + values['observed']}).")
            lines.append("")
        else:
            lines.extend([f"Not estimable: {result.get('status', 'unknown status')}.", ""])
    unavailable = contract.get("unavailable_requests", [])
    if unavailable:
        lines.extend(["## Requested concepts not supported by these data", ""])
        for item in unavailable:
            lines.append(f"- **{item.get('concept')}** — {item.get('reason')}" if isinstance(item, dict) else f"- {item}")
        lines.append("")
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "Cohorts, harmonization, endpoints, groups, and bins are defined by the approved contract. Small local cells are suppressed. Kaplan–Meier curves are binned and descriptive; observational comparisons are not causal or clinical advice.",
        ]
    )
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [report_path, *figures]


def _number(value: Any) -> str:
    return "NA" if value is None else f"{value:.3f}"


def _svg_frame(title: str, body: str, width: int = 900, height: int = 540) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#111"/>'
        f'<text x="55" y="45" fill="#f5f5f5" font-family="sans-serif" font-size="24" font-weight="700">{html.escape(title)}</text>'
        f"{body}</svg>"
    )


def _histogram_svg(result: dict[str, Any], path: Path, title: str, x_label: str) -> None:
    bins = result["bins"]
    maximum = max(max(group["counts"]) for group in result["groups"].values()) or 1
    width = min(34, 150 // max(len(result["groups"]), 1))
    step = 740 / max(len(bins) - 1, 1)
    parts = []
    for group_index, (name, group) in enumerate(result["groups"].items()):
        for index, count in enumerate(group["counts"]):
            x = 70 + index * step + group_index * width
            height = 350 * count / maximum
            parts.append(f'<rect x="{x:.1f}" y="{450-height:.1f}" width="{width-2}" height="{height:.1f}" fill="{COLORS[group_index % len(COLORS)]}"/>')
        parts.append(f'<text x="{620}" y="{75+group_index*20}" fill="{COLORS[group_index % len(COLORS)]}" font-family="sans-serif">{html.escape(name)}</text>')
    for index, value in enumerate(bins[:-1]):
        parts.append(f'<text x="{70+index*step:.1f}" y="478" fill="#aaa" font-family="sans-serif" font-size="11">{value}</text>')
    parts.append(f'<text x="380" y="520" fill="#ccc" font-family="sans-serif">{html.escape(x_label)}</text>')
    path.write_text(_svg_frame(title, "".join(parts)), encoding="utf-8")


def _bar_svg(result: dict[str, Any], path: Path, title: str) -> None:
    cells, cohorts = result["cells"], result["cohorts"]
    values = [value for cell in cells for value in cell["counts"].values() if value is not None]
    maximum = max(values, default=1)
    step = 760 / max(len(cells), 1)
    width = min(42, (step - 10) / max(len(cohorts), 1))
    parts = []
    for index, cell in enumerate(cells):
        x = 80 + index * step
        for cohort_index, cohort in enumerate(cohorts):
            value = cell["counts"].get(cohort)
            height = 0 if value is None else 340 * value / maximum
            parts.append(f'<rect x="{x+cohort_index*width:.1f}" y="{440-height:.1f}" width="{width-2:.1f}" height="{height:.1f}" fill="{COLORS[cohort_index % len(COLORS)]}"/>')
        parts.append(f'<text x="{x:.1f}" y="475" fill="#ccc" font-family="sans-serif" font-size="12">{html.escape(cell["category"])}</text>')
    for index, cohort in enumerate(cohorts):
        parts.append(f'<text x="620" y="{75+index*20}" fill="{COLORS[index % len(COLORS)]}" font-family="sans-serif">{html.escape(cohort)}</text>')
    path.write_text(_svg_frame(title, "".join(parts)), encoding="utf-8")


def _km_svg(result: dict[str, Any], path: Path, title: str, x_label: str) -> None:
    maximum_time = max(result["bins"])
    parts = ['<line x1="70" y1="460" x2="840" y2="460" stroke="#777"/><line x1="70" y1="80" x2="70" y2="460" stroke="#777"/>']
    for index, (name, group) in enumerate(result["groups"].items()):
        points = []
        for point in group["curve"]:
            x = 70 + 770 * point["time"] / maximum_time
            y = 460 - 380 * point["survival"]
            points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[index % len(COLORS)]}" stroke-width="3"/>')
        parts.append(f'<text x="620" y="{75+index*20}" fill="{COLORS[index % len(COLORS)]}" font-family="sans-serif">{html.escape(name)} (n={group["n"]})</text>')
    parts.append(f'<text x="380" y="515" fill="#ccc" font-family="sans-serif">{html.escape(x_label)}</text><text x="12" y="270" fill="#ccc" font-family="sans-serif" transform="rotate(-90 12 270)">Survival probability</text>')
    path.write_text(_svg_frame(title, "".join(parts)), encoding="utf-8")
