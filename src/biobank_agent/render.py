"""Task-neutral SVG and Markdown reporting for registered tool outputs."""

from __future__ import annotations

import html
import json
import math
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
        elif tool.startswith("dynamic_") and (
            result.get("status") == "ok" or result.get("schema_version") == "biobank.dynamic_server_output.v1"
        ):
            lines.extend(
                [
                    "This result was produced by agent-generated local and server code that was included in the human-approved contract.",
                    "",
                ]
            )
            matrices = _dynamic_matrices(result)
            if matrices:
                for matrix_name, fields, matrix in matrices:
                    matrix_title = str(matrix_name).replace("_", " ").title()
                    lines.extend([f"#### {matrix_title}", ""])
                    _append_matrix(lines, fields, matrix)
                    safe_name = "".join(character if character.isalnum() else "_" for character in str(matrix_name))
                    path = output_dir / f"{analysis_id}_{safe_name}.svg"
                    _correlation_svg(fields, matrix, path, title=f"{title} — {matrix_title}")
                    figures.append(path)
                    lines.extend([f"![{title} — {matrix_title}]({path.name})", ""])
            else:
                lines.extend(["```json", json.dumps(result, indent=2, sort_keys=True), "```", ""])
        else:
            lines.extend([f"Not estimable: {result.get('status', 'unknown status')}.", ""])
    unavailable = contract.get("unavailable_requests", [])
    if unavailable:
        lines.extend(["## Requested concepts not supported by these data", ""])
        for item in unavailable:
            lines.append(f"- **{item.get('concept')}** — {item.get('reason')}" if isinstance(item, dict) else f"- {item}")
        lines.append("")
    limits = [
        "Cohorts, harmonization, endpoints, groups, and other analysis parameters are defined by the approved contract. Small local cells are suppressed. Observational results are not causal or clinical advice."
    ]
    if any(item["tool"] == "federated_kaplan_meier" for item in aggregate["analyses"]):
        limits.append("Kaplan–Meier curves are time-binned and descriptive.")
    if any(item["tool"].startswith("dynamic_") for item in aggregate["analyses"]):
        limits.append("Dynamic results use the exact agent-generated source included in the human-approved contract.")
    lines.extend(["## Interpretation limits", "", " ".join(limits)])
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [report_path, *figures]


def _number(value: Any) -> str:
    return "NA" if value is None else f"{value:.3f}"


def _append_matrix(lines: list[str], fields: list[str], matrix: Any) -> None:
    if not fields or not isinstance(matrix, list) or len(matrix) != len(fields):
        lines.extend(["Not estimable after disclosure control.", ""])
        return
    lines.append("| Variable | " + " | ".join(fields) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in fields) + " |")
    for row_index, row in enumerate(fields):
        values = matrix[row_index] if row_index < len(matrix) else []
        rendered = [
            "NA" if not isinstance(values, list) or column_index >= len(values) or values[column_index] is None else f"{float(values[column_index]):.4f}"
            for column_index, _ in enumerate(fields)
        ]
        lines.append(f"| {row} | " + " | ".join(rendered) + " |")
    lines.append("")


def _dynamic_matrices(result: dict[str, Any]) -> list[tuple[str, list[str], list[list[Any]]]]:
    fields = [str(field) for field in result.get("fields", [])]
    matrices: list[tuple[str, list[str], list[list[Any]]]] = []
    for item in result.get("site_results", []):
        if isinstance(item, dict) and isinstance(item.get("correlation_matrix"), list):
            matrices.append((str(item.get("site_id", "site")), fields, item["correlation_matrix"]))
    federated = result.get("federated_result")
    if isinstance(federated, dict) and isinstance(federated.get("correlation_matrix"), list):
        matrices.append(("federated", fields, federated["correlation_matrix"]))
    legacy = result.get("matrices")
    if not matrices and isinstance(legacy, dict):
        for name, matrix in legacy.items():
            if not isinstance(matrix, dict):
                continue
            legacy_fields = sorted(str(field) for field in matrix)
            rows = [
                [matrix.get(row, {}).get(column) for column in legacy_fields]
                for row in legacy_fields
            ]
            matrices.append((str(name), legacy_fields, rows))
    return matrices


def _correlation_svg(fields: list[str], matrix: list[list[Any]], path: Path, title: str) -> None:
    count = max(len(fields), 1)
    cell = min(120, 600 / count)
    left, top = 220, 95
    size = left + cell * count + 55
    height = top + cell * count + 80
    parts = []
    for index, field in enumerate(fields):
        label = html.escape(field)
        coordinate = top + index * cell + cell / 2 + 4
        parts.append(f'<text x="{left-12}" y="{coordinate:.1f}" text-anchor="end" fill="#ddd" font-family="sans-serif" font-size="12">{label}</text>')
        parts.append(f'<text x="{left+index*cell+cell/2:.1f}" y="{top-12}" text-anchor="middle" fill="#ddd" font-family="sans-serif" font-size="12">{label}</text>')
    for row in range(count):
        for column in range(count):
            value = matrix[row][column] if row < len(matrix) and column < len(matrix[row]) else None
            numeric = float(value) if value is not None else 0.0
            intensity = min(1.0, abs(numeric))
            if value is None:
                color = "#303536"
            elif numeric >= 0:
                color = f"rgb({int(28+40*(1-intensity))},{int(80+110*intensity)},{int(70+20*(1-intensity))})"
            else:
                color = f"rgb({int(95+125*intensity)},{int(55+25*(1-intensity))},{int(65+35*(1-intensity))})"
            x, y = left + column * cell, top + row * cell
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell-2:.1f}" height="{cell-2:.1f}" rx="5" fill="{color}"/>')
            text = "NA" if value is None else f"{numeric:.3f}"
            parts.append(f'<text x="{x+cell/2:.1f}" y="{y+cell/2+5:.1f}" text-anchor="middle" fill="#fff" font-family="sans-serif" font-size="14">{text}</text>')
    path.write_text(_svg_frame(title, "".join(parts), width=int(size), height=int(height)), encoding="utf-8")


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
    bins = [float(value) for value in result["bins"]]
    minimum_time, maximum_time = min(bins), max(bins)
    time_span = maximum_time - minimum_time or 1.0
    left, right, top, bottom = 70.0, 840.0, 80.0, 460.0
    width, height = right - left, bottom - top
    parts = []

    # Horizontal survival-probability grid and labels.
    for survival in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = bottom - height * survival
        parts.append(
            f'<line x1="{left:.0f}" y1="{y:.1f}" x2="{right:.0f}" y2="{y:.1f}" '
            f'stroke="{("#777" if survival == 0 else "#303536")}" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{left-6:.0f}" y1="{y:.1f}" x2="{left:.0f}" y2="{y:.1f}" stroke="#8b9292"/>'
            f'<text x="{left-11:.0f}" y="{y+4:.1f}" text-anchor="end" fill="#b9c0c0" '
            f'font-family="sans-serif" font-size="12">{survival:.2f}</text>'
        )

    # Contract-defined time-bin ticks. Keep endpoints and thin only unusually
    # dense bin lists so labels remain legible in the fixed-width report SVG.
    stride = max(1, math.ceil((len(bins) - 1) / 8))
    tick_indices = list(range(0, len(bins), stride))
    if tick_indices[-1] != len(bins) - 1:
        tick_indices.append(len(bins) - 1)
    for index in tick_indices:
        value = bins[index]
        x = left + width * (value - minimum_time) / time_span
        parts.append(
            f'<line x1="{x:.1f}" y1="{top:.0f}" x2="{x:.1f}" y2="{bottom:.0f}" stroke="#252929"/>'
            f'<line x1="{x:.1f}" y1="{bottom:.0f}" x2="{x:.1f}" y2="{bottom+6:.0f}" stroke="#8b9292"/>'
            f'<text x="{x:.1f}" y="{bottom+23:.0f}" text-anchor="middle" fill="#b9c0c0" '
            f'font-family="sans-serif" font-size="12">{_axis_number(value)}</text>'
        )

    parts.append(
        f'<line x1="{left:.0f}" y1="{bottom:.0f}" x2="{right:.0f}" y2="{bottom:.0f}" stroke="#8b9292"/>'
        f'<line x1="{left:.0f}" y1="{top:.0f}" x2="{left:.0f}" y2="{bottom:.0f}" stroke="#8b9292"/>'
    )
    for index, (name, group) in enumerate(result["groups"].items()):
        points = []
        for point in group["curve"]:
            x = left + width * (float(point["time"]) - minimum_time) / time_span
            y = bottom - height * float(point["survival"])
            points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[index % len(COLORS)]}" stroke-width="3"/>')
        parts.append(f'<text x="620" y="{75+index*20}" fill="{COLORS[index % len(COLORS)]}" font-family="sans-serif">{html.escape(name)} (n={group["n"]})</text>')
    parts.append(f'<text x="455" y="520" text-anchor="middle" fill="#ccc" font-family="sans-serif">{html.escape(x_label)}</text><text x="16" y="270" text-anchor="middle" fill="#ccc" font-family="sans-serif" transform="rotate(-90 16 270)">Survival probability</text>')
    path.write_text(_svg_frame(title, "".join(parts)), encoding="utf-8")


def _axis_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.2f}".rstrip("0").rstrip(".")
