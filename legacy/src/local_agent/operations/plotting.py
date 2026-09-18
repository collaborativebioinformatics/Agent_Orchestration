# local_agent/plotting.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

# Suitable for servers without a graphical display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_correlation_heatmap(
    *,
    correlation_result: dict[str, Any],
    output_path: str | Path,
    title: str | None = None,
    annot: bool | None = None,
    dpi: int = 250,
) -> Path:
    """
    Save a publication-quality correlation heatmap.

    Expected correlation_result:

    {
        "columns": [...],
        "count": 100,
        "correlation_matrix": [[...], [...]]
    }
    """
    columns = correlation_result.get("columns")
    matrix = correlation_result.get("correlation_matrix")
    count = correlation_result.get("count")

    if not isinstance(columns, list) or len(columns) < 2:
        raise ValueError(
            "Correlation result requires at least two columns."
        )

    if not isinstance(matrix, list):
        raise ValueError(
            "correlation_matrix must be a list of rows."
        )

    dimension = len(columns)

    if len(matrix) != dimension:
        raise ValueError(
            "Correlation matrix row count does not match "
            "the number of columns."
        )

    for row in matrix:
        if not isinstance(row, list) or len(row) != dimension:
            raise ValueError(
                "Correlation matrix must be square."
            )

    numeric_matrix = np.array(
        [
            [
                np.nan if value is None else float(value)
                for value in row
            ]
            for row in matrix
        ],
        dtype=float,
    )

    if np.isinf(numeric_matrix).any():
        raise ValueError(
            "Correlation matrix cannot contain infinity."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = [
        _format_column_label(column)
        for column in columns
    ]

    correlation_dataframe = pd.DataFrame(
        numeric_matrix,
        index=labels,
        columns=labels,
    )

    if annot is None:
        annot = dimension <= 15

    figure_width = max(
        7.5,
        min(20.0, 2.5 + dimension * 0.72),
    )

    figure_height = max(
        6.5,
        min(18.0, 2.0 + dimension * 0.68),
    )

    sns.set_theme(
        context="notebook",
        style="white",
        font_scale=0.9,
    )

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
    )

    # Display only one triangle because the correlation matrix
    # is symmetric.
    upper_triangle_mask = np.triu(
        np.ones_like(
            numeric_matrix,
            dtype=bool,
        ),
        k=1,
    )

    heatmap = sns.heatmap(
        correlation_dataframe,
        mask=upper_triangle_mask,
        cmap=sns.diverging_palette(
            240,
            10,
            as_cmap=True,
        ),
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        annot=annot,
        fmt=".2f",
        annot_kws={
            "fontsize": 8,
            "fontweight": "normal",
        },
        linewidths=0.7,
        linecolor="white",
        cbar_kws={
            "label": "Pearson correlation",
            "shrink": 0.78,
            "pad": 0.03,
        },
        ax=axis,
    )

    if title is None:
        title = "Local Pearson correlation heatmap"

    subtitle = (
        f"Listwise complete cases: n={count:,}"
        if isinstance(count, int)
        else "Listwise complete-case analysis"
    )

    figure.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    axis.set_title(
        subtitle,
        fontsize=10,
        color="#555555",
        pad=14,
    )

    axis.set_xlabel("")
    axis.set_ylabel("")

    axis.set_xticklabels(
        axis.get_xticklabels(),
        rotation=45,
        horizontalalignment="right",
        rotation_mode="anchor",
    )

    axis.set_yticklabels(
        axis.get_yticklabels(),
        rotation=0,
    )

    axis.tick_params(
        axis="both",
        labelsize=9,
        length=0,
    )

    colorbar = heatmap.collections[0].colorbar

    if colorbar is not None:
        colorbar.ax.tick_params(
            labelsize=8,
        )

        colorbar.set_ticks(
            [-1.0, -0.5, 0.0, 0.5, 1.0]
        )

    # Add a small explanatory footer.
    figure.text(
        0.01,
        0.01,
        (
            "Values show Pearson correlation. "
            "Blank cells indicate undefined correlation, "
            "usually caused by a constant variable."
        ),
        ha="left",
        va="bottom",
        fontsize=8,
        color="#666666",
    )

    figure.tight_layout(
        rect=(0.02, 0.04, 0.98, 0.94)
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    return output_path


def _format_column_label(
    column_name: Any,
) -> str:
    """
    Convert machine-readable names into plot labels.

    Example:
        age_at_diagnosis -> Age at diagnosis
    """
    text = str(column_name).strip()

    if not text:
        return "Unnamed variable"

    text = text.replace("_", " ")
    text = " ".join(text.split())

    return text[0].upper() + text[1:]


    