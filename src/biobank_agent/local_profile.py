"""Disclosure-controlled local evidence for client-authored data adapters."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

_PROFILE_LIMIT = 80
_CATEGORY_LIMIT = 20
_ENDPOINT_TERMS = ("death", "event", "status", "survival", "outcome", "vital")


def build_local_profile(
    data: pd.DataFrame,
    catalog: dict[str, Any],
    question: str,
    *,
    min_cell_count: int = 10,
) -> dict[str, Any]:
    """Summarize locally observed structure without exposing rows or rare counts."""
    declared = set((catalog.get("clinical_features") or {}).keys())
    tokens = {token for token in re.findall(r"[a-z0-9]+", question.casefold()) if len(token) >= 3}
    selected = []
    for column in data.columns:
        normalized = str(column).casefold()
        if column in declared or any(token in normalized for token in tokens):
            selected.append(str(column))
    endpoint_columns = [
        str(column) for column in data.columns if any(term in str(column).casefold() for term in _ENDPOINT_TERMS)
    ]
    selected = list(dict.fromkeys(selected + endpoint_columns))[:_PROFILE_LIMIT]

    fields: dict[str, Any] = {}
    categorical: list[str] = []
    for name in selected:
        series = data[name]
        nonmissing = series.dropna()
        numeric = pd.to_numeric(nonmissing, errors="coerce")
        numeric_fraction = float(numeric.notna().mean()) if len(nonmissing) else 0.0
        unique_count = int(nonmissing.nunique(dropna=True))
        profile: dict[str, Any] = {
            "inferred_type": "numeric" if is_numeric_dtype(series.dtype) or numeric_fraction >= 0.98 else "categorical",
            "nonmissing_count": _safe_count(int(nonmissing.size), min_cell_count),
            "missing_count": _safe_count(int(series.isna().sum()), min_cell_count),
            "distinct_values": _safe_count(unique_count, min_cell_count),
        }
        if 0 < unique_count <= _CATEGORY_LIMIT:
            counts = nonmissing.astype(str).str.strip().value_counts(dropna=False)
            safe_categories = [
                {"value": str(value), "count": _safe_count(int(count), min_cell_count)}
                for value, count in counts.items()
                if int(count) >= min_cell_count
            ]
            suppressed_categories = int((counts < min_cell_count).sum())
            if suppressed_categories:
                safe_categories.append(
                    {"value": "__SUPPRESSED_CATEGORIES__", "count": "suppressed", "category_count": suppressed_categories}
                )
            profile["observed_categories"] = safe_categories
            categorical.append(name)
        fields[name] = profile

    associations = []
    endpoint_categorical = [
        name for name in categorical if any(term in name.casefold() for term in _ENDPOINT_TERMS)
    ]
    for left_index, left in enumerate(endpoint_categorical):
        for right in endpoint_categorical[left_index + 1 :]:
            table = pd.crosstab(data[left].astype("string"), data[right].astype("string"), dropna=True)
            cells = []
            for left_value in table.index:
                for right_value in table.columns:
                    count = int(table.loc[left_value, right_value])
                    if count >= min_cell_count:
                        cells.append({"left": str(left_value), "right": str(right_value), "count": count})
            if cells:
                associations.append({"left_field": left, "right_field": right, "cells": cells})

    payload = {
        "schema_version": "biobank.local_profile.v1",
        "privacy": {
            "min_cell_count": min_cell_count,
            "row_values_included": False,
            "rare_counts_represented_as": "suppressed",
        },
        "fields": fields,
        "categorical_associations": associations,
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def verify_adapter_locally(adapter: dict[str, Any], data: pd.DataFrame, profile_digest: str) -> None:
    """Reject an adapter that contradicts locally observed data or unit semantics."""
    fields = adapter.get("fields", {})
    if not isinstance(fields, dict):
        raise ValueError("data_adapter.fields must be an object")
    for canonical, specification in fields.items():
        if not isinstance(specification, dict):
            raise ValueError(f"Adapter field {canonical!r} must be an object")
        source = specification.get("source")
        if source not in data.columns:
            raise ValueError(f"Adapter source {source!r} is not a local CSV field")
        factor = float(specification.get("multiply", 1.0))
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError(f"Adapter multiplier for {canonical!r} must be finite and positive")
        value_map = specification.get("value_map", {})
        if not isinstance(value_map, dict):
            raise ValueError(f"Adapter value_map for {canonical!r} must be an object")
        observed = {str(value).strip().casefold() for value in data[source].dropna().unique()}
        explicit = {
            str(key).strip().casefold()
            for key in value_map
            if str(key).strip().upper() not in {"__OTHER_NON_MISSING__", "__OTHER_NONMISSING__", "__MISSING__"}
        }
        unknown = sorted(explicit - observed)
        if unknown:
            raise ValueError(f"Adapter for {canonical!r} maps values not observed locally: {unknown[:5]}")
        has_default = any(
            str(key).strip().upper() in {"__OTHER_NON_MISSING__", "__OTHER_NONMISSING__"} for key in value_map
        )
        if value_map and not has_default and observed - explicit:
            raise ValueError(
                f"Adapter for {canonical!r} leaves observed nonmissing categories unmapped; "
                "map them explicitly or use __OTHER_NON_MISSING__"
            )
        if "event" in str(canonical).casefold():
            mapped = {value for value in value_map.values() if value is not None}
            if value_map and not mapped.issubset({0, 1, 0.0, 1.0}):
                raise ValueError(f"Event adapter {canonical!r} must map to binary 0/1")
    adapter["verification"] = {
        "status": "verified_against_local_profile",
        "profile_digest": profile_digest,
    }
def _safe_count(value: int, min_cell_count: int) -> int | str:
    return value if value == 0 or value >= min_cell_count else "suppressed"
