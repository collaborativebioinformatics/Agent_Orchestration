"""Apply contract-provided, site-specific field harmonization."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


_OTHER_NON_MISSING = "__other_non_missing__"


def harmonize(data: pd.DataFrame, site_id: str, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = data.copy()
    site_config = config.get(site_id, {}) if isinstance(config, dict) else {}
    fields = site_config.get("fields", {}) if isinstance(site_config, dict) else {}
    applied = []
    for canonical, specification in fields.items():
        if not isinstance(specification, dict):
            raise ValueError(f"Invalid harmonization specification for {canonical}")
        source = specification.get("source", canonical)
        if source not in output:
            raise ValueError(f"Harmonization source {source!r} is missing at {site_id}")
        series = output[source].copy()
        value_map = specification.get("value_map")
        if isinstance(value_map, dict):
            normalized_map = {str(key).strip().casefold(): value for key, value in value_map.items()}
            has_default_non_missing = _OTHER_NON_MISSING in normalized_map
            default_non_missing = normalized_map.pop(_OTHER_NON_MISSING, None)

            def map_value(value: Any) -> Any:
                if pd.isna(value):
                    return value
                normalized = str(value).strip().casefold()
                if normalized in normalized_map:
                    return normalized_map[normalized]
                if has_default_non_missing:
                    return default_non_missing
                return value

            series = series.map(map_value)
        if "multiply" in specification:
            factor = float(specification["multiply"])
            # Agent-authored adapters may include an identity conversion on every
            # field. Identity multiplication must not destroy categorical labels.
            # Non-identity unit conversions remain numeric-only by definition.
            if factor != 1.0 or is_numeric_dtype(series.dtype):
                series = pd.to_numeric(series, errors="coerce") * factor
        output[canonical] = series
        applied.append({"canonical": canonical, "source": source, "operations": sorted(set(specification) - {"source"})})
    return output, {"site_id": site_id, "applied": applied}
