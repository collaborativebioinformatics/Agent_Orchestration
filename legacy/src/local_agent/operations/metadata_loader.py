#This loader reads the existing catalog.json, mappings.yaml, and policy.yaml, validates their basic structure, and builds a LocalOperationContext.
#
#It does not read data.csv; comparison against the actual DataFrame belongs in metadata_validator.py.


# local_agent/metadata_loader.py

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


import yaml

from local_agent.operations.base import LocalOperationContext


class MetadataLoadError(Exception):
    """Base exception for local metadata-loading failures."""


class MetadataFileNotFoundError(MetadataLoadError):
    """Raised when a required metadata file is missing."""


class MetadataFormatError(MetadataLoadError):
    """Raised when a metadata file has an invalid structure."""


class MetadataConsistencyError(MetadataLoadError):
    """Raised when site metadata files disagree."""


@dataclass(frozen=True, slots=True)
class SiteMetadataBundle:
    """
    Loaded and normalized metadata for one local site.

    The raw metadata are retained because later policy and validation
    components may require fields not used directly by schema discovery.
    """

    site_directory: Path
    site_id: str
    dataset_id: str
    cdm_version: int | str | None

    catalog: dict[str, Any]
    mappings: dict[str, Any]
    policy: dict[str, Any]

    column_mapping: dict[str, str]
    column_metadata: dict[str, dict[str, Any]]

    minimum_cell_size: int
    allowed_operations: tuple[str, ...]

    @property
    def data_path(self) -> Path:
        configured_path = self.catalog.get("data_file")

        if configured_path:
            return self.site_directory / str(configured_path)

        return self.site_directory / "data.csv"

    def build_context(self) -> LocalOperationContext:
        """
        Construct the context supplied to local operations.
        """
        return LocalOperationContext(
            site_id=self.site_id,
            dataset_id=self.dataset_id,
            minimum_cell_size=self.minimum_cell_size,
            column_mapping=dict(self.column_mapping),
            metadata={
                "cdm_version": self.cdm_version,
                "collection": self.catalog.get("collection"),
                "display_name": self.catalog.get(
                    "display_name"
                ),
                "patient_count_range": self.catalog.get(
                    "n_patients"
                ),
                "column_metadata": {
                    name: dict(metadata)
                    for name, metadata
                    in self.column_metadata.items()
                },
                "allowed_operations": list(
                    self.allowed_operations
                ),
                "mapping_metadata": dict(self.mappings),
                "policy": dict(self.policy),
            },
        )


class MetadataLoader:
    """
    Load site metadata from a standard simulated-site directory.

    Expected directory:

        site_directory/
        ├── catalog.json
        ├── mappings.yaml
        ├── policy.yaml
        └── data.csv
    """

    CATALOG_FILENAME = "catalog.json"
    MAPPINGS_FILENAME = "mappings.yaml"
    POLICY_FILENAME = "policy.yaml"

    def load(
        self,
        *,
        site_directory: str | Path,
        dataset_id: str | None = None,
        expected_site_id: str | None = None,
    ) -> SiteMetadataBundle:
        directory = Path(site_directory).expanduser().resolve()

        if not directory.exists():
            raise MetadataFileNotFoundError(
                f"Site directory does not exist: {directory}"
            )

        if not directory.is_dir():
            raise MetadataFormatError(
                f"Site path is not a directory: {directory}"
            )

        catalog = self._load_json(
            directory / self.CATALOG_FILENAME
        )

        mappings = self._load_yaml(
            directory / self.MAPPINGS_FILENAME
        )

        policy = self._load_yaml(
            directory / self.POLICY_FILENAME
        )

        catalog_site_id = self._required_nonempty_string(
            catalog,
            "site_id",
            source=self.CATALOG_FILENAME,
        )

        mapping_site_id = self._optional_nonempty_string(
            mappings,
            "site_id",
            source=self.MAPPINGS_FILENAME,
        )

        policy_site_id = self._optional_nonempty_string(
            policy,
            "site_id",
            source=self.POLICY_FILENAME,
        )

        site_id = self._validate_site_ids(
            catalog_site_id=catalog_site_id,
            mapping_site_id=mapping_site_id,
            policy_site_id=policy_site_id,
            expected_site_id=expected_site_id,
        )

        cdm_version = self._validate_cdm_versions(
            catalog=catalog,
            mappings=mappings,
            policy=policy,
        )

        resolved_dataset_id = (
            dataset_id
            or self._infer_dataset_id(catalog)
        )

        column_metadata = self._normalize_column_metadata(
            catalog
        )

        column_mapping = self._normalize_column_mapping(
            mappings=mappings,
            column_metadata=column_metadata,
        )

        minimum_cell_size = (
            self._extract_minimum_cell_size(policy)
        )

        allowed_operations = (
            self._extract_allowed_operations(policy)
        )

        return SiteMetadataBundle(
            site_directory=directory,
            site_id=site_id,
            dataset_id=resolved_dataset_id,
            cdm_version=cdm_version,
            catalog=catalog,
            mappings=mappings,
            policy=policy,
            column_mapping=column_mapping,
            column_metadata=column_metadata,
            minimum_cell_size=minimum_cell_size,
            allowed_operations=allowed_operations,
        )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise MetadataFileNotFoundError(
                f"Required metadata file is missing: {path}"
            )

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as stream:
                value = json.load(stream)

        except json.JSONDecodeError as exc:
            raise MetadataFormatError(
                f"Invalid JSON in {path}: "
                f"line {exc.lineno}, column {exc.colno}."
            ) from exc

        except OSError as exc:
            raise MetadataLoadError(
                f"Could not read {path}: {exc}"
            ) from exc

        if not isinstance(value, dict):
            raise MetadataFormatError(
                f"{path} must contain a JSON object."
            )

        return value

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise MetadataFileNotFoundError(
                f"Required metadata file is missing: {path}"
            )

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as stream:
                value = yaml.safe_load(stream)

        except yaml.YAMLError as exc:
            raise MetadataFormatError(
                f"Invalid YAML in {path}: {exc}"
            ) from exc

        except OSError as exc:
            raise MetadataLoadError(
                f"Could not read {path}: {exc}"
            ) from exc

        if value is None:
            return {}

        if not isinstance(value, dict):
            raise MetadataFormatError(
                f"{path} must contain a YAML mapping."
            )

        return dict(value)

    @staticmethod
    def _validate_site_ids(
        *,
        catalog_site_id: str,
        mapping_site_id: str | None,
        policy_site_id: str | None,
        expected_site_id: str | None,
    ) -> str:
        discovered = {
            "catalog.json": catalog_site_id,
        }

        if mapping_site_id is not None:
            discovered["mappings.yaml"] = mapping_site_id

        if policy_site_id is not None:
            discovered["policy.yaml"] = policy_site_id

        unique_site_ids = set(discovered.values())

        if len(unique_site_ids) != 1:
            raise MetadataConsistencyError(
                "Site IDs differ between metadata files: "
                f"{discovered}."
            )

        site_id = catalog_site_id

        if (
            expected_site_id is not None
            and site_id != expected_site_id
        ):
            raise MetadataConsistencyError(
                f"Expected site {expected_site_id!r}, but metadata "
                f"belongs to {site_id!r}."
            )

        return site_id

    @staticmethod
    def _validate_cdm_versions(
        *,
        catalog: Mapping[str, Any],
        mappings: Mapping[str, Any],
        policy: Mapping[str, Any],
    ) -> int | str | None:
        discovered: dict[str, Any] = {}

        for filename, document in (
            ("catalog.json", catalog),
            ("mappings.yaml", mappings),
            ("policy.yaml", policy),
        ):
            value = document.get("cdm_version")

            if value is not None:
                discovered[filename] = value

        if not discovered:
            return None

        normalized_versions = {
            str(value)
            for value in discovered.values()
        }

        if len(normalized_versions) != 1:
            raise MetadataConsistencyError(
                "CDM versions differ between metadata files: "
                f"{discovered}."
            )

        return next(iter(discovered.values()))

    @staticmethod
    def _infer_dataset_id(
        catalog: Mapping[str, Any],
    ) -> str:
        for key in (
            "dataset_id",
            "collection",
            "dataset",
        ):
            value = catalog.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        raise MetadataFormatError(
            "Could not determine dataset_id from catalog.json. "
            "Pass dataset_id explicitly to MetadataLoader.load()."
        )

    def _normalize_column_metadata(
        self,
        catalog: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        raw_features = catalog.get("clinical_features")

        if raw_features is None:
            raw_features = catalog.get("columns")

        if raw_features is None:
            raise MetadataFormatError(
                "catalog.json must contain either "
                "'clinical_features' or 'columns'."
            )

        if isinstance(raw_features, Mapping):
            return self._normalize_feature_mapping(
                raw_features
            )

        if isinstance(raw_features, list):
            return self._normalize_feature_list(
                raw_features
            )

        raise MetadataFormatError(
            "Catalogue clinical_features/columns must be "
            "an object or a list."
        )

    def _normalize_feature_mapping(
        self,
        features: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}

        for raw_name, raw_metadata in features.items():
            name = str(raw_name).strip()

            if not name:
                raise MetadataFormatError(
                    "Catalogue contains an empty column name."
                )

            if not isinstance(raw_metadata, Mapping):
                raise MetadataFormatError(
                    f"Metadata for column {name!r} must be an object."
                )

            normalized[name] = self._normalize_one_column(
                name=name,
                metadata=raw_metadata,
            )

        return normalized

    def _normalize_feature_list(
        self,
        features: list[Any],
    ) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}

        for position, raw_metadata in enumerate(features):
            if not isinstance(raw_metadata, Mapping):
                raise MetadataFormatError(
                    f"Column entry {position} must be an object."
                )

            raw_name = (
                raw_metadata.get("name")
                or raw_metadata.get("column")
                or raw_metadata.get("column_name")
            )

            if not isinstance(raw_name, str) or not raw_name.strip():
                raise MetadataFormatError(
                    f"Column entry {position} has no valid name."
                )

            name = raw_name.strip()

            if name in normalized:
                raise MetadataFormatError(
                    f"Duplicate catalogue column {name!r}."
                )

            normalized[name] = self._normalize_one_column(
                name=name,
                metadata=raw_metadata,
            )

        return normalized

    @staticmethod
    def _normalize_one_column(
        *,
        name: str,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(metadata)

        declared_type = (
            metadata.get("data_type")
            or metadata.get("type")
        )

        if declared_type is not None:
            if not isinstance(declared_type, str):
                raise MetadataFormatError(
                    f"Type for column {name!r} must be a string."
                )

            normalized["data_type"] = declared_type.strip()

        available_pct = metadata.get("available_pct")

        if available_pct is not None:
            if (
                not isinstance(available_pct, (int, float))
                or isinstance(available_pct, bool)
            ):
                raise MetadataFormatError(
                    f"available_pct for {name!r} must be numeric."
                )

            available_pct = float(available_pct)

            if not 0.0 <= available_pct <= 100.0:
                raise MetadataFormatError(
                    f"available_pct for {name!r} must be "
                    "between 0 and 100."
                )

            normalized["available_pct"] = available_pct

        normalized.setdefault("name", name)

        return normalized

    def _normalize_column_mapping(
        self,
        *,
        mappings: Mapping[str, Any],
        column_metadata: Mapping[str, dict[str, Any]],
    ) -> dict[str, str]:
        """
        Normalize canonical-to-local mappings.

        Supported formats:

        column_mappings:
          age:
            local_column: age_at_diagnosis

        mappings:
          age: age_at_diagnosis

        If neither exists, catalogue columns are treated as already
        standardized under the declared CDM.
        """
        raw_mappings = mappings.get("column_mappings")

        if raw_mappings is None:
            raw_mappings = mappings.get("mappings")


        

        if raw_mappings is None:
            normalized = {
                column_name: column_name
                for column_name in column_metadata
            }

            outcome = mappings.get("outcome")

            if isinstance(outcome, Mapping):
                # Derived outcomes do not map directly to physical columns.
                for outcome_name in outcome:
                    if outcome_name != "time":
                        normalized.pop(
                            str(outcome_name),
                            None,
                        )

                # In the current CDM, time_months is the canonical
                # survival-time variable.
                local_time_column = outcome.get("time")

                if (
                    isinstance(local_time_column, str)
                    and local_time_column.strip()
                    and "time_months" in column_metadata
                ):
                    normalized["time_months"] = (
                        local_time_column.strip()
                    )

            return normalized

        

        if not isinstance(raw_mappings, Mapping):
            raise MetadataFormatError(
                "mappings.yaml column_mappings/mappings must "
                "be an object."
            )

        normalized: dict[str, str] = {}

        for raw_canonical, raw_mapping in raw_mappings.items():
            canonical = str(raw_canonical).strip()

            if not canonical:
                raise MetadataFormatError(
                    "Mapping contains an empty canonical name."
                )

            if isinstance(raw_mapping, str):
                local_column = raw_mapping.strip()

            elif isinstance(raw_mapping, Mapping):
                candidate = (
                    raw_mapping.get("local_column")
                    or raw_mapping.get("column")
                    or raw_mapping.get("local_name")
                )

                if not isinstance(candidate, str):
                    raise MetadataFormatError(
                        f"Mapping for {canonical!r} requires "
                        "a local_column."
                    )

                local_column = candidate.strip()

                confirmed = raw_mapping.get("confirmed")

                if confirmed is False:
                    # Unconfirmed mappings are intentionally not added.
                    continue

            else:
                raise MetadataFormatError(
                    f"Mapping for {canonical!r} must be a string "
                    "or object."
                )

            if not local_column:
                raise MetadataFormatError(
                    f"Mapping for {canonical!r} has an empty "
                    "local column."
                )

            normalized[canonical] = local_column

        return normalized

    @staticmethod
    def _extract_minimum_cell_size(
        policy: Mapping[str, Any],
    ) -> int:
        """
        Support common policy layouts.

        privacy:
          minimum_cell_size: 10

        or:

        minimum_cell_size: 10
        """
        candidates: list[Any] = [
            policy.get("minimum_cell_size"),
            policy.get("min_cell_size"),
        ]

        privacy = policy.get("privacy")

        if isinstance(privacy, Mapping):
            candidates.extend(
                [
                    privacy.get("minimum_cell_size"),
                    privacy.get("min_cell_size"),
                ]
            )

        disclosure = policy.get("disclosure_control")

        if isinstance(disclosure, Mapping):
            candidates.extend(
                [
                    disclosure.get("minimum_cell_size"),
                    disclosure.get("min_cell_size"),
                ]
            )

        for value in candidates:
            if value is None:
                continue

            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise MetadataFormatError(
                    "minimum_cell_size must be a positive integer."
                )

            return value

        # Safe default for the initial simulation.
        return 10

    @staticmethod
    def _extract_allowed_operations(
        policy: Mapping[str, Any],
    ) -> tuple[str, ...]:
        raw_operations = policy.get("allowed_operations")

        if raw_operations is None:
            operations_section = policy.get("operations")

            if isinstance(operations_section, Mapping):
                raw_operations = operations_section.get("allowed")

        if raw_operations is None:
            return ()

        if not isinstance(raw_operations, list):
            raise MetadataFormatError(
                "allowed_operations must be a list."
            )

        normalized: list[str] = []
        seen: set[str] = set()

        for operation in raw_operations:
            if not isinstance(operation, str):
                raise MetadataFormatError(
                    "Every allowed operation must be a string."
                )

            value = operation.strip()

            if not value:
                raise MetadataFormatError(
                    "Allowed operation names cannot be empty."
                )

            if value not in seen:
                normalized.append(value)
                seen.add(value)

        return tuple(normalized)

    @staticmethod
    def _required_nonempty_string(
        document: Mapping[str, Any],
        key: str,
        *,
        source: str,
    ) -> str:
        value = document.get(key)

        if not isinstance(value, str) or not value.strip():
            raise MetadataFormatError(
                f"{source} requires a non-empty {key!r}."
            )

        return value.strip()

    @staticmethod
    def _optional_nonempty_string(
        document: Mapping[str, Any],
        key: str,
        *,
        source: str,
    ) -> str | None:
        value = document.get(key)

        if value is None:
            return None

        if not isinstance(value, str) or not value.strip():
            raise MetadataFormatError(
                f"{source} field {key!r} must be a "
                "non-empty string."
            )

        return value.strip()