"""Global access to the federated metadata catalogue.

The catalogue service operates only on privacy-safe metadata returned
by local biobank agents. It does not access patient-level data.

Responsibilities
----------------
- list participating sites;
- list available datasets;
- inspect local column metadata;
- find canonical variables shared across sites;
- resolve canonical names to local column names;
- verify column permissions;
- suggest appropriate next analytical actions.

The catalogue service does not:
- query local databases;
- run NVIDIA FLARE jobs;
- perform statistical calculations;
- infer unconfirmed mappings automatically;
- expose direct identifiers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Literal
from uuid import UUID, uuid4

from global_agent.exceptions import (
    ColumnPermissionError,
    UnknownColumnError,
    UnknownDatasetError,
    UnknownSiteError,
    UnconfirmedMappingError,
)
from shared.schemas import (
    ColumnMetadata,
    DataType,
    DatasetMetadata,
    FederatedCatalogue,
    Operation,
    SensitivityLevel,
    SiteCatalogue,
    SiteColumnMapping,
    SuggestedAction,
    VariableMapping,
)


ColumnUsage = Literal[
    "analysis",
    "grouping",
    "filtering",
    "modeling",
]


class FederatedCatalogService:
    """Inspect and resolve a federated metadata catalogue."""

    def __init__(
        self,
        catalogue: FederatedCatalogue,
    ) -> None:
        self.catalogue = catalogue

        self._site_index = self._build_site_index(
            catalogue.site_catalogues
        )

        self._mapping_index = self._build_mapping_index(
            catalogue.variable_mappings
        )

    # =================================================================
    # Construction
    # =================================================================

    @classmethod
    def from_site_catalogues(
        cls,
        *,
        site_catalogues: list[SiteCatalogue],
        unavailable_sites: list[str] | None = None,
        variable_mappings: list[VariableMapping] | None = None,
        request_id: UUID | None = None,
        warnings: list[str] | None = None,
    ) -> "FederatedCatalogService":
        """Construct a federated catalogue from local catalogues."""

        participating_sites = [
            catalogue.site_id
            for catalogue in site_catalogues
        ]

        if len(participating_sites) != len(
            set(participating_sites)
        ):
            raise ValueError(
                "Each participating site may return only "
                "one SiteCatalogue"
            )

        catalogue = FederatedCatalogue(
            request_id=request_id or uuid4(),
            participating_sites=participating_sites,
            unavailable_sites=unavailable_sites or [],
            site_catalogues=site_catalogues,
            variable_mappings=variable_mappings or [],
            warnings=warnings or [],
        )

        return cls(catalogue)

    @staticmethod
    def _build_site_index(
        site_catalogues: Iterable[SiteCatalogue],
    ) -> dict[str, SiteCatalogue]:
        """Build a site ID to SiteCatalogue index."""

        index: dict[str, SiteCatalogue] = {}

        for catalogue in site_catalogues:
            if catalogue.site_id in index:
                raise ValueError(
                    f"Duplicate SiteCatalogue for "
                    f"'{catalogue.site_id}'"
                )

            index[catalogue.site_id] = catalogue

        return index

    @staticmethod
    def _build_mapping_index(
        mappings: Iterable[VariableMapping],
    ) -> dict[str, VariableMapping]:
        """Build a canonical-name to mapping index."""

        index: dict[str, VariableMapping] = {}

        for mapping in mappings:
            if mapping.canonical_name in index:
                raise ValueError(
                    "Duplicate canonical mapping for "
                    f"'{mapping.canonical_name}'"
                )

            index[mapping.canonical_name] = mapping

        return index

    # =================================================================
    # Basic properties
    # =================================================================

    @property
    def request_id(self) -> UUID:
        """Return the catalogue request identifier."""

        return self.catalogue.request_id

    @property
    def participating_sites(self) -> list[str]:
        """Return participating site IDs."""

        return list(
            self.catalogue.participating_sites
        )

    @property
    def unavailable_sites(self) -> list[str]:
        """Return unavailable site IDs."""

        return list(
            self.catalogue.unavailable_sites
        )

    @property
    def canonical_columns(self) -> list[str]:
        """Return every canonical variable name."""

        return sorted(self._mapping_index)

    # =================================================================
    # Site and dataset lookup
    # =================================================================

    def get_site_catalogue(
        self,
        site_id: str,
    ) -> SiteCatalogue:
        """Return metadata for one participating site."""

        normalized_site = self._normalize_identifier(
            site_id
        )

        catalogue = self._site_index.get(
            normalized_site
        )

        if catalogue is None:
            raise UnknownSiteError(
                f"Site '{normalized_site}' was not found "
                "in the federated catalogue.",
                request_id=self.request_id,
                details={
                    "site_id": normalized_site,
                    "available_sites": sorted(
                        self._site_index
                    ),
                },
            )

        return catalogue

    def get_dataset(
        self,
        *,
        site_id: str,
        dataset_id: str,
    ) -> DatasetMetadata:
        """Return one local dataset description."""

        site_catalogue = self.get_site_catalogue(
            site_id
        )

        normalized_dataset = self._normalize_identifier(
            dataset_id
        )

        for dataset in site_catalogue.datasets:
            if dataset.dataset_id == normalized_dataset:
                return dataset

        raise UnknownDatasetError(
            f"Dataset '{normalized_dataset}' was not found "
            f"at site '{site_catalogue.site_id}'.",
            request_id=self.request_id,
            details={
                "site_id": site_catalogue.site_id,
                "dataset_id": normalized_dataset,
                "available_datasets": [
                    dataset.dataset_id
                    for dataset
                    in site_catalogue.datasets
                ],
            },
        )

    def get_local_column(
        self,
        *,
        site_id: str,
        dataset_id: str,
        local_name: str,
    ) -> ColumnMetadata:
        """Return metadata for one local column."""

        dataset = self.get_dataset(
            site_id=site_id,
            dataset_id=dataset_id,
        )

        normalized_column = local_name.strip()

        for column in dataset.columns:
            if column.local_name == normalized_column:
                return column

        raise UnknownColumnError(
            f"Column '{normalized_column}' was not found "
            f"in dataset '{dataset.dataset_id}' at "
            f"site '{site_id}'.",
            request_id=self.request_id,
            details={
                "site_id": site_id,
                "dataset_id": dataset.dataset_id,
                "local_name": normalized_column,
                "available_columns": [
                    column.local_name
                    for column in dataset.columns
                    if (
                        column.sensitivity
                        != SensitivityLevel.DIRECT_IDENTIFIER
                    )
                ],
            },
        )

    # =================================================================
    # Dataset and column listing
    # =================================================================

    def list_datasets(
        self,
    ) -> list[dict]:
        """Return a global summary of dataset availability."""

        datasets: dict[str, dict] = {}

        for site_id, site_catalogue in (
            self._site_index.items()
        ):
            for dataset in site_catalogue.datasets:
                summary = datasets.setdefault(
                    dataset.dataset_id,
                    {
                        "dataset_id": dataset.dataset_id,
                        "display_names": set(),
                        "sites": [],
                        "column_count_by_site": {},
                    },
                )

                if dataset.display_name:
                    summary["display_names"].add(
                        dataset.display_name
                    )

                summary["sites"].append(site_id)

                summary["column_count_by_site"][
                    site_id
                ] = len(dataset.columns)

        output: list[dict] = []

        for dataset_id in sorted(datasets):
            summary = datasets[dataset_id]

            output.append(
                {
                    "dataset_id": dataset_id,
                    "display_names": sorted(
                        summary["display_names"]
                    ),
                    "sites": sorted(
                        summary["sites"]
                    ),
                    "site_count": len(
                        summary["sites"]
                    ),
                    "column_count_by_site": (
                        summary[
                            "column_count_by_site"
                        ]
                    ),
                }
            )

        return output

    def list_columns(
        self,
        *,
        site_id: str,
        dataset_id: str,
        analysable_only: bool = True,
        include_direct_identifiers: bool = False,
    ) -> list[ColumnMetadata]:
        """Return columns from one local dataset."""

        dataset = self.get_dataset(
            site_id=site_id,
            dataset_id=dataset_id,
        )

        columns = []

        for column in dataset.columns:
            if (
                not include_direct_identifiers
                and column.sensitivity
                == SensitivityLevel.DIRECT_IDENTIFIER
            ):
                continue

            if (
                analysable_only
                and not column.allowed_for_analysis
            ):
                continue

            columns.append(column)

        return columns

    def describe_dataset_availability(
        self,
        dataset_id: str,
    ) -> dict:
        """Describe where a named dataset is available."""

        normalized_dataset = self._normalize_identifier(
            dataset_id
        )

        available_at: list[str] = []
        unavailable_at: list[str] = []
        row_counts: dict[str, int | None] = {}
        column_counts: dict[str, int] = {}

        for site_id in self.participating_sites:
            try:
                dataset = self.get_dataset(
                    site_id=site_id,
                    dataset_id=normalized_dataset,
                )

            except UnknownDatasetError:
                unavailable_at.append(site_id)
                continue

            available_at.append(site_id)
            row_counts[site_id] = dataset.row_count
            column_counts[site_id] = len(
                dataset.columns
            )

        if not available_at:
            raise UnknownDatasetError(
                f"Dataset '{normalized_dataset}' was not "
                "found at any participating site.",
                request_id=self.request_id,
                details={
                    "dataset_id": normalized_dataset,
                },
            )

        return {
            "dataset_id": normalized_dataset,
            "available_at": sorted(available_at),
            "unavailable_at": sorted(unavailable_at),
            "row_counts": row_counts,
            "column_counts": column_counts,
        }

    # =================================================================
    # Canonical-variable lookup
    # =================================================================

    def get_variable_mapping(
        self,
        canonical_name: str,
    ) -> VariableMapping:
        """Return the mapping for one canonical variable."""

        normalized_name = self._normalize_identifier(
            canonical_name
        )

        mapping = self._mapping_index.get(
            normalized_name
        )

        if mapping is None:
            raise UnknownColumnError(
                f"No canonical mapping exists for "
                f"'{normalized_name}'.",
                request_id=self.request_id,
                details={
                    "canonical_name": normalized_name,
                    "available_canonical_columns": (
                        self.canonical_columns
                    ),
                },
            )

        return mapping

    def resolve_local_column(
        self,
        *,
        canonical_name: str,
        site_id: str,
        dataset_id: str | None = None,
        require_confirmed: bool = True,
    ) -> SiteColumnMapping:
        """Resolve a canonical variable to one local column."""

        normalized_site = self._normalize_identifier(
            site_id
        )

        normalized_dataset = (
            self._normalize_identifier(dataset_id)
            if dataset_id is not None
            else None
        )

        mapping = self.get_variable_mapping(
            canonical_name
        )

        candidates = [
            site_mapping
            for site_mapping in mapping.site_columns
            if (
                site_mapping.site_id
                == normalized_site
                and (
                    normalized_dataset is None
                    or site_mapping.dataset_id
                    == normalized_dataset
                )
            )
        ]

        if not candidates:
            raise UnknownColumnError(
                f"Canonical column "
                f"'{mapping.canonical_name}' is not "
                f"available at site '{normalized_site}'.",
                request_id=self.request_id,
                details={
                    "canonical_name": (
                        mapping.canonical_name
                    ),
                    "site_id": normalized_site,
                    "dataset_id": normalized_dataset,
                },
            )

        if len(candidates) > 1:
            raise UnknownColumnError(
                f"Canonical column "
                f"'{mapping.canonical_name}' has multiple "
                f"dataset mappings at site "
                f"'{normalized_site}'. Specify dataset_id.",
                request_id=self.request_id,
                details={
                    "canonical_name": (
                        mapping.canonical_name
                    ),
                    "site_id": normalized_site,
                    "candidate_datasets": sorted(
                        candidate.dataset_id
                        for candidate in candidates
                    ),
                },
            )

        resolved = candidates[0]

        if require_confirmed and not resolved.confirmed:
            raise UnconfirmedMappingError(
                f"Mapping for "
                f"'{mapping.canonical_name}' at "
                f"'{normalized_site}' has not been confirmed.",
                request_id=self.request_id,
                details={
                    "canonical_name": (
                        mapping.canonical_name
                    ),
                    "site_id": normalized_site,
                    "dataset_id": resolved.dataset_id,
                    "local_name": resolved.local_name,
                    "confidence": resolved.confidence,
                    "mapping_method": (
                        resolved.mapping_method.value
                    ),
                },
            )

        return resolved

    def available_sites_for_column(
        self,
        canonical_name: str,
        *,
        dataset_id: str | None = None,
        confirmed_only: bool = True,
    ) -> list[str]:
        """Return sites containing a canonical column."""

        mapping = self.get_variable_mapping(
            canonical_name
        )

        normalized_dataset = (
            self._normalize_identifier(dataset_id)
            if dataset_id is not None
            else None
        )

        sites = {
            site_mapping.site_id
            for site_mapping in mapping.site_columns
            if (
                (
                    normalized_dataset is None
                    or site_mapping.dataset_id
                    == normalized_dataset
                )
                and (
                    not confirmed_only
                    or site_mapping.confirmed
                )
            )
        }

        return sorted(sites)

    def shared_columns(
        self,
        *,
        sites: list[str] | None = None,
        dataset_id: str | None = None,
        confirmed_only: bool = True,
    ) -> list[str]:
        """Return canonical columns available at every target site."""

        target_sites = self._resolve_target_sites(
            sites
        )

        shared: list[str] = []

        for canonical_name in self.canonical_columns:
            available_sites = set(
                self.available_sites_for_column(
                    canonical_name,
                    dataset_id=dataset_id,
                    confirmed_only=confirmed_only,
                )
            )

            if target_sites.issubset(
                available_sites
            ):
                shared.append(canonical_name)

        return shared

    def columns_by_site(
        self,
        *,
        dataset_id: str,
        analysable_only: bool = True,
    ) -> dict[str, list[str]]:
        """Return safe local column names grouped by site."""

        result: dict[str, list[str]] = {}

        for site_id in self.participating_sites:
            try:
                columns = self.list_columns(
                    site_id=site_id,
                    dataset_id=dataset_id,
                    analysable_only=analysable_only,
                )

            except UnknownDatasetError:
                continue

            result[site_id] = sorted(
                column.local_name
                for column in columns
            )

        return result

    def site_specific_columns(
        self,
        *,
        dataset_id: str,
    ) -> dict[str, list[str]]:
        """Return columns without a confirmed cross-site mapping.

        A variable is considered cross-site when its confirmed
        canonical mapping covers at least two sites.
        """

        cross_site_local_columns: dict[
            str,
            set[str],
        ] = defaultdict(set)

        for mapping in self.catalogue.variable_mappings:
            confirmed = [
                site_mapping
                for site_mapping in mapping.site_columns
                if (
                    site_mapping.confirmed
                    and site_mapping.dataset_id
                    == dataset_id
                )
            ]

            confirmed_sites = {
                item.site_id
                for item in confirmed
            }

            if len(confirmed_sites) >= 2:
                for item in confirmed:
                    cross_site_local_columns[
                        item.site_id
                    ].add(item.local_name)

        output: dict[str, list[str]] = {}

        for site_id, local_columns in (
            self.columns_by_site(
                dataset_id=dataset_id,
                analysable_only=True,
            ).items()
        ):
            output[site_id] = sorted(
                set(local_columns)
                - cross_site_local_columns[site_id]
            )

        return output

    # =================================================================
    # Permission validation
    # =================================================================

    def validate_column_access(
        self,
        *,
        canonical_name: str,
        usage: ColumnUsage,
        sites: list[str] | None = None,
        dataset_id: str | None = None,
        require_confirmed: bool = True,
    ) -> dict[str, SiteColumnMapping]:
        """Validate a canonical column at all requested sites.

        Returns
        -------
        dict
            Mapping from site ID to resolved local column mapping.

        Raises
        ------
        UnknownColumnError
            If the column is unavailable at a requested site.

        UnconfirmedMappingError
            If a mapping requires human confirmation.

        ColumnPermissionError
            If a site prohibits the requested use.
        """

        target_sites = sorted(
            self._resolve_target_sites(sites)
        )

        resolved: dict[
            str,
            SiteColumnMapping,
        ] = {}

        for site_id in target_sites:
            site_mapping = self.resolve_local_column(
                canonical_name=canonical_name,
                site_id=site_id,
                dataset_id=dataset_id,
                require_confirmed=require_confirmed,
            )

            column = self.get_local_column(
                site_id=site_mapping.site_id,
                dataset_id=site_mapping.dataset_id,
                local_name=site_mapping.local_name,
            )

            if not self._is_usage_permitted(
                column,
                usage,
            ):
                raise ColumnPermissionError(
                    f"Column '{canonical_name}' is not "
                    f"permitted for {usage} at site "
                    f"'{site_id}'.",
                    request_id=self.request_id,
                    details={
                        "canonical_name": canonical_name,
                        "site_id": site_id,
                        "dataset_id": (
                            site_mapping.dataset_id
                        ),
                        "local_name": (
                            site_mapping.local_name
                        ),
                        "usage": usage,
                    },
                )

            resolved[site_id] = site_mapping

        return resolved

    @staticmethod
    def _is_usage_permitted(
        column: ColumnMetadata,
        usage: ColumnUsage,
    ) -> bool:
        """Return whether a column permits one type of use."""

        permissions = {
            "analysis": column.allowed_for_analysis,
            "grouping": column.allowed_for_grouping,
            "filtering": column.allowed_for_filtering,
            "modeling": column.allowed_for_modeling,
        }

        return permissions[usage]

    # =================================================================
    # User suggestions and summaries
    # =================================================================

    def suggest_actions(
        self,
        *,
        dataset_id: str | None = None,
    ) -> list[SuggestedAction]:
        """Suggest safe next actions based on the catalogue."""

        suggestions = [
            SuggestedAction(
                action=Operation.DISCOVER_DATASETS,
                description=(
                    "View datasets available across "
                    "participating biobanks."
                ),
            )
        ]

        if dataset_id is None:
            return suggestions

        normalized_dataset = self._normalize_identifier(
            dataset_id
        )

        shared = self.shared_columns(
            dataset_id=normalized_dataset,
        )

        suggestions.append(
            SuggestedAction(
                action=Operation.DISCOVER_SCHEMA,
                description=(
                    "View columns, data types and descriptions "
                    "for the selected dataset."
                ),
                parameters={
                    "dataset_id": normalized_dataset,
                },
            )
        )

        suggestions.append(
            SuggestedAction(
                action=Operation.PROFILE_COLUMNS,
                description=(
                    "Inspect missingness and basic metadata "
                    "for selected columns."
                ),
                parameters={
                    "dataset_id": normalized_dataset,
                    "available_columns": shared,
                },
            )
        )

        numeric_columns = (
            self._shared_columns_of_type(
                dataset_id=normalized_dataset,
                allowed_types={
                    DataType.INTEGER,
                    DataType.FLOAT,
                    DataType.NUMERIC,
                },
            )
        )

        if numeric_columns:
            suggestions.append(
                SuggestedAction(
                    action=Operation.NUMERIC_SUMMARY,
                    description=(
                        "Calculate federated summaries for "
                        "shared numeric columns."
                    ),
                    parameters={
                        "dataset_id":
                            normalized_dataset,
                        "available_columns":
                            numeric_columns,
                    },
                )
            )

            suggestions.append(
                SuggestedAction(
                    action=Operation.HISTOGRAM,
                    description=(
                        "Generate a privacy-preserving "
                        "histogram of a numeric column."
                    ),
                    parameters={
                        "dataset_id":
                            normalized_dataset,
                        "available_columns":
                            numeric_columns,
                    },
                )
            )

        categorical_columns = (
            self._shared_columns_of_type(
                dataset_id=normalized_dataset,
                allowed_types={
                    DataType.BOOLEAN,
                    DataType.BINARY,
                    DataType.CATEGORICAL,
                },
            )
        )

        if categorical_columns:
            suggestions.append(
                SuggestedAction(
                    action=(
                        Operation
                        .CATEGORICAL_DISTRIBUTION
                    ),
                    description=(
                        "Compare privacy-safe category "
                        "counts across participating sites."
                    ),
                    parameters={
                        "dataset_id":
                            normalized_dataset,
                        "available_columns":
                            categorical_columns,
                    },
                )
            )

        return suggestions

    def summary(
        self,
    ) -> dict:
        """Return a JSON-compatible catalogue summary."""

        datasets = self.list_datasets()

        return {
            "request_id": str(self.request_id),
            "participating_sites": (
                self.participating_sites
            ),
            "unavailable_sites": (
                self.unavailable_sites
            ),
            "site_count": len(
                self.participating_sites
            ),
            "datasets": datasets,
            "canonical_columns": (
                self.canonical_columns
            ),
            "warnings": list(
                self.catalogue.warnings
            ),
        }

    # =================================================================
    # Internal helpers
    # =================================================================

    def _shared_columns_of_type(
        self,
        *,
        dataset_id: str,
        allowed_types: set[DataType],
    ) -> list[str]:
        """Return shared columns matching specified types."""

        matching: list[str] = []

        for canonical_name in self.shared_columns(
            dataset_id=dataset_id,
        ):
            all_types_match = True

            for site_id in self.participating_sites:
                mapping = self.resolve_local_column(
                    canonical_name=canonical_name,
                    site_id=site_id,
                    dataset_id=dataset_id,
                )

                column = self.get_local_column(
                    site_id=site_id,
                    dataset_id=dataset_id,
                    local_name=mapping.local_name,
                )

                if column.data_type not in allowed_types:
                    all_types_match = False
                    break

            if all_types_match:
                matching.append(canonical_name)

        return matching

    def _resolve_target_sites(
        self,
        sites: list[str] | None,
    ) -> set[str]:
        """Resolve an optional list of target sites."""

        if not sites:
            return set(self.participating_sites)

        normalized_sites = {
            self._normalize_identifier(site)
            for site in sites
        }

        unknown_sites = (
            normalized_sites
            - set(self.participating_sites)
        )

        if unknown_sites:
            raise UnknownSiteError(
                "The following requested sites are not "
                "participating: "
                + ", ".join(
                    sorted(unknown_sites)
                ),
                request_id=self.request_id,
                details={
                    "unknown_sites": sorted(
                        unknown_sites
                    ),
                    "participating_sites": (
                        self.participating_sites
                    ),
                },
            )

        return normalized_sites

    @staticmethod
    def _normalize_identifier(
        value: str,
    ) -> str:
        """Normalize a site, dataset or canonical identifier."""

        normalized = value.strip().lower()

        if not normalized:
            raise ValueError(
                "Identifier cannot be empty"
            )

        return normalized