
"""Generic cross-biobank schema harmonization.

This module proposes mappings between canonical variables and local
biobank columns using privacy-safe metadata.

Mapping priority
----------------
1. Semantic type supplied by a local data dictionary or steward.
2. Optional project-specific aliases supplied at runtime.
3. Exact normalized local-column names.

This module contains no hardcoded biomedical variable names.

The harmonizer does not:

- access patient-level data;
- inspect raw column values;
- execute an LLM;
- silently resolve ambiguous mappings;
- automatically confirm project-specific aliases;
- modify local datasets.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from shared.schemas import (
    ColumnMetadata,
    DataType,
    FederatedCatalogue,
    MappingMethod,
    SensitivityLevel,
    SiteCatalogue,
    SiteColumnMapping,
    VariableMapping,
)


# =====================================================================
# Harmonization outputs
# =====================================================================


@dataclass(frozen=True)
class HarmonizationIssue:
    """One issue discovered during schema harmonization."""

    issue_type: str

    message: str

    canonical_name: str | None = None

    site_id: str | None = None

    dataset_id: str | None = None

    local_columns: tuple[str, ...] = ()

    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-compatible representation."""

        return {
            "issue_type": self.issue_type,
            "message": self.message,
            "canonical_name": self.canonical_name,
            "site_id": self.site_id,
            "dataset_id": self.dataset_id,
            "local_columns": list(
                self.local_columns
            ),
            "details": dict(self.details),
        }


@dataclass
class HarmonizationResult:
    """Mappings and issues produced by harmonization."""

    variable_mappings: list[VariableMapping]

    issues: list[HarmonizationIssue]

    def confirmed_mappings(
        self,
    ) -> list[VariableMapping]:
        """Return only confirmed site-column mappings."""

        output: list[VariableMapping] = []

        for mapping in self.variable_mappings:
            confirmed_columns = [
                site_column
                for site_column in mapping.site_columns
                if site_column.confirmed
            ]

            if not confirmed_columns:
                continue

            output.append(
                mapping.model_copy(
                    update={
                        "site_columns":
                            confirmed_columns,
                    }
                )
            )

        return output

    def unconfirmed_mappings(
        self,
    ) -> list[VariableMapping]:
        """Return only unconfirmed mapping suggestions."""

        output: list[VariableMapping] = []

        for mapping in self.variable_mappings:
            unconfirmed_columns = [
                site_column
                for site_column in mapping.site_columns
                if not site_column.confirmed
            ]

            if not unconfirmed_columns:
                continue

            output.append(
                mapping.model_copy(
                    update={
                        "site_columns":
                            unconfirmed_columns,
                    }
                )
            )

        return output

    def mappings_requiring_review(
        self,
    ) -> list[VariableMapping]:
        """Return mappings containing an unconfirmed site column."""

        return [
            mapping
            for mapping in self.variable_mappings
            if any(
                not site_column.confirmed
                for site_column in mapping.site_columns
            )
        ]

    def to_dict(self) -> dict:
        """Return a JSON-compatible representation."""

        return {
            "variable_mappings": [
                mapping.model_dump(mode="json")
                for mapping in self.variable_mappings
            ],
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
        }


# =====================================================================
# Internal mapping candidate
# =====================================================================


@dataclass(frozen=True)
class _MappingCandidate:
    """Internal candidate for one local column."""

    canonical_name: str

    site_id: str

    dataset_id: str

    local_name: str

    mapping_method: MappingMethod

    confidence: float

    confirmed: bool

    column: ColumnMetadata


# =====================================================================
# Schema harmonizer
# =====================================================================


class SchemaHarmonizer:
    """Harmonize columns from multiple privacy-safe site catalogues.

    Parameters
    ----------
    aliases:
        Optional externally supplied mapping from canonical names to
        known local aliases.

        Example:

            {
                "serum_c_reactive_protein": {
                    "crp_serum",
                    "biochem_17",
                    "serum_crp",
                }
            }

        Aliases are treated as suggestions and are not automatically
        confirmed.

    include_non_analysable:
        If false, exclude columns for which
        ``allowed_for_analysis=False``.

    alias_confidence:
        Confidence assigned to externally configured alias mappings.
        Such mappings still require confirmation.
    """

    def __init__(
        self,
        *,
        aliases: dict[
            str,
            Iterable[str],
        ] | None = None,
        include_non_analysable: bool = False,
        alias_confidence: float = 0.95,
    ) -> None:
        if not 0 <= alias_confidence <= 1:
            raise ValueError(
                "alias_confidence must be between 0 and 1"
            )

        self.include_non_analysable = (
            include_non_analysable
        )

        self.alias_confidence = alias_confidence

        self.aliases = self._normalize_aliases(
            aliases or {}
        )

        self._alias_to_canonical = (
            self._build_reverse_alias_index(
                self.aliases
            )
        )

    # =================================================================
    # Public methods
    # =================================================================

    def harmonize(
        self,
        site_catalogues: list[SiteCatalogue],
        *,
        minimum_sites: int = 1,
    ) -> HarmonizationResult:
        """Create canonical mappings from local catalogues.

        Parameters
        ----------
        site_catalogues:
            Privacy-safe schema metadata from local biobanks.

        minimum_sites:
            Minimum number of distinct sites required for a canonical
            variable to be included.

            Use ``1`` to retain site-specific variables.

            Use ``2`` to retain only variables appearing at two or
            more sites.

        Returns
        -------
        HarmonizationResult
            Proposed mappings and issues requiring review.
        """

        if minimum_sites < 1:
            raise ValueError(
                "minimum_sites must be at least 1"
            )

        self._validate_unique_sites(
            site_catalogues
        )

        candidates = self._collect_candidates(
            site_catalogues
        )

        mappings, resolution_issues = (
            self._resolve_candidates(
                candidates
            )
        )

        consistency_issues = (
            self._check_mapping_consistency(
                mappings=mappings,
                site_catalogues=site_catalogues,
            )
        )

        filtered_mappings = [
            mapping
            for mapping in mappings
            if self._mapping_site_count(mapping)
            >= minimum_sites
        ]

        return HarmonizationResult(
            variable_mappings=sorted(
                filtered_mappings,
                key=lambda item:
                    item.canonical_name,
            ),
            issues=[
                *resolution_issues,
                *consistency_issues,
            ],
        )

    def apply_to_catalogue(
        self,
        catalogue: FederatedCatalogue,
        *,
        minimum_sites: int = 1,
    ) -> tuple[
        FederatedCatalogue,
        HarmonizationResult,
    ]:
        """Run harmonization and attach mappings to a catalogue."""

        result = self.harmonize(
            catalogue.site_catalogues,
            minimum_sites=minimum_sites,
        )

        harmonization_warnings = [
            issue.message
            for issue in result.issues
        ]

        updated_data = catalogue.model_dump(
            mode="json"
        )

        updated_data["variable_mappings"] = [
            mapping.model_dump(mode="json")
            for mapping in result.variable_mappings
        ]

        updated_data["warnings"] = [
            *catalogue.warnings,
            *harmonization_warnings,
        ]

        updated_catalogue = (
            FederatedCatalogue.model_validate(
                updated_data
            )
        )

        return updated_catalogue, result

    def confirm_mapping(
        self,
        mapping: VariableMapping,
        *,
        site_id: str,
        dataset_id: str,
        local_name: str,
    ) -> VariableMapping:
        """Confirm one existing mapping following human review."""

        normalized_site = self._normalize_name(
            site_id
        )

        normalized_dataset = self._normalize_name(
            dataset_id
        )

        found = False

        updated_columns: list[
            SiteColumnMapping
        ] = []

        for site_column in mapping.site_columns:
            if (
                site_column.site_id
                == normalized_site
                and site_column.dataset_id
                == normalized_dataset
                and site_column.local_name
                == local_name
            ):
                found = True

                updated_columns.append(
                    site_column.model_copy(
                        update={
                            "mapping_method": (
                                MappingMethod
                                .HUMAN_CONFIRMED
                            ),
                            "confidence": 1.0,
                            "confirmed": True,
                        }
                    )
                )

            else:
                updated_columns.append(
                    site_column
                )

        if not found:
            raise ValueError(
                "The requested site-column mapping "
                "does not exist"
            )

        updated_mapping = mapping.model_copy(
            update={
                "site_columns": updated_columns,
            }
        )

        return VariableMapping.model_validate(
            updated_mapping.model_dump(
                mode="json"
            )
        )

    def reject_mapping(
        self,
        mapping: VariableMapping,
        *,
        site_id: str,
        dataset_id: str,
        local_name: str,
    ) -> VariableMapping | None:
        """Remove a rejected site-column mapping.

        Returns ``None`` if rejecting the site column leaves the
        canonical mapping with no columns.
        """

        normalized_site = self._normalize_name(
            site_id
        )

        normalized_dataset = self._normalize_name(
            dataset_id
        )

        found = False

        retained_columns: list[
            SiteColumnMapping
        ] = []

        for site_column in mapping.site_columns:
            should_remove = (
                site_column.site_id
                == normalized_site
                and site_column.dataset_id
                == normalized_dataset
                and site_column.local_name
                == local_name
            )

            if should_remove:
                found = True
            else:
                retained_columns.append(
                    site_column
                )

        if not found:
            raise ValueError(
                "The requested site-column mapping "
                "does not exist"
            )

        if not retained_columns:
            return None

        updated_mapping = mapping.model_copy(
            update={
                "site_columns": retained_columns,
            }
        )

        return VariableMapping.model_validate(
            updated_mapping.model_dump(
                mode="json"
            )
        )

    # =================================================================
    # Candidate collection
    # =================================================================

    def _collect_candidates(
        self,
        site_catalogues: list[SiteCatalogue],
    ) -> list[_MappingCandidate]:
        """Create one mapping candidate per eligible column."""

        candidates: list[_MappingCandidate] = []

        for site_catalogue in site_catalogues:
            for dataset in site_catalogue.datasets:
                for column in dataset.columns:
                    if not self._should_consider_column(
                        column
                    ):
                        continue

                    candidates.append(
                        self._candidate_from_column(
                            site_id=(
                                site_catalogue.site_id
                            ),
                            dataset_id=(
                                dataset.dataset_id
                            ),
                            column=column,
                        )
                    )

        return candidates

    def _should_consider_column(
        self,
        column: ColumnMetadata,
    ) -> bool:
        """Determine whether metadata may be harmonized."""

        if (
            column.sensitivity
            == SensitivityLevel.DIRECT_IDENTIFIER
        ):
            return False

        if (
            not self.include_non_analysable
            and not column.allowed_for_analysis
        ):
            return False

        return True

    def _candidate_from_column(
        self,
        *,
        site_id: str,
        dataset_id: str,
        column: ColumnMetadata,
    ) -> _MappingCandidate:
        """Create one candidate using generic mapping rules."""

        normalized_local_name = (
            self._normalize_name(
                column.local_name
            )
        )

        # -------------------------------------------------------------
        # 1. Steward-provided semantic metadata
        # -------------------------------------------------------------

        if column.semantic_type:
            canonical_name = (
                self._normalize_name(
                    column.semantic_type
                )
            )

            return _MappingCandidate(
                canonical_name=canonical_name,
                site_id=site_id,
                dataset_id=dataset_id,
                local_name=column.local_name,
                mapping_method=(
                    MappingMethod.DATA_DICTIONARY
                ),
                confidence=1.0,
                confirmed=True,
                column=column,
            )

        # -------------------------------------------------------------
        # 2. Optional externally configured alias
        # -------------------------------------------------------------

        configured_canonical = (
            self._alias_to_canonical.get(
                normalized_local_name
            )
        )

        if configured_canonical is not None:
            return _MappingCandidate(
                canonical_name=(
                    configured_canonical
                ),
                site_id=site_id,
                dataset_id=dataset_id,
                local_name=column.local_name,
                mapping_method=(
                    MappingMethod.RULE_BASED
                ),
                confidence=self.alias_confidence,
                confirmed=False,
                column=column,
            )

        # -------------------------------------------------------------
        # 3. Exact normalized name
        # -------------------------------------------------------------

        # Identical normalized names across sites will naturally be
        # grouped together. A site-specific name is retained as its
        # own canonical entry until an explicit mapping is proposed.
        return _MappingCandidate(
            canonical_name=normalized_local_name,
            site_id=site_id,
            dataset_id=dataset_id,
            local_name=column.local_name,
            mapping_method=MappingMethod.EXACT_NAME,
            confidence=1.0,
            confirmed=True,
            column=column,
        )

    # =================================================================
    # Candidate resolution
    # =================================================================

    def _resolve_candidates(
        self,
        candidates: list[_MappingCandidate],
    ) -> tuple[
        list[VariableMapping],
        list[HarmonizationIssue],
    ]:
        """Resolve candidates while preserving ambiguity."""

        candidates_by_canonical: dict[
            str,
            list[_MappingCandidate],
        ] = defaultdict(list)

        for candidate in candidates:
            candidates_by_canonical[
                candidate.canonical_name
            ].append(candidate)

        mappings: list[VariableMapping] = []
        issues: list[HarmonizationIssue] = []

        for (
            canonical_name,
            canonical_candidates,
        ) in candidates_by_canonical.items():
            candidates_by_location: dict[
                tuple[str, str],
                list[_MappingCandidate],
            ] = defaultdict(list)

            for candidate in canonical_candidates:
                location = (
                    candidate.site_id,
                    candidate.dataset_id,
                )

                candidates_by_location[
                    location
                ].append(candidate)

            resolved_columns: list[
                SiteColumnMapping
            ] = []

            for (
                site_id,
                dataset_id,
            ), local_candidates in (
                candidates_by_location.items()
            ):
                if len(local_candidates) > 1:
                    issues.append(
                        HarmonizationIssue(
                            issue_type=(
                                "ambiguous_mapping"
                            ),
                            message=(
                                f"Multiple columns at site "
                                f"'{site_id}' in dataset "
                                f"'{dataset_id}' map to "
                                f"'{canonical_name}'. "
                                "Human review is required."
                            ),
                            canonical_name=(
                                canonical_name
                            ),
                            site_id=site_id,
                            dataset_id=dataset_id,
                            local_columns=tuple(
                                sorted(
                                    candidate.local_name
                                    for candidate
                                    in local_candidates
                                )
                            ),
                        )
                    )

                    # Do not silently select a column.
                    continue

                candidate = local_candidates[0]

                resolved_columns.append(
                    SiteColumnMapping(
                        site_id=candidate.site_id,
                        dataset_id=(
                            candidate.dataset_id
                        ),
                        local_name=(
                            candidate.local_name
                        ),
                        mapping_method=(
                            candidate.mapping_method
                        ),
                        confidence=(
                            candidate.confidence
                        ),
                        confirmed=(
                            candidate.confirmed
                        ),
                    )
                )

            if not resolved_columns:
                continue

            mappings.append(
                VariableMapping(
                    canonical_name=canonical_name,
                    display_name=(
                        self._make_display_name(
                            canonical_name
                        )
                    ),
                    semantic_type=canonical_name,
                    site_columns=sorted(
                        resolved_columns,
                        key=lambda item: (
                            item.site_id,
                            item.dataset_id,
                            item.local_name,
                        ),
                    ),
                )
            )

        return mappings, issues

    # =================================================================
    # Cross-site consistency checks
    # =================================================================

    def _check_mapping_consistency(
        self,
        *,
        mappings: list[VariableMapping],
        site_catalogues: list[SiteCatalogue],
    ) -> list[HarmonizationIssue]:
        """Detect incompatible data types and inconsistent units."""

        column_index = self._build_column_index(
            site_catalogues
        )

        issues: list[HarmonizationIssue] = []

        for mapping in mappings:
            mapped_columns: list[
                tuple[
                    SiteColumnMapping,
                    ColumnMetadata,
                ]
            ] = []

            for site_mapping in mapping.site_columns:
                key = (
                    site_mapping.site_id,
                    site_mapping.dataset_id,
                    site_mapping.local_name,
                )

                column = column_index.get(key)

                if column is not None:
                    mapped_columns.append(
                        (
                            site_mapping,
                            column,
                        )
                    )

            type_families = {
                self._data_type_family(
                    column.data_type
                )
                for _, column in mapped_columns
            }

            if len(type_families) > 1:
                issues.append(
                    HarmonizationIssue(
                        issue_type=(
                            "incompatible_data_types"
                        ),
                        message=(
                            f"Canonical variable "
                            f"'{mapping.canonical_name}' "
                            "has potentially incompatible "
                            "data types across sites."
                        ),
                        canonical_name=(
                            mapping.canonical_name
                        ),
                        details={
                            "columns": [
                                {
                                    "site_id": (
                                        site_mapping
                                        .site_id
                                    ),
                                    "dataset_id": (
                                        site_mapping
                                        .dataset_id
                                    ),
                                    "local_name": (
                                        site_mapping
                                        .local_name
                                    ),
                                    "data_type": (
                                        column
                                        .data_type
                                        .value
                                    ),
                                }
                                for (
                                    site_mapping,
                                    column,
                                ) in mapped_columns
                            ]
                        },
                    )
                )

            units = {
                self._normalize_unit(column.unit)
                for _, column in mapped_columns
                if column.unit
            }

            if len(units) > 1:
                issues.append(
                    HarmonizationIssue(
                        issue_type="unit_mismatch",
                        message=(
                            f"Canonical variable "
                            f"'{mapping.canonical_name}' "
                            "uses different units across sites."
                        ),
                        canonical_name=(
                            mapping.canonical_name
                        ),
                        details={
                            "columns": [
                                {
                                    "site_id": (
                                        site_mapping
                                        .site_id
                                    ),
                                    "dataset_id": (
                                        site_mapping
                                        .dataset_id
                                    ),
                                    "local_name": (
                                        site_mapping
                                        .local_name
                                    ),
                                    "unit": column.unit,
                                }
                                for (
                                    site_mapping,
                                    column,
                                ) in mapped_columns
                            ]
                        },
                    )
                )

        return issues

    @staticmethod
    def _build_column_index(
        site_catalogues: list[SiteCatalogue],
    ) -> dict[
        tuple[str, str, str],
        ColumnMetadata,
    ]:
        """Index metadata by site, dataset and local column."""

        index: dict[
            tuple[str, str, str],
            ColumnMetadata,
        ] = {}

        for site_catalogue in site_catalogues:
            for dataset in site_catalogue.datasets:
                for column in dataset.columns:
                    key = (
                        site_catalogue.site_id,
                        dataset.dataset_id,
                        column.local_name,
                    )

                    index[key] = column

        return index

    # =================================================================
    # Alias configuration
    # =================================================================

    @classmethod
    def _normalize_aliases(
        cls,
        aliases: dict[
            str,
            Iterable[str],
        ],
    ) -> dict[str, frozenset[str]]:
        """Normalize externally supplied aliases."""

        normalized_aliases: dict[
            str,
            frozenset[str],
        ] = {}

        for canonical_name, aliases_for_variable in (
            aliases.items()
        ):
            normalized_canonical = (
                cls._normalize_name(
                    canonical_name
                )
            )

            normalized_values = {
                cls._normalize_name(alias)
                for alias in aliases_for_variable
            }

            normalized_values.add(
                normalized_canonical
            )

            normalized_aliases[
                normalized_canonical
            ] = frozenset(
                normalized_values
            )

        return normalized_aliases

    @staticmethod
    def _build_reverse_alias_index(
        aliases: dict[
            str,
            frozenset[str],
        ],
    ) -> dict[str, str]:
        """Build local-alias to canonical-name index."""

        reverse_index: dict[str, str] = {}

        for (
            canonical_name,
            aliases_for_variable,
        ) in aliases.items():
            for alias in aliases_for_variable:
                existing = reverse_index.get(alias)

                if (
                    existing is not None
                    and existing != canonical_name
                ):
                    raise ValueError(
                        f"Alias '{alias}' maps to both "
                        f"'{existing}' and "
                        f"'{canonical_name}'"
                    )

                reverse_index[alias] = (
                    canonical_name
                )

        return reverse_index

    # =================================================================
    # Helper methods
    # =================================================================

    @staticmethod
    def _mapping_site_count(
        mapping: VariableMapping,
    ) -> int:
        """Return the number of distinct sites in a mapping."""

        return len(
            {
                site_column.site_id
                for site_column in mapping.site_columns
            }
        )

    @staticmethod
    def _data_type_family(
        data_type: DataType,
    ) -> str:
        """Group mutually compatible data types."""

        if data_type in {
            DataType.INTEGER,
            DataType.FLOAT,
            DataType.NUMERIC,
        }:
            return "numeric"

        if data_type in {
            DataType.BOOLEAN,
            DataType.BINARY,
            DataType.CATEGORICAL,
        }:
            return "categorical"

        if data_type == DataType.DATETIME:
            return "datetime"

        if data_type in {
            DataType.STRING,
            DataType.TEXT,
        }:
            return "text"

        return "unknown"

    @staticmethod
    def _normalize_name(
        value: str,
    ) -> str:
        """Normalize a column, dataset, site or concept name."""

        normalized = value.strip().lower()

        normalized = re.sub(
            r"[^a-z0-9]+",
            "_",
            normalized,
        )

        normalized = normalized.strip("_")

        if not normalized:
            raise ValueError(
                "Name cannot be empty"
            )

        return normalized

    @staticmethod
    def _normalize_unit(
        value: str,
    ) -> str:
        """Normalize a unit for comparison."""

        return "".join(
            value.lower().split()
        )

    @staticmethod
    def _make_display_name(
        canonical_name: str,
    ) -> str:
        """Create a readable display label."""

        return canonical_name.replace(
            "_",
            " ",
        ).title()

    @staticmethod
    def _validate_unique_sites(
        site_catalogues: list[SiteCatalogue],
    ) -> None:
        """Ensure each site returned only one catalogue."""

        site_ids = [
            catalogue.site_id
            for catalogue in site_catalogues
        ]

        if len(site_ids) != len(set(site_ids)):
            raise ValueError(
                "Each site may provide only one "
                "SiteCatalogue"
            )









