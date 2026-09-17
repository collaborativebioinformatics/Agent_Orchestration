"""Catalogue-aware policy validation for the global agent.

This module validates whether an AnalysisPlan is permitted and
executable using the currently discovered federated catalogue.

 Columns are validated dynamically using:

- the federated catalogue;
- confirmed canonical-to-local mappings;
- local column permissions;
- local sensitivity classifications.

The policy does not:

- access patient-level data;
- execute NVIDIA FLARE;
- calculate statistics;
- modify the submitted plan;
- use an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from global_agent.catalog import (
    FederatedCatalogService,
)
from global_agent.exceptions import (
    CatalogueError,
    PolicyViolation,
)
from shared.schemas import (
    AggregationMethod,
    AnalysisPlan,
    FederatedCatalogue,
    ModelType,
    Operation,
    PlotType,
)


ColumnUsage = Literal[
    "analysis",
    "grouping",
    "filtering",
    "modeling",
]


# =====================================================================
# Default policy configuration
# =====================================================================


DEFAULT_ALLOWED_OPERATIONS = frozenset(
    {
        Operation.DISCOVER_DATASETS,
        Operation.DISCOVER_SCHEMA,
        Operation.SEARCH_COLUMNS,
        Operation.PROFILE_COLUMNS,

        Operation.COHORT_COUNT,
        Operation.NUMERIC_SUMMARY,
        Operation.CATEGORICAL_DISTRIBUTION,
        Operation.GROUPED_SUMMARY,
        Operation.MISSINGNESS,
        Operation.CORRELATION_STATISTICS,
        Operation.STANDARDIZATION_STATISTICS,
        Operation.HISTOGRAM,
        Operation.TWO_DIMENSIONAL_HISTOGRAM,

        Operation.TRAIN_MODEL,
        Operation.EVALUATE_MODEL,
    }
)


DEFAULT_ALLOWED_MODELS = frozenset(
    {
        ModelType.LOGISTIC_REGRESSION,
        ModelType.LINEAR_SVM,
        ModelType.RANDOM_FOREST,
        ModelType.PYTORCH_MLP,
    }
)


DEFAULT_MODEL_HYPERPARAMETERS: dict[
    ModelType,
    frozenset[str],
] = {
    ModelType.LOGISTIC_REGRESSION: frozenset(
        {
            "learning_rate",
            "weight_decay",
            "class_weight",
            "batch_size",
        }
    ),
    ModelType.LINEAR_SVM: frozenset(
        {
            "learning_rate",
            "weight_decay",
            "class_weight",
            "batch_size",
        }
    ),
    ModelType.RANDOM_FOREST: frozenset(
        {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "class_weight",
        }
    ),
    ModelType.PYTORCH_MLP: frozenset(
        {
            "learning_rate",
            "weight_decay",
            "hidden_dimensions",
            "dropout",
            "batch_size",
            "class_weight",
        }
    ),
}


DEFAULT_OPERATION_AGGREGATIONS: dict[
    Operation,
    frozenset[AggregationMethod],
] = {
    Operation.COHORT_COUNT: frozenset(
        {
            AggregationMethod.SUM,
        }
    ),
    Operation.NUMERIC_SUMMARY: frozenset(
        {
            AggregationMethod.SUFFICIENT_STATISTICS,
        }
    ),
    Operation.CATEGORICAL_DISTRIBUTION: frozenset(
        {
            AggregationMethod.SUM,
        }
    ),
    Operation.GROUPED_SUMMARY: frozenset(
        {
            AggregationMethod.SUM,
            AggregationMethod.SUFFICIENT_STATISTICS,
        }
    ),
    Operation.MISSINGNESS: frozenset(
        {
            AggregationMethod.SUM,
        }
    ),
    Operation.CORRELATION_STATISTICS: frozenset(
        {
            AggregationMethod.SUFFICIENT_STATISTICS,
        }
    ),
    Operation.STANDARDIZATION_STATISTICS: frozenset(
        {
            AggregationMethod.SUFFICIENT_STATISTICS,
        }
    ),
    Operation.HISTOGRAM: frozenset(
        {
            AggregationMethod.SUM,
        }
    ),
    Operation.TWO_DIMENSIONAL_HISTOGRAM: frozenset(
        {
            AggregationMethod.SUM,
        }
    ),
    Operation.EVALUATE_MODEL: frozenset(
        {
            AggregationMethod.SUM,
            AggregationMethod.WEIGHTED_MEAN,
            AggregationMethod.SUFFICIENT_STATISTICS,
        }
    ),
}


DEFAULT_MODEL_AGGREGATIONS: dict[
    ModelType,
    frozenset[AggregationMethod],
] = {
    ModelType.LOGISTIC_REGRESSION: frozenset(
        {
            AggregationMethod.FEDAVG,
        }
    ),
    ModelType.LINEAR_SVM: frozenset(
        {
            AggregationMethod.FEDAVG,
        }
    ),
    ModelType.PYTORCH_MLP: frozenset(
        {
            AggregationMethod.FEDAVG,
        }
    ),
    ModelType.RANDOM_FOREST: frozenset(
        {
            AggregationMethod.MODEL_ENSEMBLE,
        }
    ),
}


DEFAULT_PROHIBITED_METADATA_KEYS = frozenset(
    {
        "patient_records",
        "patient_data",
        "raw_data",
        "raw_rows",
        "identifiers",
        "direct_identifiers",
        "sql",
        "query_code",
        "python_code",
        "source_code",
        "shell_command",
        "command",
    }
)


DISCOVERY_OPERATIONS = frozenset(
    {
        Operation.DISCOVER_DATASETS,
        Operation.DISCOVER_SCHEMA,
        Operation.SEARCH_COLUMNS,
        Operation.PROFILE_COLUMNS,
    }
)


MODEL_OPERATIONS = frozenset(
    {
        Operation.TRAIN_MODEL,
        Operation.EVALUATE_MODEL,
    }
)


# =====================================================================
# Policy configuration
# =====================================================================


@dataclass(frozen=True)
class PolicyConfig:
    """Configuration for catalogue-aware policy validation."""

    allowed_operations: frozenset[Operation] = field(
        default_factory=lambda:
            DEFAULT_ALLOWED_OPERATIONS
    )

    allowed_models: frozenset[ModelType] = field(
        default_factory=lambda:
            DEFAULT_ALLOWED_MODELS
    )

    allowed_model_hyperparameters: Mapping[
        ModelType,
        frozenset[str],
    ] = field(
        default_factory=lambda:
            DEFAULT_MODEL_HYPERPARAMETERS.copy()
    )

    allowed_operation_aggregations: Mapping[
        Operation,
        frozenset[AggregationMethod],
    ] = field(
        default_factory=lambda:
            DEFAULT_OPERATION_AGGREGATIONS.copy()
    )

    allowed_model_aggregations: Mapping[
        ModelType,
        frozenset[AggregationMethod],
    ] = field(
        default_factory=lambda:
            DEFAULT_MODEL_AGGREGATIONS.copy()
    )

    prohibited_metadata_keys: frozenset[str] = field(
        default_factory=lambda:
            DEFAULT_PROHIBITED_METADATA_KEYS
    )

    # If None, site eligibility is determined by the catalogue.
    globally_allowed_sites: frozenset[str] | None = None

    minimum_cell_size: int = 10

    minimum_eligible_sites: int = 2

    maximum_local_epochs: int = 10

    maximum_federated_rounds: int = 50

    maximum_histogram_bins: int = 100

    maximum_search_page_size: int = 100


# =====================================================================
# Policy decision
# =====================================================================


@dataclass(frozen=True)
class PolicyDecision:
    """Result of successful policy validation."""

    request_id: str

    approved: bool

    eligible_sites: tuple[str, ...]

    excluded_sites: dict[str, str]

    # canonical column -> site -> local column
    resolved_columns: dict[
        str,
        dict[str, str],
    ]

    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "request_id": self.request_id,
            "approved": self.approved,
            "eligible_sites": list(
                self.eligible_sites
            ),
            "excluded_sites": dict(
                self.excluded_sites
            ),
            "resolved_columns": {
                canonical: dict(site_columns)
                for canonical, site_columns
                in self.resolved_columns.items()
            },
            "warnings": list(self.warnings),
        }


# =====================================================================
# Global policy
# =====================================================================


class GlobalPolicy:
    """Validate plans against governance and catalogue metadata."""

    def __init__(
        self,
        config: PolicyConfig | None = None,
    ) -> None:
        self.config = config or PolicyConfig()

        self._validate_configuration()

    def validate(
        self,
        plan: AnalysisPlan,
        *,
        catalogue: (
            FederatedCatalogue
            | FederatedCatalogService
            | None
        ) = None,
    ) -> PolicyDecision:
        """Validate a plan and return an execution decision.

        Discovery operations may be validated without a catalogue.

        Analytical and modelling operations require a discovered and
        harmonized catalogue.
        """

        self._validate_operation(plan)
        self._validate_minimum_cell_size(plan)
        self._validate_metadata(plan)
        self._validate_aggregation(plan)
        self._validate_histogram(plan)
        self._validate_plot(plan)

        if plan.model is not None:
            self._validate_model(plan)

        catalog_service = self._coerce_catalogue(
            catalogue
        )

        if plan.operation in DISCOVERY_OPERATIONS:
            return self._validate_discovery_plan(
                plan=plan,
                catalogue=catalog_service,
            )

        if catalog_service is None:
            self._raise_violation(
                plan=plan,
                message=(
                    "A discovered and harmonized catalogue "
                    "is required before analytical execution."
                ),
                rule="catalogue_required",
                details={
                    "required_action":
                        Operation.DISCOVER_DATASETS.value,
                },
            )

        return self._validate_analytical_plan(
            plan=plan,
            catalogue=catalog_service,
        )

    # =================================================================
    # Policy configuration validation
    # =================================================================

    def _validate_configuration(self) -> None:
        if self.config.minimum_cell_size < 1:
            raise ValueError(
                "minimum_cell_size must be at least 1"
            )

        if self.config.minimum_eligible_sites < 1:
            raise ValueError(
                "minimum_eligible_sites must be at least 1"
            )

        if self.config.maximum_local_epochs < 1:
            raise ValueError(
                "maximum_local_epochs must be at least 1"
            )

        if self.config.maximum_federated_rounds < 1:
            raise ValueError(
                "maximum_federated_rounds must be at least 1"
            )

        if self.config.maximum_histogram_bins < 2:
            raise ValueError(
                "maximum_histogram_bins must be at least 2"
            )

        if self.config.maximum_search_page_size < 1:
            raise ValueError(
                "maximum_search_page_size must be at least 1"
            )

    # =================================================================
    # General validation
    # =================================================================

    def _validate_operation(
        self,
        plan: AnalysisPlan,
    ) -> None:
        if (
            plan.operation
            not in self.config.allowed_operations
        ):
            self._raise_violation(
                plan=plan,
                message=(
                    f"Operation '{plan.operation.value}' "
                    "is not permitted."
                ),
                rule="allowed_operations",
                details={
                    "operation": plan.operation.value,
                },
            )

    def _validate_minimum_cell_size(
        self,
        plan: AnalysisPlan,
    ) -> None:
        if (
            plan.minimum_cell_size
            < self.config.minimum_cell_size
        ):
            self._raise_violation(
                plan=plan,
                message=(
                    f"Requested minimum cell size "
                    f"{plan.minimum_cell_size} is below "
                    f"the policy minimum of "
                    f"{self.config.minimum_cell_size}."
                ),
                rule="minimum_cell_size",
                details={
                    "requested":
                        plan.minimum_cell_size,
                    "required":
                        self.config.minimum_cell_size,
                },
            )

    def _validate_metadata(
        self,
        plan: AnalysisPlan,
    ) -> None:
        prohibited_keys = (
            self._find_prohibited_metadata_keys(
                plan.metadata
            )
        )

        if prohibited_keys:
            self._raise_violation(
                plan=plan,
                message=(
                    "The plan contains prohibited metadata "
                    "fields: "
                    + ", ".join(
                        sorted(prohibited_keys)
                    )
                ),
                rule="prohibited_metadata",
                details={
                    "keys": sorted(
                        prohibited_keys
                    ),
                },
            )

    def _find_prohibited_metadata_keys(
        self,
        value: Any,
    ) -> set[str]:
        """Search nested metadata for prohibited keys."""

        found: set[str] = set()

        if isinstance(value, Mapping):
            for key, nested_value in value.items():
                normalized_key = (
                    str(key).strip().lower()
                )

                if (
                    normalized_key
                    in self.config
                    .prohibited_metadata_keys
                ):
                    found.add(normalized_key)

                found.update(
                    self._find_prohibited_metadata_keys(
                        nested_value
                    )
                )

        elif isinstance(value, (list, tuple)):
            for item in value:
                found.update(
                    self._find_prohibited_metadata_keys(
                        item
                    )
                )

        return found

    # =================================================================
    # Discovery validation
    # =================================================================

    def _validate_discovery_plan(
        self,
        *,
        plan: AnalysisPlan,
        catalogue: FederatedCatalogService | None,
    ) -> PolicyDecision:
        """Validate metadata-discovery operations."""

        if (
            plan.operation
            == Operation.SEARCH_COLUMNS
        ):
            search_text = plan.metadata.get(
                "search_text"
            )

            if (
                not isinstance(search_text, str)
                or not search_text.strip()
            ):
                self._raise_violation(
                    plan=plan,
                    message=(
                        "search_columns requires a "
                        "non-empty search_text."
                    ),
                    rule="search_text",
                )

            page = plan.metadata.get("page", 1)
            page_size = plan.metadata.get(
                "page_size",
                20,
            )

            if (
                not isinstance(page, int)
                or isinstance(page, bool)
                or page < 1
            ):
                self._raise_violation(
                    plan=plan,
                    message=(
                        "Search page must be a positive integer."
                    ),
                    rule="search_page",
                )

            if (
                not isinstance(page_size, int)
                or isinstance(page_size, bool)
                or page_size < 1
                or page_size
                > self.config.maximum_search_page_size
            ):
                self._raise_violation(
                    plan=plan,
                    message=(
                        "Search page_size must be between "
                        f"1 and "
                        f"{self.config.maximum_search_page_size}."
                    ),
                    rule="search_page_size",
                )

        candidate_sites = self._candidate_sites(
            plan=plan,
            catalogue=catalogue,
            allow_without_catalogue=True,
        )

        if (
            catalogue is not None
            and plan.dataset_id is not None
        ):
            available_sites = []
            excluded_sites = {}

            for site_id in candidate_sites:
                try:
                    catalogue.get_dataset(
                        site_id=site_id,
                        dataset_id=plan.dataset_id,
                    )
                    available_sites.append(site_id)

                except CatalogueError as error:
                    excluded_sites[site_id] = str(error)

            if not available_sites:
                self._raise_violation(
                    plan=plan,
                    message=(
                        f"Dataset '{plan.dataset_id}' "
                        "is not available at any selected site."
                    ),
                    rule="dataset_availability",
                )

            return PolicyDecision(
                request_id=str(plan.request_id),
                approved=True,
                eligible_sites=tuple(
                    sorted(available_sites)
                ),
                excluded_sites=excluded_sites,
                resolved_columns={},
                warnings=self._exclusion_warnings(
                    excluded_sites
                ),
            )

        return PolicyDecision(
            request_id=str(plan.request_id),
            approved=True,
            eligible_sites=tuple(
                sorted(candidate_sites)
            ),
            excluded_sites={},
            resolved_columns={},
            warnings=(),
        )

    # =================================================================
    # Analytical validation
    # =================================================================

    def _validate_analytical_plan(
        self,
        *,
        plan: AnalysisPlan,
        catalogue: FederatedCatalogService,
    ) -> PolicyDecision:
        """Validate dataset and column eligibility by site."""

        if plan.dataset_id is None:
            self._raise_violation(
                plan=plan,
                message=(
                    "Analytical operations require dataset_id."
                ),
                rule="dataset_required",
            )

        candidate_sites = self._candidate_sites(
            plan=plan,
            catalogue=catalogue,
        )

        requirements = (
            self._collect_column_requirements(plan)
        )

        eligible_sites: list[str] = []

        excluded_sites: dict[str, str] = {}

        resolved_columns: dict[
            str,
            dict[str, str],
        ] = defaultdict_dict()

        for site_id in candidate_sites:
            try:
                catalogue.get_dataset(
                    site_id=site_id,
                    dataset_id=plan.dataset_id,
                )

                site_resolutions = (
                    self._validate_site_requirements(
                        plan=plan,
                        catalogue=catalogue,
                        site_id=site_id,
                        requirements=requirements,
                    )
                )

            except CatalogueError as error:
                excluded_sites[site_id] = str(error)
                continue

            eligible_sites.append(site_id)

            for canonical_name, local_name in (
                site_resolutions.items()
            ):
                resolved_columns[
                    canonical_name
                ][site_id] = local_name

        required_site_count = (
            self.config.minimum_eligible_sites
        )

        if len(eligible_sites) < required_site_count:
            self._raise_violation(
                plan=plan,
                message=(
                    "Insufficient eligible biobank sites. "
                    f"At least {required_site_count} sites "
                    f"are required, but only "
                    f"{len(eligible_sites)} are eligible."
                ),
                rule="minimum_eligible_sites",
                details={
                    "required": required_site_count,
                    "eligible_sites": sorted(
                        eligible_sites
                    ),
                    "excluded_sites":
                        excluded_sites,
                },
            )

        return PolicyDecision(
            request_id=str(plan.request_id),
            approved=True,
            eligible_sites=tuple(
                sorted(eligible_sites)
            ),
            excluded_sites=excluded_sites,
            resolved_columns={
                canonical: dict(site_columns)
                for canonical, site_columns
                in resolved_columns.items()
            },
            warnings=self._exclusion_warnings(
                excluded_sites
            ),
        )

    def _validate_site_requirements(
        self,
        *,
        plan: AnalysisPlan,
        catalogue: FederatedCatalogService,
        site_id: str,
        requirements: list[
            tuple[str, ColumnUsage]
        ],
    ) -> dict[str, str]:
        """Validate all required columns at one site."""

        resolved: dict[str, str] = {}

        for canonical_name, usage in requirements:
            mapping = (
                catalogue.resolve_local_column(
                    canonical_name=canonical_name,
                    site_id=site_id,
                    dataset_id=plan.dataset_id,
                    require_confirmed=True,
                )
            )

            column = catalogue.get_local_column(
                site_id=site_id,
                dataset_id=mapping.dataset_id,
                local_name=mapping.local_name,
            )

            if not self._usage_is_permitted(
                column=column,
                usage=usage,
            ):
                raise CatalogueError(
                    f"Column '{canonical_name}' is not "
                    f"permitted for {usage} at site "
                    f"'{site_id}'.",
                    request_id=plan.request_id,
                    details={
                        "canonical_name":
                            canonical_name,
                        "site_id": site_id,
                        "dataset_id":
                            mapping.dataset_id,
                        "local_name":
                            mapping.local_name,
                        "usage": usage,
                    },
                )

            resolved[canonical_name] = (
                mapping.local_name
            )

        return resolved

    @staticmethod
    def _collect_column_requirements(
        plan: AnalysisPlan,
    ) -> list[tuple[str, ColumnUsage]]:
        """Collect columns and their intended use."""

        requirements: list[
            tuple[str, ColumnUsage]
        ] = []

        requirements.extend(
            (column, "analysis")
            for column in plan.columns
        )

        requirements.extend(
            (column, "grouping")
            for column in plan.group_by
        )

        requirements.extend(
            (condition.column, "filtering")
            for condition in plan.filters
        )

        if plan.model is not None:
            requirements.append(
                (
                    plan.model.target,
                    "modeling",
                )
            )

            requirements.extend(
                (
                    feature,
                    "modeling",
                )
                for feature in plan.model.features
            )

        # Remove exact duplicates while preserving order.
        unique: list[
            tuple[str, ColumnUsage]
        ] = []

        seen: set[
            tuple[str, ColumnUsage]
        ] = set()

        for requirement in requirements:
            if requirement in seen:
                continue

            seen.add(requirement)
            unique.append(requirement)

        return unique

    @staticmethod
    def _usage_is_permitted(
        *,
        column,
        usage: ColumnUsage,
    ) -> bool:
        permissions = {
            "analysis":
                column.allowed_for_analysis,
            "grouping":
                column.allowed_for_grouping,
            "filtering":
                column.allowed_for_filtering,
            "modeling":
                column.allowed_for_modeling,
        }

        return permissions[usage]

    # =================================================================
    # Site validation
    # =================================================================

    def _candidate_sites(
        self,
        *,
        plan: AnalysisPlan,
        catalogue: FederatedCatalogService | None,
        allow_without_catalogue: bool = False,
    ) -> list[str]:
        """Determine globally permitted candidate sites."""

        if catalogue is None:
            if not allow_without_catalogue:
                self._raise_violation(
                    plan=plan,
                    message=(
                        "A catalogue is required to resolve sites."
                    ),
                    rule="catalogue_required",
                )

            requested = list(
                plan.requested_sites
            )

            if (
                self.config.globally_allowed_sites
                is not None
            ):
                disallowed = (
                    set(requested)
                    - self.config.globally_allowed_sites
                )

                if disallowed:
                    self._raise_violation(
                        plan=plan,
                        message=(
                            "Requested sites are not globally "
                            "permitted: "
                            + ", ".join(
                                sorted(disallowed)
                            )
                        ),
                        rule="globally_allowed_sites",
                    )

            return requested

        participating = set(
            catalogue.participating_sites
        )

        if plan.requested_sites:
            requested = set(
                plan.requested_sites
            )

            unknown = requested - participating

            if unknown:
                self._raise_violation(
                    plan=plan,
                    message=(
                        "Requested sites are not participating: "
                        + ", ".join(sorted(unknown))
                    ),
                    rule="participating_sites",
                    details={
                        "sites": sorted(unknown),
                    },
                )

            candidate_sites = requested

        else:
            candidate_sites = participating

        if (
            self.config.globally_allowed_sites
            is not None
        ):
            disallowed = (
                candidate_sites
                - self.config.globally_allowed_sites
            )

            if disallowed:
                self._raise_violation(
                    plan=plan,
                    message=(
                        "Sites are not globally permitted: "
                        + ", ".join(
                            sorted(disallowed)
                        )
                    ),
                    rule="globally_allowed_sites",
                    details={
                        "sites": sorted(disallowed),
                    },
                )

        return sorted(candidate_sites)

    # =================================================================
    # Aggregation validation
    # =================================================================

    def _validate_aggregation(
        self,
        plan: AnalysisPlan,
    ) -> None:
        if plan.aggregation_method is None:
            return

        if (
            plan.operation
            == Operation.TRAIN_MODEL
            and plan.model is not None
        ):
            permitted = (
                self.config
                .allowed_model_aggregations
                .get(
                    plan.model.model_type,
                    frozenset(),
                )
            )

            if (
                plan.aggregation_method
                not in permitted
            ):
                self._raise_violation(
                    plan=plan,
                    message=(
                        f"Aggregation "
                        f"'{plan.aggregation_method.value}' "
                        f"is not permitted for model "
                        f"'{plan.model.model_type.value}'."
                    ),
                    rule=(
                        "model_aggregation_compatibility"
                    ),
                    details={
                        "model":
                            plan.model.model_type.value,
                        "aggregation":
                            plan
                            .aggregation_method
                            .value,
                        "permitted": sorted(
                            method.value
                            for method in permitted
                        ),
                    },
                )

            return

        if plan.operation in DISCOVERY_OPERATIONS:
            self._raise_violation(
                plan=plan,
                message=(
                    "Discovery operations should not define "
                    "a numerical aggregation method."
                ),
                rule="discovery_aggregation",
            )

        permitted = (
            self.config
            .allowed_operation_aggregations
            .get(
                plan.operation,
                frozenset(),
            )
        )

        if plan.aggregation_method not in permitted:
            self._raise_violation(
                plan=plan,
                message=(
                    f"Aggregation "
                    f"'{plan.aggregation_method.value}' "
                    f"is not permitted for operation "
                    f"'{plan.operation.value}'."
                ),
                rule=(
                    "operation_aggregation_compatibility"
                ),
                details={
                    "operation":
                        plan.operation.value,
                    "aggregation":
                        plan
                        .aggregation_method
                        .value,
                    "permitted": sorted(
                        method.value
                        for method in permitted
                    ),
                },
            )

    # =================================================================
    # Model validation
    # =================================================================

    def _validate_model(
        self,
        plan: AnalysisPlan,
    ) -> None:
        model = plan.model

        if model is None:
            return

        if (
            model.model_type
            not in self.config.allowed_models
        ):
            self._raise_violation(
                plan=plan,
                message=(
                    f"Model '{model.model_type.value}' "
                    "is not permitted."
                ),
                rule="allowed_models",
            )

        if (
            model.local_epochs
            > self.config.maximum_local_epochs
        ):
            self._raise_violation(
                plan=plan,
                message=(
                    f"Requested local epochs "
                    f"{model.local_epochs} exceed the "
                    f"maximum of "
                    f"{self.config.maximum_local_epochs}."
                ),
                rule="maximum_local_epochs",
            )

        if (
            model.federated_rounds
            > self.config.maximum_federated_rounds
        ):
            self._raise_violation(
                plan=plan,
                message=(
                    f"Requested federated rounds "
                    f"{model.federated_rounds} exceed the "
                    f"maximum of "
                    f"{self.config.maximum_federated_rounds}."
                ),
                rule="maximum_federated_rounds",
            )

        allowed_parameters = (
            self.config
            .allowed_model_hyperparameters
            .get(
                model.model_type,
                frozenset(),
            )
        )

        unsupported = (
            set(model.hyperparameters)
            - allowed_parameters
        )

        if unsupported:
            self._raise_violation(
                plan=plan,
                message=(
                    "Unsupported hyperparameters for "
                    f"'{model.model_type.value}': "
                    + ", ".join(
                        sorted(unsupported)
                    )
                ),
                rule="allowed_model_hyperparameters",
                details={
                    "model":
                        model.model_type.value,
                    "hyperparameters":
                        sorted(unsupported),
                },
            )

    # =================================================================
    # Plot and histogram validation
    # =================================================================

    def _validate_histogram(
        self,
        plan: AnalysisPlan,
    ) -> None:
        if plan.histogram is None:
            return

        if (
            plan.histogram.number_of_bins
            > self.config.maximum_histogram_bins
        ):
            self._raise_violation(
                plan=plan,
                message=(
                    f"Histogram requests "
                    f"{plan.histogram.number_of_bins} bins, "
                    f"exceeding the policy maximum of "
                    f"{self.config.maximum_histogram_bins}."
                ),
                rule="maximum_histogram_bins",
            )

    def _validate_plot(
        self,
        plan: AnalysisPlan,
    ) -> None:
        if plan.plot is None:
            return

        expected_types = {
            Operation.HISTOGRAM: {
                PlotType.HISTOGRAM,
            },
            Operation.TWO_DIMENSIONAL_HISTOGRAM: {
                PlotType.TWO_DIMENSIONAL_HISTOGRAM,
            },
            Operation.CORRELATION_STATISTICS: {
                PlotType.CORRELATION_HEATMAP,
            },
            Operation.CATEGORICAL_DISTRIBUTION: {
                PlotType.BAR,
            },
            Operation.GROUPED_SUMMARY: {
                PlotType.BAR,
                PlotType.GROUPED_BAR,
                PlotType.LINE,
            },
        }

        permitted = expected_types.get(
            plan.operation
        )

        if (
            permitted is not None
            and plan.plot.plot_type
            not in permitted
        ):
            self._raise_violation(
                plan=plan,
                message=(
                    f"Plot type "
                    f"'{plan.plot.plot_type.value}' is not "
                    f"compatible with operation "
                    f"'{plan.operation.value}'."
                ),
                rule="plot_operation_compatibility",
            )

    # =================================================================
    # Helpers
    # =================================================================

    @staticmethod
    def _coerce_catalogue(
        catalogue: (
            FederatedCatalogue
            | FederatedCatalogService
            | None
        ),
    ) -> FederatedCatalogService | None:
        if catalogue is None:
            return None

        if isinstance(
            catalogue,
            FederatedCatalogService,
        ):
            return catalogue

        return FederatedCatalogService(catalogue)

    @staticmethod
    def _exclusion_warnings(
        excluded_sites: dict[str, str],
    ) -> tuple[str, ...]:
        if not excluded_sites:
            return ()

        return (
            "Some participating sites were excluded "
            "because the requested dataset or confirmed "
            "column mappings were unavailable.",
        )

    @staticmethod
    def _raise_violation(
        *,
        plan: AnalysisPlan,
        message: str,
        rule: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        safe_details = dict(details or {})
        safe_details["rule"] = rule

        raise PolicyViolation(
            message,
            request_id=plan.request_id,
            details=safe_details,
        )


def defaultdict_dict() -> dict[
    str,
    dict[str, str],
]:
    """Create the nested dictionary used for resolved columns."""

    from collections import defaultdict

    return defaultdict(dict)