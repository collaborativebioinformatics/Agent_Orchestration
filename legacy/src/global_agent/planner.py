"""Catalogue-aware planning for federated biobank analysis.

The planner converts a natural-language question into an AnalysisPlan.

It supports two stages:

1. Discovery planning, which can run without an existing catalogue.
2. Analytical planning, which uses a discovered and harmonized
   FederatedCatalogue.

The deterministic planner does not contain predefined biomedical
feature names. It extracts available columns dynamically from the
federated catalogue.

An LLM planner can later implement the same interface.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from global_agent.catalog import FederatedCatalogService
from global_agent.exceptions import PlanningError
from shared.schemas import (
    AggregationMethod,
    AnalysisPlan,
    DataType,
    FederatedCatalogue,
    FilterCondition,
    FilterOperator,
    HistogramSpecification,
    ModelSpecification,
    ModelType,
    Operation,
    PlotFormat,
    PlotSpecification,
    PlotType,
    Statistic,
)


# =====================================================================
# Planning context
# =====================================================================


@dataclass(frozen=True)
class PlanningContext:
    """Optional context supplied to the planner."""

    catalogue: FederatedCatalogue | None = None

    default_dataset_id: str | None = None

    minimum_cell_size: int = 10


# =====================================================================
# Planner interface
# =====================================================================


class Planner(ABC):
    """Interface implemented by every planner."""

    @abstractmethod
    async def create_plan(
        self,
        query: str,
        context: PlanningContext | None = None,
    ) -> AnalysisPlan:
        """Convert a user question into an AnalysisPlan."""
        raise NotImplementedError


# =====================================================================
# Catalogue-aware deterministic planner
# =====================================================================


class CatalogueAwareRulePlanner(Planner):
    """Create plans using dynamically discovered metadata.

    This planner is intentionally conservative. It resolves only:

    - canonical columns in the federated catalogue;
    - local column names represented in confirmed mappings;
    - display names from the catalogue;
    - explicitly quoted column names.

    It does not guess unknown scientific meanings.
    """

    async def create_plan(
        self,
        query: str,
        context: PlanningContext | None = None,
    ) -> AnalysisPlan:
        """Create one validated analysis plan."""

        context = context or PlanningContext()

        normalized_query = self._normalize_text(
            query
        )

        minimum_cell_size = (
            context.minimum_cell_size
        )

        if minimum_cell_size < 1:
            raise PlanningError(
                "minimum_cell_size must be at least 1."
            )

        # Dataset discovery does not require a catalogue.
        if self._is_dataset_discovery_query(
            normalized_query
        ):
            return AnalysisPlan(
                operation=Operation.DISCOVER_DATASETS,
                minimum_cell_size=minimum_cell_size,
                metadata={
                    "planner": "catalogue_aware_rules",
                    "original_query": query,
                },
            )

        # Column searching can be requested before the user knows
        # exact feature names. A catalogue is still required for
        # executing the search.
        if self._is_column_search_query(
            normalized_query
        ):
            search_text = self._extract_search_text(
                normalized_query
            )

            if not search_text:
                raise PlanningError(
                    "Please specify what kind of column "
                    "you want to search for.",
                    details={
                        "example": (
                            "Search for columns related "
                            "to inflammation"
                        )
                    },
                )

            return AnalysisPlan(
                operation=Operation.SEARCH_COLUMNS,
                dataset_id=(
                    context.default_dataset_id
                ),
                minimum_cell_size=minimum_cell_size,
                metadata={
                    "planner": "catalogue_aware_rules",
                    "search_text": search_text,
                    "page": 1,
                    "page_size": 20,
                },
            )

        if context.catalogue is None:
            raise PlanningError(
                "A federated catalogue is required before "
                "this analysis can be planned.",
                details={
                    "required_action":
                        "discover_datasets",
                },
            )

        catalogue = FederatedCatalogService(
            context.catalogue
        )

        dataset_id = self._resolve_dataset_id(
            query=normalized_query,
            catalogue=catalogue,
            default_dataset_id=(
                context.default_dataset_id
            ),
        )

        if self._is_schema_discovery_query(
            normalized_query
        ):
            return AnalysisPlan(
                operation=Operation.DISCOVER_SCHEMA,
                dataset_id=dataset_id,
                minimum_cell_size=minimum_cell_size,
                metadata={
                    "planner": "catalogue_aware_rules",
                },
            )

        columns = self._extract_columns(
            original_query=query,
            normalized_query=normalized_query,
            catalogue=catalogue,
            dataset_id=dataset_id,
        )

        requested_sites = self._extract_sites(
            normalized_query,
            catalogue,
        )

        filters = self._extract_filters(
            original_query=query,
            normalized_query=normalized_query,
            available_columns=columns,
        )

        operation = self._identify_operation(
            normalized_query,
            columns=columns,
            catalogue=catalogue,
            dataset_id=dataset_id,
        )

        group_by = self._extract_group_by(
            normalized_query=normalized_query,
            columns=columns,
        )

        analysis_columns = [
            column
            for column in columns
            if column not in group_by
        ]

        if operation == Operation.TRAIN_MODEL:
            model = self._create_model_specification(
                normalized_query=normalized_query,
                columns=columns,
            )

            return AnalysisPlan(
                operation=operation,
                dataset_id=dataset_id,
                filters=filters,
                model=model,
                requested_sites=requested_sites,
                minimum_cell_size=minimum_cell_size,
                aggregation_method=(
                    self._model_aggregation(
                        model.model_type
                    )
                ),
                metadata={
                    "planner": "catalogue_aware_rules",
                },
            )

        if operation == Operation.EVALUATE_MODEL:
            model = self._create_model_specification(
                normalized_query=normalized_query,
                columns=columns,
            )

            return AnalysisPlan(
                operation=operation,
                dataset_id=dataset_id,
                filters=filters,
                model=model,
                requested_sites=requested_sites,
                minimum_cell_size=minimum_cell_size,
                aggregation_method=(
                    AggregationMethod
                    .SUFFICIENT_STATISTICS
                ),
                metadata={
                    "planner": "catalogue_aware_rules",
                },
            )

        return self._create_exploration_plan(
            operation=operation,
            dataset_id=dataset_id,
            columns=analysis_columns,
            group_by=group_by,
            filters=filters,
            requested_sites=requested_sites,
            minimum_cell_size=minimum_cell_size,
        )

    # =================================================================
    # Discovery intent
    # =================================================================

    @staticmethod
    def _is_dataset_discovery_query(
        query: str,
    ) -> bool:
        phrases = (
            "what datasets",
            "which datasets",
            "list datasets",
            "available datasets",
            "what data are available",
            "what data is available",
            "show me the datasets",
        )

        return any(
            phrase in query
            for phrase in phrases
        )

    @staticmethod
    def _is_schema_discovery_query(
        query: str,
    ) -> bool:
        phrases = (
            "what columns",
            "which columns",
            "list columns",
            "available columns",
            "what features",
            "which features",
            "list features",
            "available features",
            "show schema",
            "describe schema",
            "dataset schema",
            "what variables",
            "which variables",
            "list variables",
            "available variables",
        )

        return any(
            phrase in query
            for phrase in phrases
        )

    @staticmethod
    def _is_column_search_query(
        query: str,
    ) -> bool:
        search_words = (
            "search for",
            "find columns",
            "find features",
            "find variables",
            "columns related to",
            "features related to",
            "variables related to",
        )

        return any(
            phrase in query
            for phrase in search_words
        )

    @staticmethod
    def _extract_search_text(
        query: str,
    ) -> str | None:
        patterns = (
            r"(?:search for|find)\s+"
            r"(?:columns|features|variables)?\s*"
            r"(?:related to|about|matching)?\s*(.+)",

            r"(?:columns|features|variables)\s+"
            r"related to\s+(.+)",
        )

        for pattern in patterns:
            match = re.search(pattern, query)

            if match:
                value = match.group(1).strip(
                    " .?!"
                )

                if value:
                    return value

        return None

    # =================================================================
    # Dataset resolution
    # =================================================================

    def _resolve_dataset_id(
        self,
        *,
        query: str,
        catalogue: FederatedCatalogService,
        default_dataset_id: str | None,
    ) -> str:
        dataset_summaries = (
            catalogue.list_datasets()
        )

        available_dataset_ids = [
            item["dataset_id"]
            for item in dataset_summaries
        ]

        explicitly_mentioned = [
            dataset_id
            for dataset_id in available_dataset_ids
            if self._contains_identifier(
                query,
                dataset_id,
            )
        ]

        if len(explicitly_mentioned) == 1:
            return explicitly_mentioned[0]

        if len(explicitly_mentioned) > 1:
            raise PlanningError(
                "The query mentions multiple datasets. "
                "Please select one dataset for this analysis.",
                details={
                    "datasets": explicitly_mentioned,
                },
            )

        if default_dataset_id is not None:
            normalized_default = (
                self._normalize_identifier(
                    default_dataset_id
                )
            )

            if (
                normalized_default
                not in available_dataset_ids
            ):
                raise PlanningError(
                    f"Default dataset "
                    f"'{normalized_default}' is not "
                    "available.",
                    details={
                        "available_datasets":
                            available_dataset_ids,
                    },
                )

            return normalized_default

        if len(available_dataset_ids) == 1:
            return available_dataset_ids[0]

        raise PlanningError(
            "Multiple datasets are available. Please "
            "select the dataset to analyse.",
            details={
                "available_datasets":
                    available_dataset_ids,
            },
        )

    # =================================================================
    # Column extraction
    # =================================================================

    def _extract_columns(
        self,
        *,
        original_query: str,
        normalized_query: str,
        catalogue: FederatedCatalogService,
        dataset_id: str,
    ) -> list[str]:
        """Extract dynamically discovered canonical columns."""

        canonical_names = (
            catalogue.canonical_columns
        )

        aliases = self._build_column_alias_index(
            catalogue=catalogue,
            dataset_id=dataset_id,
        )

        found: list[str] = []

        # Quoted or backtick-delimited names receive priority.
        quoted_names = self._extract_quoted_names(
            original_query
        )

        for quoted_name in quoted_names:
            normalized_name = (
                self._normalize_identifier(
                    quoted_name
                )
            )

            canonical = aliases.get(
                normalized_name
            )

            if canonical is None:
                raise PlanningError(
                    f"Column '{quoted_name}' was not found "
                    "in the harmonized catalogue.",
                    details={
                        "column": quoted_name,
                        "available_canonical_columns":
                            canonical_names,
                    },
                )

            if canonical not in found:
                found.append(canonical)

        # Match known canonical names and approved local/display names.
        for alias, canonical in sorted(
            aliases.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if self._contains_identifier(
                normalized_query,
                alias,
            ):
                if canonical not in found:
                    found.append(canonical)

        return found

    def _build_column_alias_index(
        self,
        *,
        catalogue: FederatedCatalogService,
        dataset_id: str,
    ) -> dict[str, str]:
        """Build metadata-driven aliases for column recognition."""

        alias_index: dict[str, str] = {}

        for mapping in (
            catalogue.catalogue.variable_mappings
        ):
            canonical = mapping.canonical_name

            self._add_column_alias(
                alias_index,
                alias=canonical,
                canonical=canonical,
            )

            if mapping.display_name:
                self._add_column_alias(
                    alias_index,
                    alias=mapping.display_name,
                    canonical=canonical,
                )

            for site_column in mapping.site_columns:
                if (
                    site_column.dataset_id
                    != dataset_id
                ):
                    continue

                # Use only confirmed local mappings for analysis.
                if not site_column.confirmed:
                    continue

                self._add_column_alias(
                    alias_index,
                    alias=site_column.local_name,
                    canonical=canonical,
                )

                try:
                    metadata = (
                        catalogue.get_local_column(
                            site_id=(
                                site_column.site_id
                            ),
                            dataset_id=(
                                site_column.dataset_id
                            ),
                            local_name=(
                                site_column.local_name
                            ),
                        )
                    )

                except Exception:
                    continue

                if metadata.display_name:
                    self._add_column_alias(
                        alias_index,
                        alias=metadata.display_name,
                        canonical=canonical,
                    )

        return alias_index

    def _add_column_alias(
        self,
        alias_index: dict[str, str],
        *,
        alias: str,
        canonical: str,
    ) -> None:
        normalized_alias = (
            self._normalize_identifier(alias)
        )

        existing = alias_index.get(
            normalized_alias
        )

        if (
            existing is not None
            and existing != canonical
        ):
            # Do not use an ambiguous alias.
            alias_index.pop(
                normalized_alias,
                None,
            )

            return

        alias_index[normalized_alias] = canonical

    @staticmethod
    def _extract_quoted_names(
        query: str,
    ) -> list[str]:
        """Extract names enclosed in quotes or backticks."""

        matches = re.findall(
            r"`([^`]+)`|\"([^\"]+)\"|'([^']+)'",
            query,
        )

        output: list[str] = []

        for match in matches:
            value = next(
                item
                for item in match
                if item
            )

            output.append(value.strip())

        return output

    # =================================================================
    # Operation identification
    # =================================================================

    def _identify_operation(
        self,
        query: str,
        *,
        columns: list[str],
        catalogue: FederatedCatalogService,
        dataset_id: str,
    ) -> Operation:
        if any(
            phrase in query
            for phrase in (
                "evaluate model",
                "evaluate the model",
                "test model",
                "test the model",
                "model performance",
            )
        ):
            return Operation.EVALUATE_MODEL

        if any(
            phrase in query
            for phrase in (
                "train model",
                "train a model",
                "build model",
                "build a model",
                "fit model",
                "fit a model",
                "predict ",
                "prediction model",
            )
        ):
            return Operation.TRAIN_MODEL

        if any(
            phrase in query
            for phrase in (
                "scatter plot",
                "scatterplot",
                "two dimensional histogram",
                "2d histogram",
                "heatmap of two",
            )
        ):
            return (
                Operation
                .TWO_DIMENSIONAL_HISTOGRAM
            )

        if any(
            phrase in query
            for phrase in (
                "correlation",
                "correlate",
                "relationship between",
                "association between",
            )
        ):
            return (
                Operation
                .CORRELATION_STATISTICS
            )

        if any(
            phrase in query
            for phrase in (
                "histogram",
                "distribution plot",
                "plot distribution",
            )
        ):
            return Operation.HISTOGRAM

        if any(
            phrase in query
            for phrase in (
                "missingness",
                "missing values",
                "missing data",
                "data completeness",
            )
        ):
            return Operation.MISSINGNESS

        if any(
            phrase in query
            for phrase in (
                "group by",
                "grouped by",
                "by category",
                "by site",
            )
        ):
            return Operation.GROUPED_SUMMARY

        if any(
            phrase in query
            for phrase in (
                "how many",
                "count rows",
                "count patients",
                "count participants",
                "cohort count",
                "cohort size",
            )
        ):
            return Operation.COHORT_COUNT

        if any(
            phrase in query
            for phrase in (
                "standardization",
                "standardisation",
                "global mean and variance",
                "global mean and standard deviation",
            )
        ):
            return (
                Operation
                .STANDARDIZATION_STATISTICS
            )

        if any(
            phrase in query
            for phrase in (
                "numeric summary",
                "summary statistics",
                "describe ",
                "mean ",
                "average ",
                "standard deviation",
                "variance",
                "minimum",
                "maximum",
            )
        ):
            return Operation.NUMERIC_SUMMARY

        if len(columns) == 1:
            if self._is_categorical_column(
                catalogue=catalogue,
                canonical_name=columns[0],
                dataset_id=dataset_id,
            ):
                return (
                    Operation
                    .CATEGORICAL_DISTRIBUTION
                )

        raise PlanningError(
            "The requested analytical operation could "
            "not be determined.",
            details={
                "detected_columns": columns,
                "suggestion": (
                    "Ask for a numeric summary, missingness, "
                    "category distribution, group-by summary, "
                    "correlation, histogram or model."
                ),
            },
        )

    # =================================================================
    # Grouping
    # =================================================================

    def _extract_group_by(
        self,
        *,
        normalized_query: str,
        columns: list[str],
    ) -> list[str]:
        """Identify which detected column follows a by-clause."""

        if not columns:
            return []

        patterns = (
            r"(?:grouped by|group by|separated by)\s+(.+)",
            r"\sby\s+(.+)",
        )

        group_text = None

        for pattern in patterns:
            match = re.search(
                pattern,
                normalized_query,
            )

            if match:
                group_text = match.group(1)
                break

        if group_text is None:
            return []

        group_columns = [
            column
            for column in columns
            if self._contains_identifier(
                group_text,
                column,
            )
        ]

        if len(group_columns) > 1:
            raise PlanningError(
                "Multiple possible grouping columns were "
                "detected. Please specify one grouping column.",
                details={
                    "grouping_candidates":
                        group_columns,
                },
            )

        return group_columns

    # =================================================================
    # Filters
    # =================================================================

    def _extract_filters(
        self,
        *,
        original_query: str,
        normalized_query: str,
        available_columns: list[str],
    ) -> list[FilterCondition]:
        """Extract conservative, column-aware filters."""

        filters: list[FilterCondition] = []

        # Generic explicit syntax:
        # where `column_name` >= 10
        comparison_pattern = re.compile(
            r"(?:where|with)\s+"
            r"[`'\"]?([a-zA-Z0-9_\- ]+?)[`'\"]?\s*"
            r"(>=|<=|!=|=|>|<)\s*"
            r"(-?\d+(?:\.\d+)?)"
        )

        comparison_match = (
            comparison_pattern.search(
                original_query
            )
        )

        if comparison_match:
            raw_column = (
                comparison_match
                .group(1)
                .strip()
            )

            canonical_column = (
                self._match_detected_column(
                    raw_column,
                    available_columns,
                )
            )

            if canonical_column is not None:
                operator = {
                    "=": FilterOperator.EQUAL,
                    "!=": FilterOperator.NOT_EQUAL,
                    ">": FilterOperator.GREATER_THAN,
                    ">=": (
                        FilterOperator
                        .GREATER_THAN_OR_EQUAL
                    ),
                    "<": FilterOperator.LESS_THAN,
                    "<=": (
                        FilterOperator
                        .LESS_THAN_OR_EQUAL
                    ),
                }[
                    comparison_match.group(2)
                ]

                numeric_value = float(
                    comparison_match.group(3)
                )

                value: int | float = (
                    int(numeric_value)
                    if numeric_value.is_integer()
                    else numeric_value
                )

                filters.append(
                    FilterCondition(
                        column=canonical_column,
                        operator=operator,
                        value=value,
                    )
                )

        # Generic availability syntax:
        # "with available `column_name`"
        for column in available_columns:
            readable = column.replace(
                "_",
                " ",
            )

            patterns = (
                f"available {readable}",
                f"{readable} available",
                f"with {readable} data",
                f"non missing {readable}",
                f"nonmissing {readable}",
            )

            if any(
                pattern in normalized_query
                for pattern in patterns
            ):
                filters.append(
                    FilterCondition(
                        column=column,
                        operator=(
                            FilterOperator
                            .IS_NOT_NULL
                        ),
                    )
                )

        return self._deduplicate_filters(
            filters
        )

    # =================================================================
    # Modelling
    # =================================================================

    def _create_model_specification(
        self,
        *,
        normalized_query: str,
        columns: list[str],
    ) -> ModelSpecification:
        """Create a model specification from detected columns."""

        if len(columns) < 2:
            raise PlanningError(
                "Model training requires a target and at "
                "least one feature.",
                details={
                    "detected_columns": columns,
                },
            )

        target = self._extract_model_target(
            normalized_query=normalized_query,
            columns=columns,
        )

        if target is None:
            raise PlanningError(
                "The model target could not be identified. "
                "Use wording such as 'predict TARGET using "
                "FEATURE1 and FEATURE2'.",
                details={
                    "detected_columns": columns,
                },
            )

        features = [
            column
            for column in columns
            if column != target
        ]

        return ModelSpecification(
            model_type=self._extract_model_type(
                normalized_query
            ),
            target=target,
            features=features,
            hyperparameters={},
            local_epochs=1,
            federated_rounds=5,
            random_seed=42,
        )

    def _extract_model_target(
        self,
        *,
        normalized_query: str,
        columns: list[str],
    ) -> str | None:
        """Extract the column following predict/target wording."""

        patterns = (
            r"predict\s+([a-zA-Z0-9_\- ]+?)"
            r"(?:\s+using|\s+from|\s+with|$)",

            r"target\s+(?:is\s+)?"
            r"([a-zA-Z0-9_\- ]+?)"
            r"(?:\s+using|\s+from|\s+with|$)",
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                normalized_query,
            )

            if not match:
                continue

            target_text = match.group(1).strip()

            for column in columns:
                if self._contains_identifier(
                    target_text,
                    column,
                ):
                    return column

        return None

    @staticmethod
    def _extract_model_type(
        query: str,
    ) -> ModelType:
        if any(
            phrase in query
            for phrase in (
                "random forest",
                "randomforest",
            )
        ):
            return ModelType.RANDOM_FOREST

        if any(
            phrase in query
            for phrase in (
                "linear svm",
                "support vector machine",
            )
        ):
            return ModelType.LINEAR_SVM

        if any(
            phrase in query
            for phrase in (
                "mlp",
                "neural network",
                "pytorch",
            )
        ):
            return ModelType.PYTORCH_MLP

        return ModelType.LOGISTIC_REGRESSION

    @staticmethod
    def _model_aggregation(
        model_type: ModelType,
    ) -> AggregationMethod:
        if model_type == ModelType.RANDOM_FOREST:
            return (
                AggregationMethod.MODEL_ENSEMBLE
            )

        return AggregationMethod.FEDAVG

    # =================================================================
    # Exploration-plan construction
    # =================================================================

    def _create_exploration_plan(
        self,
        *,
        operation: Operation,
        dataset_id: str,
        columns: list[str],
        group_by: list[str],
        filters: list[FilterCondition],
        requested_sites: list[str],
        minimum_cell_size: int,
    ) -> AnalysisPlan:
        """Construct an operation-specific exploration plan."""

        if operation == Operation.COHORT_COUNT:
            return AnalysisPlan(
                operation=operation,
                dataset_id=dataset_id,
                filters=filters,
                requested_sites=requested_sites,
                minimum_cell_size=minimum_cell_size,
                aggregation_method=AggregationMethod.SUM,
                metadata={
                    "planner": "catalogue_aware_rules",
                },
            )

        if operation == Operation.NUMERIC_SUMMARY:
            self._require_columns(
                columns,
                operation,
            )

            return AnalysisPlan(
                operation=operation,
                dataset_id=dataset_id,
                columns=columns,
                filters=filters,
                statistics=[
                    Statistic.COUNT,
                    Statistic.NON_MISSING_COUNT,
                    Statistic.SUM,
                    Statistic.SUM_OF_SQUARES,
                    Statistic.MEAN,
                    Statistic.VARIANCE,
                    Statistic.STANDARD_DEVIATION,
                    Statistic.MINIMUM,
                    Statistic.MAXIMUM,
                ],
                requested_sites=requested_sites,
                minimum_cell_size=minimum_cell_size,
                aggregation_method=(
                    AggregationMethod
                    .SUFFICIENT_STATISTICS
                ),
                metadata={
                    "planner": "catalogue_aware_rules",
                },
            )

        if (
            operation
            == Operation.CATEGORICAL_DISTRIBUTION
        ):
            if len(columns) != 1:
                raise PlanningError(
                    "A categorical distribution requires "
                    "exactly one selected column.",
                    details={
                        "detected_columns": columns,
                    },
                )

            return AnalysisPlan(
                operation=operation,
                dataset_id=dataset_id,
                columns=columns,
                filters=filters,
                statistics=[
                    Statistic.COUNT,
                ],
                requested_sites=requested_sites,
                minimum_cell_size=minimum_cell_size,
                aggregation_method=AggregationMethod.SUM,
                metadata={
                    "planner": "catalogue_aware_rules",
                },
            )

        if operation == Operation.GROUPED_SUMMARY:
            self._require_columns(
                columns,
                operation,
            )

            if not group_by:
                raise PlanningError(
                    "A grouped summary requires a "
                    "grouping column.",
                )

            return AnalysisPlan(
                operation=operation,
                dataset_id=dataset_id,
                columns=columns,
                group_by=group_by,
                filters=filters,
                statistics=[
                    Statistic.COUNT,
                ],
                requested_sites=requested_sites,
                minimum_cell_size=minimum_cell_size,
                aggregation_method=AggregationMethod.SUM,
                metadata={
                    "planner": "catalogue_aware_rules",
                },
            )

        if operation == Operation.MISSINGNESS:
            self._require_columns(
                columns,
                operation,
            )

            return AnalysisPlan(
                operation=operation,
                dataset_id=dataset_id,
                columns=columns,
                filters=filters,
                statistics=[
                    Statistic.COUNT,
                    Statistic.NON_MISSING_COUNT,
                    Statistic.MISSING_COUNT,
                    Statistic.MISSING_FRACTION,
                ],
                requested_sites=requested_sites,
                minimum_cell_size=minimum_cell_size,
                aggregation_method=AggregationMethod.SUM,
                metadata={
                    "planner": "catalogue_aware_rules",
                },
            )

        if (
            operation
            == Operation.CORRELATION_STATISTICS
        ):
            if len(columns) < 2:
                raise PlanningError(
                    "Correlation requires at least "
                    "two selected columns.",
                    details={
                        "detected_columns": columns,
                    },
                )

            return AnalysisPlan(
                operation=operation,
                dataset_id=dataset_id,
                columns=columns,
                filters=filters,
                statistics=[
                    Statistic.CORRELATION,
                ],
                requested_sites=requested_sites,
                minimum_cell_size=minimum_cell_size,
                aggregation_method=(
                    AggregationMethod
                    .SUFFICIENT_STATISTICS
                ),
                plot=PlotSpecification(
                    plot_type=(
                        PlotType
                        .CORRELATION_HEATMAP
                    ),
                    title="Federated correlation",
                    output_format=PlotFormat.PNG,
                ),
                metadata={
                    "planner": "catalogue_aware_rules",
                },
            )

        if (
            operation
            == Operation.STANDARDIZATION_STATISTICS
        ):
            self._require_columns(
                columns,
                operation,
            )

            return AnalysisPlan(
                operation=operation,
                dataset_id=dataset_id,
                columns=columns,
                filters=filters,
                statistics=[
                    Statistic.COUNT,
                    Statistic.SUM,
                    Statistic.SUM_OF_SQUARES,
                    Statistic.MEAN,
                    Statistic.VARIANCE,
                ],
                requested_sites=requested_sites,
                minimum_cell_size=minimum_cell_size,
                aggregation_method=(
                    AggregationMethod
                    .SUFFICIENT_STATISTICS
                ),
                metadata={
                    "planner": "catalogue_aware_rules",
                },
            )

        if operation == Operation.HISTOGRAM:
            if len(columns) != 1:
                raise PlanningError(
                    "A histogram requires exactly one "
                    "selected numeric column.",
                    details={
                        "detected_columns": columns,
                    },
                )

            column = columns[0]

            return AnalysisPlan(
                operation=operation,
                dataset_id=dataset_id,
                columns=[column],
                filters=filters,
                histogram=HistogramSpecification(
                    number_of_bins=20,
                ),
                plot=PlotSpecification(
                    plot_type=PlotType.HISTOGRAM,
                    title=(
                        f"Federated distribution of "
                        f"{column}"
                    ),
                    x=column,
                    output_format=PlotFormat.PNG,
                    include_site_breakdown=True,
                ),
                requested_sites=requested_sites,
                minimum_cell_size=minimum_cell_size,
                aggregation_method=AggregationMethod.SUM,
                metadata={
                    "planner": "catalogue_aware_rules",
                },
            )

        if (
            operation
            == Operation.TWO_DIMENSIONAL_HISTOGRAM
        ):
            if len(columns) != 2:
                raise PlanningError(
                    "A two-dimensional histogram requires "
                    "exactly two selected columns.",
                    details={
                        "detected_columns": columns,
                    },
                )

            return AnalysisPlan(
                operation=operation,
                dataset_id=dataset_id,
                columns=columns,
                filters=filters,
                histogram=HistogramSpecification(
                    number_of_bins=20,
                ),
                plot=PlotSpecification(
                    plot_type=(
                        PlotType
                        .TWO_DIMENSIONAL_HISTOGRAM
                    ),
                    title=(
                        f"Federated distribution of "
                        f"{columns[0]} and "
                        f"{columns[1]}"
                    ),
                    x=columns[0],
                    y=columns[1],
                    output_format=PlotFormat.PNG,
                    include_site_breakdown=False,
                ),
                requested_sites=requested_sites,
                minimum_cell_size=minimum_cell_size,
                aggregation_method=AggregationMethod.SUM,
                metadata={
                    "planner": "catalogue_aware_rules",
                },
            )

        raise PlanningError(
            f"Operation '{operation.value}' is not "
            "implemented by the rule-based planner."
        )

    # =================================================================
    # Site and data-type helpers
    # =================================================================

    @staticmethod
    def _extract_sites(
        query: str,
        catalogue: FederatedCatalogService,
    ) -> list[str]:
        return [
            site_id
            for site_id
            in catalogue.participating_sites
            if CatalogueAwareRulePlanner
            ._contains_identifier(
                query,
                site_id,
            )
        ]

    def _is_categorical_column(
        self,
        *,
        catalogue: FederatedCatalogService,
        canonical_name: str,
        dataset_id: str,
    ) -> bool:
        """Return true if mapped columns are categorical."""

        mapped_types: set[DataType] = set()

        for site_id in (
            catalogue.participating_sites
        ):
            try:
                mapping = (
                    catalogue.resolve_local_column(
                        canonical_name=canonical_name,
                        site_id=site_id,
                        dataset_id=dataset_id,
                    )
                )

                column = (
                    catalogue.get_local_column(
                        site_id=site_id,
                        dataset_id=dataset_id,
                        local_name=(
                            mapping.local_name
                        ),
                    )
                )

            except Exception:
                continue

            mapped_types.add(column.data_type)

        if not mapped_types:
            return False

        categorical_types = {
            DataType.BOOLEAN,
            DataType.BINARY,
            DataType.CATEGORICAL,
        }

        return mapped_types.issubset(
            categorical_types
        )

    # =================================================================
    # General helpers
    # =================================================================

    @staticmethod
    def _require_columns(
        columns: list[str],
        operation: Operation,
    ) -> None:
        if not columns:
            raise PlanningError(
                f"Operation '{operation.value}' requires "
                "at least one column. Select a column from "
                "the discovered catalogue."
            )

    @staticmethod
    def _match_detected_column(
        raw_name: str,
        columns: list[str],
    ) -> str | None:
        normalized_raw = (
            CatalogueAwareRulePlanner
            ._normalize_identifier(raw_name)
        )

        for column in columns:
            if (
                CatalogueAwareRulePlanner
                ._normalize_identifier(column)
                == normalized_raw
            ):
                return column

        return None

    @staticmethod
    def _deduplicate_filters(
        filters: list[FilterCondition],
    ) -> list[FilterCondition]:
        unique: list[FilterCondition] = []
        seen: set[tuple[str, str, str]] = set()

        for condition in filters:
            key = (
                condition.column,
                condition.operator.value,
                repr(condition.value),
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(condition)

        return unique

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:
        if not isinstance(value, str):
            raise PlanningError(
                "The user query must be a string."
            )

        normalized = " ".join(
            value.strip().lower().split()
        )

        if not normalized:
            raise PlanningError(
                "The user query cannot be empty."
            )

        return normalized

    @staticmethod
    def _normalize_identifier(
        value: str,
    ) -> str:
        normalized = value.strip().lower()

        normalized = re.sub(
            r"[^a-z0-9]+",
            "_",
            normalized,
        )

        normalized = normalized.strip("_")

        if not normalized:
            raise ValueError(
                "Identifier cannot be empty"
            )

        return normalized

    @staticmethod
    def _contains_identifier(
        text: str,
        identifier: str,
    ) -> bool:
        """Match underscored or spaced identifiers safely."""

        normalized_identifier = (
            CatalogueAwareRulePlanner
            ._normalize_identifier(identifier)
        )

        readable_identifier = (
            normalized_identifier.replace(
                "_",
                " ",
            )
        )

        normalized_text = re.sub(
            r"[^a-z0-9]+",
            " ",
            text.lower(),
        )

        normalized_text = " ".join(
            normalized_text.split()
        )

        pattern = (
            r"(?:^|\s)"
            + re.escape(readable_identifier)
            + r"(?:$|\s)"
        )

        return bool(
            re.search(
                pattern,
                normalized_text,
            )
        )