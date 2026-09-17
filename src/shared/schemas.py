"""Shared schemas for federated biobank exploration and modelling.

These schemas define the communication contract between:

- the global agent;
- NVIDIA FLARE workflows;
- local biobank agents;
- federated aggregators;
- visualization services;
- the chatbot API.

The schemas contain no patient data processing or model execution.
They only validate structured messages and metadata.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# =====================================================================
# General enums
# =====================================================================


class DataType(str, Enum):
    """General data types reported by local biobanks."""

    INTEGER = "integer"
    FLOAT = "float"
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    STRING = "string"
    DATETIME = "datetime"
    TEXT = "text"
    UNKNOWN = "unknown"


class SensitivityLevel(str, Enum):
    """Sensitivity classification of a local column."""

    NON_SENSITIVE = "non_sensitive"
    SENSITIVE = "sensitive"
    QUASI_IDENTIFIER = "quasi_identifier"
    DIRECT_IDENTIFIER = "direct_identifier"


class MappingMethod(str, Enum):
    """How a local-to-canonical variable mapping was created."""

    EXACT_NAME = "exact_name"
    DATA_DICTIONARY = "data_dictionary"
    STANDARD_VOCABULARY = "standard_vocabulary"
    RULE_BASED = "rule_based"
    LLM_SUGGESTED = "llm_suggested"
    HUMAN_CONFIRMED = "human_confirmed"


class Operation(str, Enum):
    """High-level operations supported by the system."""

    # Dataset and schema discovery
    DISCOVER_DATASETS = "discover_datasets"
    DISCOVER_SCHEMA = "discover_schema"
    SEARCH_COLUMNS = "search_columns"
    PROFILE_COLUMNS = "profile_columns"

    # Federated exploratory analysis
    COHORT_COUNT = "cohort_count"
    NUMERIC_SUMMARY = "numeric_summary"
    CATEGORICAL_DISTRIBUTION = (
        "categorical_distribution"
    )
    GROUPED_SUMMARY = "grouped_summary"
    MISSINGNESS = "missingness"
    CORRELATION_STATISTICS = (
        "correlation_statistics"
    )
    STANDARDIZATION_STATISTICS = (
        "standardization_statistics"
    )

    # Privacy-preserving plotting data
    HISTOGRAM = "histogram"
    TWO_DIMENSIONAL_HISTOGRAM = (
        "two_dimensional_histogram"
    )

    # Federated machine learning
    TRAIN_MODEL = "train_model"
    EVALUATE_MODEL = "evaluate_model"


class FilterOperator(str, Enum):
    """Supported cohort-filter operators."""

    EQUAL = "eq"
    NOT_EQUAL = "ne"

    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "ge"

    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "le"

    BETWEEN = "between"

    IN = "in"
    NOT_IN = "not_in"

    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class Statistic(str, Enum):
    """Statistics that may be requested."""

    COUNT = "count"
    NON_MISSING_COUNT = "non_missing_count"
    MISSING_COUNT = "missing_count"
    MISSING_FRACTION = "missing_fraction"

    SUM = "sum"
    SUM_OF_SQUARES = "sum_of_squares"

    MEAN = "mean"
    STANDARD_DEVIATION = "standard_deviation"
    VARIANCE = "variance"

    MINIMUM = "minimum"
    MAXIMUM = "maximum"

    MEDIAN = "median"
    QUANTILES = "quantiles"

    PREVALENCE = "prevalence"
    CORRELATION = "correlation"


class AggregationMethod(str, Enum):
    """Methods for federating site outputs."""

    SUM = "sum"
    WEIGHTED_MEAN = "weighted_mean"
    SUFFICIENT_STATISTICS = "sufficient_statistics"

    FEDAVG = "fedavg"
    MODEL_ENSEMBLE = "model_ensemble"


class PlotType(str, Enum):
    """Supported privacy-aware visualization types."""

    HISTOGRAM = "histogram"
    BAR = "bar"
    GROUPED_BAR = "grouped_bar"
    LINE = "line"
    CORRELATION_HEATMAP = "correlation_heatmap"
    TWO_DIMENSIONAL_HISTOGRAM = (
        "two_dimensional_histogram"
    )


class PlotFormat(str, Enum):
    """Supported downloadable plot formats."""

    PNG = "png"
    PDF = "pdf"
    SVG = "svg"


class ModelType(str, Enum):
    """Supported predictive models."""

    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_SVM = "linear_svm"
    RANDOM_FOREST = "random_forest"
    PYTORCH_MLP = "pytorch_mlp"


class ResponseStatus(str, Enum):
    """Possible local-site response statuses."""

    SUCCESS = "success"
    SUPPRESSED = "suppressed"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


# =====================================================================
# Local dataset catalogue
# =====================================================================


class ColumnMetadata(BaseModel):
    """Privacy-safe metadata describing one local column.

    This object describes the column but contains no row-level values.
    """

    model_config = ConfigDict(extra="forbid")

    local_name: str = Field(min_length=1)

    display_name: str | None = None

    data_type: DataType

    semantic_type: str | None = Field(
        default=None,
        description=(
            "Optional canonical concept, such as age, BMI, "
            "diabetes or systolic blood pressure."
        ),
    )

    description: str | None = None

    unit: str | None = None

    missing_fraction: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    unique_count: int | None = Field(
        default=None,
        ge=0,
    )

    sensitivity: SensitivityLevel = (
        SensitivityLevel.NON_SENSITIVE
    )

    allowed_for_analysis: bool = True

    allowed_for_grouping: bool = True

    allowed_for_filtering: bool = True

    allowed_for_modeling: bool = True

    @field_validator("local_name")
    @classmethod
    def normalize_local_name(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "local_name cannot be empty"
            )

        return normalized

    @field_validator("semantic_type")
    @classmethod
    def normalize_semantic_type(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.strip().lower()

        return normalized or None

    @model_validator(mode="after")
    def protect_direct_identifiers(
        self,
    ) -> "ColumnMetadata":
        """Direct identifiers cannot be exposed for analysis."""

        if (
            self.sensitivity
            == SensitivityLevel.DIRECT_IDENTIFIER
        ):
            if any(
                (
                    self.allowed_for_analysis,
                    self.allowed_for_grouping,
                    self.allowed_for_filtering,
                    self.allowed_for_modeling,
                )
            ):
                raise ValueError(
                    "Direct identifiers cannot be allowed "
                    "for analysis, grouping, filtering or modeling"
                )

        return self


class DatasetMetadata(BaseModel):
    """Privacy-safe description of one dataset at one site."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)

    display_name: str | None = None

    description: str | None = None

    row_count: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional disclosure-approved row count."
        ),
    )

    row_count_suppressed: bool = False

    columns: list[ColumnMetadata] = Field(
        default_factory=list,
    )

    @field_validator("dataset_id")
    @classmethod
    def normalize_dataset_id(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip().lower()

        if not normalized:
            raise ValueError(
                "dataset_id cannot be empty"
            )

        return normalized

    @model_validator(mode="after")
    def validate_row_count(
        self,
    ) -> "DatasetMetadata":
        if (
            self.row_count_suppressed
            and self.row_count is not None
        ):
            raise ValueError(
                "A suppressed row count cannot be included"
            )

        column_names = [
            column.local_name
            for column in self.columns
        ]

        if len(column_names) != len(set(column_names)):
            raise ValueError(
                "Dataset column names must be unique"
            )

        return self


class SiteCatalogue(BaseModel):
    """Catalogue returned by one local biobank."""

    model_config = ConfigDict(extra="forbid")

    site_id: str = Field(min_length=1)

    datasets: list[DatasetMetadata] = Field(
        default_factory=list,
    )

    schema_version: str = "1.0"

    warnings: list[str] = Field(default_factory=list)

    @field_validator("site_id")
    @classmethod
    def normalize_site_id(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip().lower()

        if not normalized:
            raise ValueError(
                "site_id cannot be empty"
            )

        return normalized

    @model_validator(mode="after")
    def validate_unique_datasets(
        self,
    ) -> "SiteCatalogue":
        dataset_ids = [
            dataset.dataset_id
            for dataset in self.datasets
        ]

        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError(
                "Dataset IDs must be unique within a site"
            )

        return self


# =====================================================================
# Cross-site variable harmonization
# =====================================================================


class SiteColumnMapping(BaseModel):
    """Mapping from a canonical variable to one local column."""

    model_config = ConfigDict(extra="forbid")

    site_id: str = Field(min_length=1)

    dataset_id: str = Field(min_length=1)

    local_name: str = Field(min_length=1)

    mapping_method: MappingMethod

    confidence: float = Field(
        ge=0,
        le=1,
    )

    confirmed: bool = False

    @field_validator(
        "site_id",
        "dataset_id",
    )
    @classmethod
    def normalize_identifier(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip().lower()

        if not normalized:
            raise ValueError(
                "Identifier cannot be empty"
            )

        return normalized


class VariableMapping(BaseModel):
    """Mapping of one canonical variable across biobanks."""

    model_config = ConfigDict(extra="forbid")

    canonical_name: str = Field(min_length=1)

    display_name: str | None = None

    semantic_type: str | None = None

    site_columns: list[SiteColumnMapping] = Field(
        default_factory=list,
    )

    @field_validator(
        "canonical_name",
        "semantic_type",
    )
    @classmethod
    def normalize_canonical_name(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.strip().lower()

        return normalized or None

    @model_validator(mode="after")
    def validate_unique_site_mappings(
        self,
    ) -> "VariableMapping":
        mapping_keys = [
            (
                mapping.site_id,
                mapping.dataset_id,
            )
            for mapping in self.site_columns
        ]

        if len(mapping_keys) != len(set(mapping_keys)):
            raise ValueError(
                "A canonical variable can have only one "
                "mapping per site and dataset"
            )

        return self


class FederatedCatalogue(BaseModel):
    """Federated view of available site metadata."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(default_factory=uuid4)

    participating_sites: list[str] = Field(
        default_factory=list,
    )

    unavailable_sites: list[str] = Field(
        default_factory=list,
    )

    site_catalogues: list[SiteCatalogue] = Field(
        default_factory=list,
    )

    variable_mappings: list[VariableMapping] = Field(
        default_factory=list,
    )

    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "participating_sites",
        "unavailable_sites",
    )
    @classmethod
    def normalize_site_lists(
        cls,
        values: list[str],
    ) -> list[str]:
        normalized = [
            value.strip().lower()
            for value in values
            if value.strip()
        ]

        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "Site lists cannot contain duplicates"
            )

        return normalized

    @model_validator(mode="after")
    def validate_catalogue_sites(
        self,
    ) -> "FederatedCatalogue":
        participating = set(self.participating_sites)
        unavailable = set(self.unavailable_sites)

        if participating & unavailable:
            raise ValueError(
                "A site cannot be both participating "
                "and unavailable"
            )

        catalogue_sites = {
            catalogue.site_id
            for catalogue in self.site_catalogues
        }

        undeclared = catalogue_sites - participating

        if undeclared:
            raise ValueError(
                "Site catalogues were returned by undeclared "
                "participating sites: "
                + ", ".join(sorted(undeclared))
            )

        return self


# =====================================================================
# Filters and analysis parameters
# =====================================================================


class FilterCondition(BaseModel):
    """One condition used to select a local cohort."""

    model_config = ConfigDict(extra="forbid")

    column: str = Field(
        min_length=1,
        description=(
            "Canonical column name. The local agent maps it "
            "to the corresponding local column."
        ),
    )

    operator: FilterOperator

    value: Any | None = None

    @field_validator("column")
    @classmethod
    def normalize_column(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip().lower()

        if not normalized:
            raise ValueError(
                "Filter column cannot be empty"
            )

        return normalized

    @model_validator(mode="after")
    def validate_operator_value(
        self,
    ) -> "FilterCondition":
        if self.operator == FilterOperator.BETWEEN:
            if not isinstance(self.value, (list, tuple)):
                raise ValueError(
                    "'between' requires a list or tuple"
                )

            if len(self.value) != 2:
                raise ValueError(
                    "'between' requires exactly two values"
                )

            lower, upper = self.value

            if lower is None or upper is None:
                raise ValueError(
                    "'between' bounds cannot be null"
                )

            try:
                if lower > upper:
                    raise ValueError(
                        "The lower bound cannot exceed "
                        "the upper bound"
                    )
            except TypeError as error:
                raise ValueError(
                    "'between' bounds must be comparable"
                ) from error

        elif self.operator in {
            FilterOperator.IN,
            FilterOperator.NOT_IN,
        }:
            if not isinstance(self.value, (list, tuple)):
                raise ValueError(
                    f"'{self.operator.value}' requires "
                    "a list or tuple"
                )

            if len(self.value) == 0:
                raise ValueError(
                    f"'{self.operator.value}' cannot use "
                    "an empty collection"
                )

        elif self.operator in {
            FilterOperator.IS_NULL,
            FilterOperator.IS_NOT_NULL,
        }:
            if self.value is not None:
                raise ValueError(
                    f"'{self.operator.value}' does not "
                    "accept a value"
                )

        elif self.value is None:
            raise ValueError(
                f"'{self.operator.value}' requires a value"
            )

        return self


class HistogramSpecification(BaseModel):
    """Configuration for privacy-preserving histogram counts."""

    model_config = ConfigDict(extra="forbid")

    number_of_bins: int = Field(
        default=20,
        ge=2,
        le=200,
    )

    minimum: float | None = None

    maximum: float | None = None

    bin_edges: list[float] | None = None

    @model_validator(mode="after")
    def validate_histogram_configuration(
        self,
    ) -> "HistogramSpecification":
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum >= self.maximum
        ):
            raise ValueError(
                "Histogram minimum must be below maximum"
            )

        if self.bin_edges is not None:
            if len(self.bin_edges) < 3:
                raise ValueError(
                    "Histogram bin_edges must contain at "
                    "least three edges"
                )

            if any(
                left >= right
                for left, right in zip(
                    self.bin_edges,
                    self.bin_edges[1:],
                )
            ):
                raise ValueError(
                    "Histogram bin_edges must be strictly "
                    "increasing"
                )

        return self


# =====================================================================
# Visualization specification
# =====================================================================


class PlotSpecification(BaseModel):
    """Description of a downloadable aggregate visualization."""

    model_config = ConfigDict(extra="forbid")

    plot_type: PlotType

    title: str | None = None

    x: str | None = None

    y: str | None = None

    group_by: str | None = None

    output_format: PlotFormat = PlotFormat.PNG

    width: int = Field(
        default=1000,
        ge=400,
        le=4000,
    )

    height: int = Field(
        default=700,
        ge=300,
        le=4000,
    )

    include_site_breakdown: bool = False

    @field_validator(
        "x",
        "y",
        "group_by",
    )
    @classmethod
    def normalize_optional_column(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.strip().lower()

        return normalized or None

    @model_validator(mode="after")
    def validate_plot_columns(
        self,
    ) -> "PlotSpecification":
        if (
            self.plot_type == PlotType.HISTOGRAM
            and self.x is None
        ):
            raise ValueError(
                "A histogram requires an x column"
            )

        if (
            self.plot_type
            == PlotType.TWO_DIMENSIONAL_HISTOGRAM
            and (
                self.x is None
                or self.y is None
            )
        ):
            raise ValueError(
                "A two-dimensional histogram requires "
                "both x and y columns"
            )

        if (
            self.plot_type
            == PlotType.CORRELATION_HEATMAP
            and (
                self.x is not None
                or self.y is not None
            )
        ):
            raise ValueError(
                "A correlation heatmap uses the plan columns "
                "and should not define x or y"
            )

        return self


# =====================================================================
# Federated model specification
# =====================================================================


class ModelSpecification(BaseModel):
    """Specification for federated model training or evaluation."""

    model_config = ConfigDict(extra="forbid")

    model_type: ModelType

    target: str = Field(min_length=1)

    features: list[str] = Field(min_length=1)

    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
    )

    local_epochs: int = Field(
        default=1,
        ge=1,
        le=100,
    )

    federated_rounds: int = Field(
        default=5,
        ge=1,
        le=1000,
    )

    random_seed: int = Field(default=42, ge=0)

    @field_validator("target")
    @classmethod
    def normalize_target(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip().lower()

        if not normalized:
            raise ValueError(
                "Model target cannot be empty"
            )

        return normalized

    @field_validator("features")
    @classmethod
    def normalize_features(
        cls,
        values: list[str],
    ) -> list[str]:
        normalized = [
            value.strip().lower()
            for value in values
            if value.strip()
        ]

        if not normalized:
            raise ValueError(
                "At least one model feature is required"
            )

        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "Model features must be unique"
            )

        return normalized

    @model_validator(mode="after")
    def validate_target_and_features(
        self,
    ) -> "ModelSpecification":
        if self.target in self.features:
            raise ValueError(
                "The target cannot also be a model feature"
            )

        return self


# =====================================================================
# General analysis plan
# =====================================================================


class AnalysisPlan(BaseModel):
    """Structured request generated by the global planner."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(default_factory=uuid4)

    operation: Operation

    dataset_id: str | None = None

    columns: list[str] = Field(default_factory=list)

    filters: list[FilterCondition] = Field(
        default_factory=list,
    )

    group_by: list[str] = Field(default_factory=list)

    statistics: list[Statistic] = Field(
        default_factory=list,
    )

    histogram: HistogramSpecification | None = None

    plot: PlotSpecification | None = None

    model: ModelSpecification | None = None

    requested_sites: list[str] = Field(
        default_factory=list,
    )

    minimum_cell_size: int = Field(
        default=10,
        ge=1,
    )

    aggregation_method: AggregationMethod | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("dataset_id")
    @classmethod
    def normalize_dataset_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.strip().lower()

        return normalized or None

    @field_validator(
        "columns",
        "group_by",
        "requested_sites",
    )
    @classmethod
    def normalize_string_lists(
        cls,
        values: list[str],
    ) -> list[str]:
        normalized = [
            value.strip().lower()
            for value in values
            if value.strip()
        ]

        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "List entries must be unique"
            )

        return normalized

    @field_validator("statistics")
    @classmethod
    def validate_unique_statistics(
        cls,
        values: list[Statistic],
    ) -> list[Statistic]:
        if len(values) != len(set(values)):
            raise ValueError(
                "Requested statistics must be unique"
            )

        return values

    @model_validator(mode="after")
    def validate_operation_requirements(
        self,
    ) -> "AnalysisPlan":
        """Validate requirements associated with each operation."""

        discovery_operations = {
            Operation.DISCOVER_DATASETS,
            Operation.DISCOVER_SCHEMA,
            Operation.SEARCH_COLUMNS,
            Operation.PROFILE_COLUMNS,
        }

        model_operations = {
            Operation.TRAIN_MODEL,
            Operation.EVALUATE_MODEL,
        }

        column_required_operations = {
            Operation.NUMERIC_SUMMARY,
            Operation.CATEGORICAL_DISTRIBUTION,
            Operation.MISSINGNESS,
            Operation.CORRELATION_STATISTICS,
            Operation.STANDARDIZATION_STATISTICS,
            Operation.HISTOGRAM,
            Operation.TWO_DIMENSIONAL_HISTOGRAM,
        }

        # discover_datasets searches globally, so it should not
        # be restricted to one dataset.
        if (
            self.operation
            == Operation.DISCOVER_DATASETS
            and self.dataset_id is not None
        ):
            raise ValueError(
                "discover_datasets should not specify dataset_id"
            )

        # These operations inspect a particular dataset.
        if (
            self.operation
            in {
                Operation.DISCOVER_SCHEMA,
                Operation.PROFILE_COLUMNS,
            }
            and self.dataset_id is None
        ):
            raise ValueError(
                f"Operation '{self.operation.value}' "
                "requires dataset_id"
            )

        # search_columns may search one dataset or every dataset.
        # Therefore dataset_id is optional for SEARCH_COLUMNS.

        if (
            self.operation in discovery_operations
            and self.model is not None
        ):
            raise ValueError(
                "Discovery operations cannot include a model"
            )

        if (
            self.operation in discovery_operations
            and self.plot is not None
        ):
            raise ValueError(
                "Discovery operations cannot include a plot"
            )

        if (
            self.operation in discovery_operations
            and self.histogram is not None
        ):
            raise ValueError(
                "Discovery operations cannot include "
                "a histogram specification"
            )

        if (
            self.operation in column_required_operations
            and not self.columns
        ):
            raise ValueError(
                f"Operation '{self.operation.value}' "
                "requires at least one column"
            )

        if (
            self.operation
            == Operation.CATEGORICAL_DISTRIBUTION
            and len(self.columns) != 1
        ):
            raise ValueError(
                "categorical_distribution requires "
                "exactly one column"
            )

        if self.operation == Operation.GROUPED_SUMMARY:
            if not self.columns:
                raise ValueError(
                    "grouped_summary requires at least "
                    "one analysis column"
                )

            if not self.group_by:
                raise ValueError(
                    "grouped_summary requires at least "
                    "one group_by column"
                )

        if (
            self.operation
            == Operation.CORRELATION_STATISTICS
            and len(self.columns) < 2
        ):
            raise ValueError(
                "correlation_statistics requires at least "
                "two columns"
            )

        if self.operation == Operation.HISTOGRAM:
            if len(self.columns) != 1:
                raise ValueError(
                    "histogram requires exactly one column"
                )

            if self.histogram is None:
                raise ValueError(
                    "histogram requires a histogram "
                    "specification"
                )

        if (
            self.operation
            == Operation.TWO_DIMENSIONAL_HISTOGRAM
        ):
            if len(self.columns) != 2:
                raise ValueError(
                    "two_dimensional_histogram requires "
                    "exactly two columns"
                )

            if self.histogram is None:
                raise ValueError(
                    "two_dimensional_histogram requires "
                    "a histogram specification"
                )

        if (
            self.operation in model_operations
            and self.model is None
        ):
            raise ValueError(
                f"Operation '{self.operation.value}' "
                "requires a model specification"
            )

        if (
            self.operation not in model_operations
            and self.model is not None
        ):
            raise ValueError(
                "A model specification is only allowed for "
                "train_model or evaluate_model"
            )

        return self


# =====================================================================
# Privacy metadata and site responses
# =====================================================================


class PrivacyMetadata(BaseModel):
    """Disclosure-control status returned by a local site."""

    model_config = ConfigDict(extra="forbid")

    minimum_cell_size: int = Field(ge=1)

    suppressed: bool = False

    reason: str | None = None

    disclosure_checks_passed: bool = True

    @model_validator(mode="after")
    def validate_privacy_state(
        self,
    ) -> "PrivacyMetadata":
        if self.suppressed and not self.reason:
            raise ValueError(
                "A suppression reason is required"
            )

        if (
            self.suppressed
            and self.disclosure_checks_passed
        ):
            raise ValueError(
                "A suppressed response cannot pass "
                "all disclosure checks"
            )

        if (
            not self.suppressed
            and self.reason is not None
        ):
            raise ValueError(
                "A suppression reason is only valid when "
                "suppressed=True"
            )

        return self


class SiteResponse(BaseModel):
    """Privacy-approved response from one local biobank."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID

    site_id: str = Field(min_length=1)

    status: ResponseStatus

    result: dict[str, Any] = Field(
        default_factory=dict,
    )

    warnings: list[str] = Field(default_factory=list)

    privacy: PrivacyMetadata

    execution_time_seconds: float | None = Field(
        default=None,
        ge=0,
    )

    schema_version: str = "1.0"

    @field_validator("site_id")
    @classmethod
    def normalize_site_id(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip().lower()

        if not normalized:
            raise ValueError(
                "site_id cannot be empty"
            )

        return normalized

    @model_validator(mode="after")
    def validate_status_and_result(
        self,
    ) -> "SiteResponse":
        if (
            self.status == ResponseStatus.SUCCESS
            and self.privacy.suppressed
        ):
            raise ValueError(
                "A successful response cannot be suppressed"
            )

        if self.status == ResponseStatus.SUPPRESSED:
            if not self.privacy.suppressed:
                raise ValueError(
                    "Suppressed status requires "
                    "privacy.suppressed=True"
                )

            if self.result:
                raise ValueError(
                    "A suppressed response cannot "
                    "contain a result"
                )

        if self.status in {
            ResponseStatus.ERROR,
            ResponseStatus.UNSUPPORTED,
            ResponseStatus.UNAVAILABLE,
        }:
            if self.result:
                raise ValueError(
                    f"Status '{self.status.value}' cannot "
                    "contain results"
                )

        return self


class ArtifactReference(BaseModel):
    """Reference to a generated downloadable artifact."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=1)

    filename: str = Field(min_length=1)

    media_type: str = Field(min_length=1)

    download_path: str = Field(
        min_length=1,
        description=(
            "Controlled API path, not an arbitrary filesystem path."
        ),
    )


class FederatedResult(BaseModel):
    """Aggregated result returned by the federated layer."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID

    operation: Operation

    aggregation_method: AggregationMethod | None = None

    participating_sites: list[str] = Field(
        default_factory=list,
    )

    unavailable_sites: list[str] = Field(
        default_factory=list,
    )

    suppressed_sites: list[str] = Field(
        default_factory=list,
    )

    result: dict[str, Any] = Field(
        default_factory=dict,
    )

    site_results: list[SiteResponse] = Field(
        default_factory=list,
    )

    artifacts: list[ArtifactReference] = Field(
        default_factory=list,
    )

    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "participating_sites",
        "unavailable_sites",
        "suppressed_sites",
    )
    @classmethod
    def normalize_site_lists(
        cls,
        values: list[str],
    ) -> list[str]:
        normalized = [
            value.strip().lower()
            for value in values
            if value.strip()
        ]

        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "Site lists cannot contain duplicates"
            )

        return normalized

    @model_validator(mode="after")
    def validate_sites_and_responses(
        self,
    ) -> "FederatedResult":
        participating = set(self.participating_sites)
        unavailable = set(self.unavailable_sites)
        suppressed = set(self.suppressed_sites)

        if participating & unavailable:
            raise ValueError(
                "A site cannot be both participating "
                "and unavailable"
            )

        if participating & suppressed:
            raise ValueError(
                "A site cannot be both participating "
                "and suppressed"
            )

        if unavailable & suppressed:
            raise ValueError(
                "A site cannot be both unavailable "
                "and suppressed"
            )

        declared_sites = (
            participating
            | unavailable
            | suppressed
        )

        for response in self.site_results:
            if response.request_id != self.request_id:
                raise ValueError(
                    "Every SiteResponse must use the "
                    "FederatedResult request_id"
                )

            if response.site_id not in declared_sites:
                raise ValueError(
                    f"Site '{response.site_id}' is not "
                    "declared in a site list"
                )

            if (
                response.status == ResponseStatus.SUCCESS
                and response.site_id not in participating
            ):
                raise ValueError(
                    f"Successful site '{response.site_id}' "
                    "must appear in participating_sites"
                )

            if (
                response.status
                == ResponseStatus.SUPPRESSED
                and response.site_id not in suppressed
            ):
                raise ValueError(
                    f"Suppressed site '{response.site_id}' "
                    "must appear in suppressed_sites"
                )

            if (
                response.status
                in {
                    ResponseStatus.ERROR,
                    ResponseStatus.UNSUPPORTED,
                    ResponseStatus.UNAVAILABLE,
                }
                and response.site_id not in unavailable
            ):
                raise ValueError(
                    f"Nonparticipating site "
                    f"'{response.site_id}' must appear in "
                    "unavailable_sites"
                )

        return self


# =====================================================================
# Chat interaction
# =====================================================================


class SuggestedAction(BaseModel):
    """One follow-up action offered to the user."""

    model_config = ConfigDict(extra="forbid")

    action: Operation

    description: str = Field(min_length=1)

    parameters: dict[str, Any] = Field(
        default_factory=dict,
    )


class ChatRequest(BaseModel):
    """Natural-language request received by the global API."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=3,
        max_length=4000,
    )

    @field_validator("query")
    @classmethod
    def normalize_query(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "Query cannot be empty"
            )

        return normalized


class ChatResponse(BaseModel):
    """Complete response returned by the global agent."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID

    plan: AnalysisPlan | None = None

    catalogue: FederatedCatalogue | None = None

    federated_result: FederatedResult | None = None

    answer: str | None = None

    clarification_required: bool = False

    clarification_question: str | None = None

    suggested_actions: list[SuggestedAction] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_response(
        self,
    ) -> "ChatResponse":
        if (
            self.clarification_required
            and not self.clarification_question
        ):
            raise ValueError(
                "A clarification question is required when "
                "clarification_required=True"
            )

        if (
            not self.clarification_required
            and self.clarification_question is not None
        ):
            raise ValueError(
                "clarification_question must be null when "
                "clarification_required=False"
            )

        if (
            self.plan is not None
            and self.plan.request_id != self.request_id
        ):
            raise ValueError(
                "ChatResponse request_id must match "
                "AnalysisPlan request_id"
            )

        if (
            self.catalogue is not None
            and self.catalogue.request_id
            != self.request_id
        ):
            raise ValueError(
                "ChatResponse request_id must match "
                "FederatedCatalogue request_id"
            )

        if (
            self.federated_result is not None
            and self.federated_result.request_id
            != self.request_id
        ):
            raise ValueError(
                "ChatResponse request_id must match "
                "FederatedResult request_id"
            )

        return self