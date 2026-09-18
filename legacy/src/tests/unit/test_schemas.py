"""Tests for the shared federated-analysis schemas."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from shared.schemas import (
    AggregationMethod,
    AnalysisPlan,
    ArtifactReference,
    ChatRequest,
    ChatResponse,
    ColumnMetadata,
    DataType,
    DatasetMetadata,
    FederatedCatalogue,
    FederatedResult,
    FilterCondition,
    FilterOperator,
    HistogramSpecification,
    MappingMethod,
    ModelSpecification,
    ModelType,
    Operation,
    PlotFormat,
    PlotSpecification,
    PlotType,
    PrivacyMetadata,
    ResponseStatus,
    SensitivityLevel,
    SiteCatalogue,
    SiteColumnMapping,
    SiteResponse,
    Statistic,
    SuggestedAction,
    VariableMapping,
)


# =====================================================================
# ColumnMetadata
# =====================================================================


def test_valid_numeric_column_metadata() -> None:
    column = ColumnMetadata(
        local_name="age_at_visit",
        display_name="Age at visit",
        data_type=DataType.NUMERIC,
        semantic_type=" Age ",
        description="Age at the clinical visit",
        unit="years",
        missing_fraction=0.02,
        unique_count=68,
        sensitivity=(
            SensitivityLevel.QUASI_IDENTIFIER
        ),
        allowed_for_analysis=True,
        allowed_for_grouping=False,
        allowed_for_filtering=True,
        allowed_for_modeling=True,
    )

    assert column.local_name == "age_at_visit"
    assert column.semantic_type == "age"
    assert column.missing_fraction == 0.02


def test_column_rejects_invalid_missing_fraction() -> None:
    with pytest.raises(ValidationError):
        ColumnMetadata(
            local_name="age",
            data_type=DataType.NUMERIC,
            missing_fraction=1.5,
        )


def test_direct_identifier_cannot_be_analysed() -> None:
    with pytest.raises(
        ValidationError,
        match="Direct identifiers cannot be allowed",
    ):
        ColumnMetadata(
            local_name="participant_id",
            data_type=DataType.STRING,
            sensitivity=(
                SensitivityLevel.DIRECT_IDENTIFIER
            ),
            allowed_for_analysis=True,
            allowed_for_grouping=False,
            allowed_for_filtering=False,
            allowed_for_modeling=False,
        )


def test_valid_protected_direct_identifier() -> None:
    column = ColumnMetadata(
        local_name="participant_id",
        data_type=DataType.STRING,
        sensitivity=(
            SensitivityLevel.DIRECT_IDENTIFIER
        ),
        allowed_for_analysis=False,
        allowed_for_grouping=False,
        allowed_for_filtering=False,
        allowed_for_modeling=False,
    )

    assert (
        column.sensitivity
        == SensitivityLevel.DIRECT_IDENTIFIER
    )

    assert column.allowed_for_analysis is False


# =====================================================================
# DatasetMetadata and SiteCatalogue
# =====================================================================


def make_age_column() -> ColumnMetadata:
    """Create reusable age metadata."""

    return ColumnMetadata(
        local_name="age_at_visit",
        display_name="Age at visit",
        data_type=DataType.NUMERIC,
        semantic_type="age",
        unit="years",
        missing_fraction=0.01,
        sensitivity=(
            SensitivityLevel.QUASI_IDENTIFIER
        ),
        allowed_for_analysis=True,
        allowed_for_grouping=False,
        allowed_for_filtering=True,
        allowed_for_modeling=True,
    )


def make_diabetes_column() -> ColumnMetadata:
    """Create reusable diabetes metadata."""

    return ColumnMetadata(
        local_name="t2d_status",
        display_name="Type 2 diabetes status",
        data_type=DataType.BINARY,
        semantic_type="diabetes",
        missing_fraction=0.03,
        unique_count=2,
        sensitivity=SensitivityLevel.SENSITIVE,
        allowed_for_analysis=True,
        allowed_for_grouping=True,
        allowed_for_filtering=True,
        allowed_for_modeling=True,
    )


def make_dataset(
    dataset_id: str = "health_records",
) -> DatasetMetadata:
    """Create a reusable dataset description."""

    return DatasetMetadata(
        dataset_id=dataset_id,
        display_name="Health records",
        row_count=1000,
        columns=[
            make_age_column(),
            make_diabetes_column(),
        ],
    )


def test_valid_dataset_metadata() -> None:
    dataset = make_dataset()

    assert dataset.dataset_id == "health_records"
    assert dataset.row_count == 1000
    assert len(dataset.columns) == 2


def test_dataset_id_is_normalized() -> None:
    dataset = DatasetMetadata(
        dataset_id=" Health_Records ",
    )

    assert dataset.dataset_id == "health_records"


def test_suppressed_row_count_cannot_be_returned() -> None:
    with pytest.raises(
        ValidationError,
        match="suppressed row count cannot be included",
    ):
        DatasetMetadata(
            dataset_id="small_dataset",
            row_count=4,
            row_count_suppressed=True,
        )


def test_duplicate_local_column_names_are_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="column names must be unique",
    ):
        DatasetMetadata(
            dataset_id="health_records",
            columns=[
                ColumnMetadata(
                    local_name="age",
                    data_type=DataType.NUMERIC,
                ),
                ColumnMetadata(
                    local_name="age",
                    data_type=DataType.NUMERIC,
                ),
            ],
        )


def test_valid_site_catalogue() -> None:
    catalogue = SiteCatalogue(
        site_id=" Biobank-A ",
        datasets=[make_dataset()],
    )

    assert catalogue.site_id == "biobank-a"
    assert len(catalogue.datasets) == 1


def test_duplicate_dataset_ids_are_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="Dataset IDs must be unique",
    ):
        SiteCatalogue(
            site_id="biobank-a",
            datasets=[
                make_dataset("health_records"),
                make_dataset("health_records"),
            ],
        )


# =====================================================================
# Variable harmonization
# =====================================================================


def make_age_mapping(
    site_id: str,
    local_name: str,
) -> SiteColumnMapping:
    """Create one reusable age mapping."""

    return SiteColumnMapping(
        site_id=site_id,
        dataset_id="health_records",
        local_name=local_name,
        mapping_method=(
            MappingMethod.DATA_DICTIONARY
        ),
        confidence=1.0,
        confirmed=True,
    )


def test_valid_variable_mapping() -> None:
    mapping = VariableMapping(
        canonical_name=" Age ",
        display_name="Age at visit",
        semantic_type=" Age ",
        site_columns=[
            make_age_mapping(
                "biobank-a",
                "age_at_visit",
            ),
            make_age_mapping(
                "biobank-b",
                "patient_age",
            ),
        ],
    )

    assert mapping.canonical_name == "age"
    assert mapping.semantic_type == "age"
    assert len(mapping.site_columns) == 2


def test_duplicate_site_dataset_mapping_is_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="only one mapping per site and dataset",
    ):
        VariableMapping(
            canonical_name="age",
            site_columns=[
                make_age_mapping(
                    "biobank-a",
                    "age_at_visit",
                ),
                make_age_mapping(
                    "biobank-a",
                    "patient_age",
                ),
            ],
        )


# =====================================================================
# FederatedCatalogue
# =====================================================================


def test_valid_federated_catalogue() -> None:
    request_id = uuid4()

    site_catalogue = SiteCatalogue(
        site_id="biobank-a",
        datasets=[make_dataset()],
    )

    catalogue = FederatedCatalogue(
        request_id=request_id,
        participating_sites=["biobank-a"],
        unavailable_sites=["biobank-b"],
        site_catalogues=[site_catalogue],
        variable_mappings=[
            VariableMapping(
                canonical_name="age",
                site_columns=[
                    make_age_mapping(
                        "biobank-a",
                        "age_at_visit",
                    )
                ],
            )
        ],
    )

    assert catalogue.request_id == request_id
    assert catalogue.participating_sites == [
        "biobank-a"
    ]


def test_catalogue_site_must_be_participating() -> None:
    with pytest.raises(
        ValidationError,
        match="undeclared participating sites",
    ):
        FederatedCatalogue(
            participating_sites=["biobank-a"],
            site_catalogues=[
                SiteCatalogue(
                    site_id="biobank-b",
                    datasets=[make_dataset()],
                )
            ],
        )


def test_catalogue_site_cannot_be_available_and_unavailable() -> None:
    with pytest.raises(
        ValidationError,
        match="both participating and unavailable",
    ):
        FederatedCatalogue(
            participating_sites=["biobank-a"],
            unavailable_sites=["biobank-a"],
        )


# =====================================================================
# FilterCondition
# =====================================================================


def test_valid_between_filter() -> None:
    condition = FilterCondition(
        column=" Age ",
        operator=FilterOperator.BETWEEN,
        value=[60, 80],
    )

    assert condition.column == "age"
    assert condition.value == [60, 80]


def test_between_requires_two_values() -> None:
    with pytest.raises(
        ValidationError,
        match="exactly two values",
    ):
        FilterCondition(
            column="age",
            operator=FilterOperator.BETWEEN,
            value=[60],
        )


def test_between_rejects_reversed_bounds() -> None:
    with pytest.raises(
        ValidationError,
        match="lower bound cannot exceed",
    ):
        FilterCondition(
            column="age",
            operator=FilterOperator.BETWEEN,
            value=[80, 60],
        )


def test_in_operator_requires_collection() -> None:
    with pytest.raises(
        ValidationError,
        match="requires a list or tuple",
    ):
        FilterCondition(
            column="diagnosis",
            operator=FilterOperator.IN,
            value="diabetes",
        )


def test_null_operator_rejects_value() -> None:
    with pytest.raises(
        ValidationError,
        match="does not accept a value",
    ):
        FilterCondition(
            column="bmi",
            operator=FilterOperator.IS_NULL,
            value=True,
        )


# =====================================================================
# HistogramSpecification
# =====================================================================


def test_valid_histogram_specification() -> None:
    specification = HistogramSpecification(
        number_of_bins=20,
        minimum=20,
        maximum=100,
    )

    assert specification.number_of_bins == 20


def test_histogram_rejects_invalid_range() -> None:
    with pytest.raises(
        ValidationError,
        match="minimum must be below maximum",
    ):
        HistogramSpecification(
            minimum=100,
            maximum=20,
        )


def test_histogram_rejects_unsorted_bin_edges() -> None:
    with pytest.raises(
        ValidationError,
        match="strictly increasing",
    ):
        HistogramSpecification(
            bin_edges=[
                0,
                10,
                5,
                20,
            ],
        )


# =====================================================================
# PlotSpecification
# =====================================================================


def test_valid_histogram_plot() -> None:
    plot = PlotSpecification(
        plot_type=PlotType.HISTOGRAM,
        title="Age distribution",
        x=" Age ",
        output_format=PlotFormat.PNG,
    )

    assert plot.x == "age"
    assert plot.output_format == PlotFormat.PNG


def test_histogram_plot_requires_x_column() -> None:
    with pytest.raises(
        ValidationError,
        match="requires an x column",
    ):
        PlotSpecification(
            plot_type=PlotType.HISTOGRAM,
        )


def test_two_dimensional_histogram_requires_x_and_y() -> None:
    with pytest.raises(
        ValidationError,
        match="requires both x and y",
    ):
        PlotSpecification(
            plot_type=(
                PlotType.TWO_DIMENSIONAL_HISTOGRAM
            ),
            x="age",
        )


def test_correlation_heatmap_rejects_x_and_y() -> None:
    with pytest.raises(
        ValidationError,
        match="should not define x or y",
    ):
        PlotSpecification(
            plot_type=PlotType.CORRELATION_HEATMAP,
            x="age",
        )


# =====================================================================
# ModelSpecification
# =====================================================================


def make_logistic_model() -> ModelSpecification:
    """Create a reusable model specification."""

    return ModelSpecification(
        model_type=ModelType.LOGISTIC_REGRESSION,
        target="diabetes",
        features=[
            "age",
            "sex",
            "bmi",
        ],
        local_epochs=2,
        federated_rounds=5,
    )


def test_valid_model_specification() -> None:
    model = make_logistic_model()

    assert (
        model.model_type
        == ModelType.LOGISTIC_REGRESSION
    )

    assert model.target == "diabetes"
    assert model.features == [
        "age",
        "sex",
        "bmi",
    ]


def test_model_rejects_duplicate_features() -> None:
    with pytest.raises(
        ValidationError,
        match="features must be unique",
    ):
        ModelSpecification(
            model_type=ModelType.LOGISTIC_REGRESSION,
            target="diabetes",
            features=[
                "age",
                "bmi",
                "age",
            ],
        )


def test_model_rejects_target_as_feature() -> None:
    with pytest.raises(
        ValidationError,
        match="cannot also be a model feature",
    ):
        ModelSpecification(
            model_type=ModelType.LOGISTIC_REGRESSION,
            target="diabetes",
            features=[
                "age",
                "diabetes",
            ],
        )


# =====================================================================
# Discovery AnalysisPlan
# =====================================================================


def test_valid_discover_datasets_plan() -> None:
    plan = AnalysisPlan(
        operation=Operation.DISCOVER_DATASETS,
    )

    assert plan.dataset_id is None
    assert plan.columns == []


def test_discover_datasets_rejects_dataset_id() -> None:
    with pytest.raises(
        ValidationError,
        match="should not specify dataset_id",
    ):
        AnalysisPlan(
            operation=Operation.DISCOVER_DATASETS,
            dataset_id="health_records",
        )


def test_discover_schema_requires_dataset_id() -> None:
    with pytest.raises(
        ValidationError,
        match="requires dataset_id",
    ):
        AnalysisPlan(
            operation=Operation.DISCOVER_SCHEMA,
        )


def test_valid_discover_schema_plan() -> None:
    plan = AnalysisPlan(
        operation=Operation.DISCOVER_SCHEMA,
        dataset_id=" Health_Records ",
    )

    assert plan.dataset_id == "health_records"


# =====================================================================
# Exploratory AnalysisPlan
# =====================================================================


def test_valid_numeric_summary_plan() -> None:
    plan = AnalysisPlan(
        operation=Operation.NUMERIC_SUMMARY,
        dataset_id="health_records",
        columns=[
            "age",
            "bmi",
        ],
        statistics=[
            Statistic.COUNT,
            Statistic.MEAN,
            Statistic.STANDARD_DEVIATION,
        ],
        aggregation_method=(
            AggregationMethod.SUFFICIENT_STATISTICS
        ),
    )

    assert plan.columns == ["age", "bmi"]
    assert Statistic.MEAN in plan.statistics


def test_numeric_summary_requires_column() -> None:
    with pytest.raises(
        ValidationError,
        match="requires at least one column",
    ):
        AnalysisPlan(
            operation=Operation.NUMERIC_SUMMARY,
        )


def test_categorical_distribution_requires_one_column() -> None:
    with pytest.raises(
        ValidationError,
        match="requires exactly one column",
    ):
        AnalysisPlan(
            operation=(
                Operation.CATEGORICAL_DISTRIBUTION
            ),
            columns=[
                "diabetes",
                "sex",
            ],
        )


def test_valid_grouped_summary_plan() -> None:
    plan = AnalysisPlan(
        operation=Operation.GROUPED_SUMMARY,
        dataset_id="health_records",
        columns=["diabetes"],
        group_by=["sex"],
        statistics=[
            Statistic.COUNT,
            Statistic.PREVALENCE,
        ],
        aggregation_method=AggregationMethod.SUM,
    )

    assert plan.columns == ["diabetes"]
    assert plan.group_by == ["sex"]


def test_grouped_summary_requires_group_by() -> None:
    with pytest.raises(
        ValidationError,
        match="requires at least one group_by",
    ):
        AnalysisPlan(
            operation=Operation.GROUPED_SUMMARY,
            columns=["diabetes"],
        )


def test_correlation_requires_two_columns() -> None:
    with pytest.raises(
        ValidationError,
        match="requires at least two columns",
    ):
        AnalysisPlan(
            operation=(
                Operation.CORRELATION_STATISTICS
            ),
            columns=["age"],
        )


def test_valid_histogram_plan() -> None:
    plan = AnalysisPlan(
        operation=Operation.HISTOGRAM,
        dataset_id="health_records",
        columns=["age"],
        histogram=HistogramSpecification(
            number_of_bins=20,
        ),
        plot=PlotSpecification(
            plot_type=PlotType.HISTOGRAM,
            title="Age distribution",
            x="age",
            output_format=PlotFormat.PNG,
        ),
        aggregation_method=AggregationMethod.SUM,
    )

    assert plan.histogram is not None
    assert plan.plot is not None


def test_histogram_plan_requires_one_column() -> None:
    with pytest.raises(
        ValidationError,
        match="requires exactly one column",
    ):
        AnalysisPlan(
            operation=Operation.HISTOGRAM,
            columns=[
                "age",
                "bmi",
            ],
            histogram=HistogramSpecification(),
        )


def test_histogram_plan_requires_histogram_specification() -> None:
    with pytest.raises(
        ValidationError,
        match="requires a histogram specification",
    ):
        AnalysisPlan(
            operation=Operation.HISTOGRAM,
            columns=["age"],
        )


def test_valid_two_dimensional_histogram_plan() -> None:
    plan = AnalysisPlan(
        operation=(
            Operation.TWO_DIMENSIONAL_HISTOGRAM
        ),
        columns=[
            "age",
            "bmi",
        ],
        histogram=HistogramSpecification(
            number_of_bins=20,
        ),
        plot=PlotSpecification(
            plot_type=(
                PlotType.TWO_DIMENSIONAL_HISTOGRAM
            ),
            x="age",
            y="bmi",
        ),
    )

    assert len(plan.columns) == 2


# =====================================================================
# Model AnalysisPlan
# =====================================================================


def test_valid_model_training_plan() -> None:
    plan = AnalysisPlan(
        operation=Operation.TRAIN_MODEL,
        dataset_id="health_records",
        model=make_logistic_model(),
        aggregation_method=AggregationMethod.FEDAVG,
    )

    assert plan.model is not None
    assert plan.model.target == "diabetes"


def test_train_model_requires_model_specification() -> None:
    with pytest.raises(
        ValidationError,
        match="requires a model specification",
    ):
        AnalysisPlan(
            operation=Operation.TRAIN_MODEL,
        )


def test_non_model_operation_rejects_model() -> None:
    with pytest.raises(
        ValidationError,
        match="only allowed for train_model",
    ):
        AnalysisPlan(
            operation=Operation.COHORT_COUNT,
            model=make_logistic_model(),
        )


def test_plan_rejects_duplicate_columns() -> None:
    with pytest.raises(
        ValidationError,
        match="List entries must be unique",
    ):
        AnalysisPlan(
            operation=Operation.NUMERIC_SUMMARY,
            columns=[
                "age",
                "age",
            ],
        )


def test_plan_rejects_duplicate_statistics() -> None:
    with pytest.raises(
        ValidationError,
        match="statistics must be unique",
    ):
        AnalysisPlan(
            operation=Operation.NUMERIC_SUMMARY,
            columns=["age"],
            statistics=[
                Statistic.MEAN,
                Statistic.MEAN,
            ],
        )


# =====================================================================
# PrivacyMetadata and SiteResponse
# =====================================================================


def make_success_privacy() -> PrivacyMetadata:
    """Create reusable successful privacy metadata."""

    return PrivacyMetadata(
        minimum_cell_size=10,
        suppressed=False,
        disclosure_checks_passed=True,
    )


def test_valid_suppressed_privacy_metadata() -> None:
    privacy = PrivacyMetadata(
        minimum_cell_size=10,
        suppressed=True,
        reason="Cohort below minimum cell size",
        disclosure_checks_passed=False,
    )

    assert privacy.suppressed is True


def test_suppressed_privacy_requires_reason() -> None:
    with pytest.raises(
        ValidationError,
        match="suppression reason is required",
    ):
        PrivacyMetadata(
            minimum_cell_size=10,
            suppressed=True,
            disclosure_checks_passed=False,
        )


def test_valid_site_response() -> None:
    request_id = uuid4()

    response = SiteResponse(
        request_id=request_id,
        site_id=" Biobank-A ",
        status=ResponseStatus.SUCCESS,
        result={
            "count": 120,
        },
        privacy=make_success_privacy(),
    )

    assert response.site_id == "biobank-a"
    assert response.result["count"] == 120


def test_suppressed_response_cannot_reveal_result() -> None:
    with pytest.raises(
        ValidationError,
        match="cannot contain a result",
    ):
        SiteResponse(
            request_id=uuid4(),
            site_id="biobank-a",
            status=ResponseStatus.SUPPRESSED,
            result={
                "count": 4,
            },
            privacy=PrivacyMetadata(
                minimum_cell_size=10,
                suppressed=True,
                reason="Small cohort",
                disclosure_checks_passed=False,
            ),
        )


def test_error_response_cannot_contain_results() -> None:
    with pytest.raises(
        ValidationError,
        match="cannot contain results",
    ):
        SiteResponse(
            request_id=uuid4(),
            site_id="biobank-a",
            status=ResponseStatus.ERROR,
            result={
                "count": 100,
            },
            privacy=make_success_privacy(),
        )


# =====================================================================
# ArtifactReference and FederatedResult
# =====================================================================


def test_valid_artifact_reference() -> None:
    artifact = ArtifactReference(
        artifact_id="plot-001",
        filename="age_histogram.png",
        media_type="image/png",
        download_path=(
            "/results/request-001/plot/plot-001"
        ),
    )

    assert artifact.filename == "age_histogram.png"


def test_valid_federated_result_with_artifact() -> None:
    request_id = uuid4()

    site_response = SiteResponse(
        request_id=request_id,
        site_id="biobank-a",
        status=ResponseStatus.SUCCESS,
        result={
            "bin_edges": [0, 20, 40, 60, 80, 100],
            "counts": [20, 100, 230, 180, 50],
        },
        privacy=make_success_privacy(),
    )

    result = FederatedResult(
        request_id=request_id,
        operation=Operation.HISTOGRAM,
        aggregation_method=AggregationMethod.SUM,
        participating_sites=["biobank-a"],
        result={
            "bin_edges": [0, 20, 40, 60, 80, 100],
            "counts": [20, 100, 230, 180, 50],
        },
        site_results=[site_response],
        artifacts=[
            ArtifactReference(
                artifact_id="plot-001",
                filename="age_histogram.png",
                media_type="image/png",
                download_path=(
                    f"/results/{request_id}/"
                    "artifacts/plot-001"
                ),
            )
        ],
    )

    assert len(result.artifacts) == 1
    assert result.participating_sites == [
        "biobank-a"
    ]


def test_federated_result_rejects_mismatched_request_id() -> None:
    site_response = SiteResponse(
        request_id=uuid4(),
        site_id="biobank-a",
        status=ResponseStatus.SUCCESS,
        result={
            "count": 100,
        },
        privacy=make_success_privacy(),
    )

    with pytest.raises(
        ValidationError,
        match="must use the FederatedResult request_id",
    ):
        FederatedResult(
            request_id=uuid4(),
            operation=Operation.COHORT_COUNT,
            aggregation_method=AggregationMethod.SUM,
            participating_sites=["biobank-a"],
            site_results=[site_response],
        )


def test_federated_result_rejects_undeclared_site() -> None:
    request_id = uuid4()

    site_response = SiteResponse(
        request_id=request_id,
        site_id="biobank-b",
        status=ResponseStatus.SUCCESS,
        result={
            "count": 100,
        },
        privacy=make_success_privacy(),
    )

    with pytest.raises(
        ValidationError,
        match="is not declared in a site list",
    ):
        FederatedResult(
            request_id=request_id,
            operation=Operation.COHORT_COUNT,
            aggregation_method=AggregationMethod.SUM,
            participating_sites=["biobank-a"],
            site_results=[site_response],
        )


# =====================================================================
# Chat schemas
# =====================================================================


def test_chat_request_strips_whitespace() -> None:
    request = ChatRequest(
        query="  What datasets are available?  "
    )

    assert request.query == (
        "What datasets are available?"
    )


def test_valid_discovery_chat_response() -> None:
    request_id = uuid4()

    catalogue = FederatedCatalogue(
        request_id=request_id,
        participating_sites=["biobank-a"],
        site_catalogues=[
            SiteCatalogue(
                site_id="biobank-a",
                datasets=[make_dataset()],
            )
        ],
    )

    response = ChatResponse(
        request_id=request_id,
        catalogue=catalogue,
        answer=(
            "One dataset is available from biobank-a."
        ),
        clarification_required=True,
        clarification_question=(
            "Which columns would you like to inspect?"
        ),
        suggested_actions=[
            SuggestedAction(
                action=Operation.PROFILE_COLUMNS,
                description=(
                    "Inspect column types and missingness."
                ),
                parameters={
                    "dataset_id": "health_records",
                },
            )
        ],
    )

    assert response.catalogue is not None
    assert response.clarification_required is True
    assert len(response.suggested_actions) == 1


def test_chat_response_requires_clarification_question() -> None:
    with pytest.raises(
        ValidationError,
        match="clarification question is required",
    ):
        ChatResponse(
            request_id=uuid4(),
            clarification_required=True,
        )


def test_chat_response_rejects_mismatched_catalogue_id() -> None:
    catalogue = FederatedCatalogue(
        request_id=uuid4(),
    )

    with pytest.raises(
        ValidationError,
        match="must match FederatedCatalogue request_id",
    ):
        ChatResponse(
            request_id=uuid4(),
            catalogue=catalogue,
            answer="Datasets discovered.",
        )


def test_schema_objects_are_json_serializable() -> None:
    plan = AnalysisPlan(
        operation=Operation.HISTOGRAM,
        dataset_id="health_records",
        columns=["age"],
        histogram=HistogramSpecification(
            number_of_bins=20,
        ),
        plot=PlotSpecification(
            plot_type=PlotType.HISTOGRAM,
            x="age",
            output_format=PlotFormat.PNG,
        ),
        aggregation_method=AggregationMethod.SUM,
    )

    json_output = plan.model_dump_json()

    assert isinstance(json_output, str)
    assert '"operation":"histogram"' in json_output
    assert '"columns":["age"]' in json_output