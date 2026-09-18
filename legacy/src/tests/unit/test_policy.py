"""Unit tests for global-agent policy validation."""

from __future__ import annotations

import pytest

from global_agent.exceptions import PolicyViolation
from global_agent.policy import (
    GlobalPolicy,
    PolicyConfig,
)
from shared.schemas import (
    AggregationMethod,
    AnalysisPlan,
    FilterCondition,
    FilterOperator,
    ModelSpecification,
    ModelType,
    Operation,
)


# =====================================================================
# Helper functions
# =====================================================================


def make_logistic_model(
    *,
    local_epochs: int = 2,
    federated_rounds: int = 5,
    hyperparameters: dict | None = None,
) -> ModelSpecification:
    """Create a valid logistic-regression model specification."""

    return ModelSpecification(
        model_type=ModelType.LOGISTIC_REGRESSION,
        target="diabetes",
        features=[
            "age",
            "sex",
            "bmi",
            "systolic_bp",
        ],
        local_epochs=local_epochs,
        federated_rounds=federated_rounds,
        hyperparameters=hyperparameters or {},
    )


def make_random_forest_model(
    *,
    hyperparameters: dict | None = None,
) -> ModelSpecification:
    """Create a valid Random Forest model specification."""

    return ModelSpecification(
        model_type=ModelType.RANDOM_FOREST,
        target="diabetes",
        features=[
            "age",
            "sex",
            "bmi",
        ],
        hyperparameters=hyperparameters or {},
    )


# =====================================================================
# Valid-plan tests
# =====================================================================


def test_valid_cohort_count_plan_passes() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.COHORT_COUNT,
        variables=["bmi"],
        filters=[
            FilterCondition(
                variable="age",
                operator=FilterOperator.BETWEEN,
                value=[60, 80],
            )
        ],
        minimum_cell_size=10,
        aggregation_method=AggregationMethod.SUM,
    )

    result = policy.validate(plan)

    assert result is None


def test_valid_prevalence_plan_passes() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.PREVALENCE,
        variables=["diabetes"],
        filters=[
            FilterCondition(
                variable="age",
                operator=FilterOperator.GREATER_THAN_OR_EQUAL,
                value=60,
            )
        ],
        group_by=["sex"],
        minimum_cell_size=10,
        aggregation_method=AggregationMethod.SUM,
    )

    policy.validate(plan)


def test_valid_numeric_summary_plan_passes() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.NUMERIC_SUMMARY,
        variables=[
            "age",
            "bmi",
            "systolic_bp",
        ],
        aggregation_method=(
            AggregationMethod.SUFFICIENT_STATISTICS
        ),
    )

    policy.validate(plan)


def test_valid_logistic_regression_plan_passes() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.TRAIN_MODEL,
        model=make_logistic_model(
            hyperparameters={
                "learning_rate": 0.01,
                "weight_decay": 0.0001,
            }
        ),
        aggregation_method=AggregationMethod.FEDAVG,
    )

    policy.validate(plan)


def test_valid_random_forest_ensemble_passes() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.TRAIN_MODEL,
        model=make_random_forest_model(
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_leaf": 10,
            }
        ),
        aggregation_method=(
            AggregationMethod.MODEL_ENSEMBLE
        ),
    )

    policy.validate(plan)


def test_missing_aggregation_method_is_allowed() -> None:
    """The gateway may select the permitted default later."""

    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.COHORT_COUNT,
        variables=["bmi"],
        aggregation_method=None,
    )

    policy.validate(plan)


# =====================================================================
# Operation tests
# =====================================================================


def test_disallowed_operation_is_rejected() -> None:
    config = PolicyConfig(
        allowed_operations=frozenset(
            {
                Operation.COHORT_COUNT,
            }
        )
    )

    policy = GlobalPolicy(config)

    plan = AnalysisPlan(
        operation=Operation.PREVALENCE,
        variables=["diabetes"],
    )

    with pytest.raises(
        PolicyViolation,
        match="Operation 'prevalence' is not permitted",
    ) as exception_info:
        policy.validate(plan)

    assert (
        exception_info.value.details["rule"]
        == "allowed_operations"
    )

    assert (
        exception_info.value.request_id
        == str(plan.request_id)
    )


# =====================================================================
# Variable tests
# =====================================================================


def test_prohibited_identifier_is_rejected() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.NUMERIC_SUMMARY,
        variables=["participant_id"],
    )

    with pytest.raises(
        PolicyViolation,
        match="prohibited variables",
    ) as exception_info:
        policy.validate(plan)

    assert exception_info.value.details == {
        "variables": ["participant_id"],
        "rule": "prohibited_variables",
    }


def test_unknown_variable_is_rejected() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.NUMERIC_SUMMARY,
        variables=["unknown_biomarker"],
    )

    with pytest.raises(
        PolicyViolation,
        match="not allowlisted",
    ) as exception_info:
        policy.validate(plan)

    assert (
        exception_info.value.details["variables"]
        == ["unknown_biomarker"]
    )

    assert (
        exception_info.value.details["rule"]
        == "allowed_variables"
    )


def test_filter_variable_is_checked() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.COHORT_COUNT,
        filters=[
            FilterCondition(
                variable="patient_id",
                operator=FilterOperator.EQUAL,
                value="patient-001",
            )
        ],
    )

    with pytest.raises(
        PolicyViolation,
        match="patient_id",
    ):
        policy.validate(plan)


def test_group_by_variable_is_checked() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.PREVALENCE,
        variables=["diabetes"],
        group_by=["hospital_room"],
    )

    with pytest.raises(
        PolicyViolation,
        match="hospital_room",
    ):
        policy.validate(plan)


def test_model_target_and_features_are_checked() -> None:
    policy = GlobalPolicy()

    model = ModelSpecification(
        model_type=ModelType.LOGISTIC_REGRESSION,
        target="diabetes",
        features=[
            "age",
            "unapproved_feature",
        ],
    )

    plan = AnalysisPlan(
        operation=Operation.TRAIN_MODEL,
        model=model,
        aggregation_method=AggregationMethod.FEDAVG,
    )

    with pytest.raises(
        PolicyViolation,
        match="unapproved_feature",
    ):
        policy.validate(plan)


# =====================================================================
# Privacy-threshold tests
# =====================================================================


def test_cell_size_below_policy_minimum_is_rejected() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.COHORT_COUNT,
        minimum_cell_size=5,
    )

    with pytest.raises(
        PolicyViolation,
        match="below the policy minimum",
    ) as exception_info:
        policy.validate(plan)

    assert exception_info.value.details == {
        "requested": 5,
        "required": 10,
        "rule": "minimum_cell_size",
    }


def test_custom_cell_size_policy_is_enforced() -> None:
    policy = GlobalPolicy(
        PolicyConfig(
            minimum_cell_size=20,
        )
    )

    plan = AnalysisPlan(
        operation=Operation.COHORT_COUNT,
        minimum_cell_size=10,
    )

    with pytest.raises(
        PolicyViolation,
        match="policy minimum of 20",
    ):
        policy.validate(plan)


# =====================================================================
# Site tests
# =====================================================================


def test_valid_requested_sites_pass() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.COHORT_COUNT,
        requested_sites=[
            "biobank-a",
            "biobank-c",
        ],
    )

    policy.validate(plan)


def test_empty_requested_sites_means_all_sites() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.COHORT_COUNT,
        requested_sites=[],
    )

    policy.validate(plan)


def test_unsupported_site_is_rejected() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.COHORT_COUNT,
        requested_sites=[
            "biobank-a",
            "unknown-site",
        ],
    )

    with pytest.raises(
        PolicyViolation,
        match="unknown-site",
    ) as exception_info:
        policy.validate(plan)

    assert exception_info.value.details == {
        "sites": ["unknown-site"],
        "rule": "allowed_sites",
    }


# =====================================================================
# Aggregation tests
# =====================================================================


def test_fedavg_is_rejected_for_cohort_count() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.COHORT_COUNT,
        aggregation_method=AggregationMethod.FEDAVG,
    )

    with pytest.raises(
        PolicyViolation,
        match="not permitted for operation",
    ) as exception_info:
        policy.validate(plan)

    assert (
        exception_info.value.details[
            "aggregation_method"
        ]
        == "fedavg"
    )


def test_weighted_mean_is_rejected_for_prevalence() -> None:
    """Prevalence should aggregate counts, not site percentages."""

    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.PREVALENCE,
        variables=["diabetes"],
        aggregation_method=(
            AggregationMethod.WEIGHTED_MEAN
        ),
    )

    with pytest.raises(
        PolicyViolation,
        match="not permitted for operation",
    ):
        policy.validate(plan)


def test_fedavg_is_rejected_for_random_forest() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.TRAIN_MODEL,
        model=make_random_forest_model(),
        aggregation_method=AggregationMethod.FEDAVG,
    )

    with pytest.raises(
        PolicyViolation,
        match="not permitted for model",
    ) as exception_info:
        policy.validate(plan)

    assert (
        exception_info.value.details["model"]
        == "random_forest"
    )

    assert (
        exception_info.value.details["permitted"]
        == ["model_ensemble"]
    )


def test_model_ensemble_is_rejected_for_logistic_regression() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.TRAIN_MODEL,
        model=make_logistic_model(),
        aggregation_method=(
            AggregationMethod.MODEL_ENSEMBLE
        ),
    )

    with pytest.raises(
        PolicyViolation,
        match="not permitted for model",
    ):
        policy.validate(plan)


# =====================================================================
# Model-policy tests
# =====================================================================


def test_model_not_allowed_by_custom_policy_is_rejected() -> None:
    policy = GlobalPolicy(
        PolicyConfig(
            allowed_models=frozenset(
                {
                    ModelType.LOGISTIC_REGRESSION,
                }
            )
        )
    )

    plan = AnalysisPlan(
        operation=Operation.TRAIN_MODEL,
        model=make_random_forest_model(),
        aggregation_method=(
            AggregationMethod.MODEL_ENSEMBLE
        ),
    )

    with pytest.raises(
        PolicyViolation,
        match="Model 'random_forest' is not permitted",
    ):
        policy.validate(plan)


def test_excessive_local_epochs_are_rejected() -> None:
    policy = GlobalPolicy(
        PolicyConfig(
            maximum_local_epochs=5,
        )
    )

    plan = AnalysisPlan(
        operation=Operation.TRAIN_MODEL,
        model=make_logistic_model(
            local_epochs=6,
        ),
        aggregation_method=AggregationMethod.FEDAVG,
    )

    with pytest.raises(
        PolicyViolation,
        match="local epochs",
    ) as exception_info:
        policy.validate(plan)

    assert exception_info.value.details == {
        "requested": 6,
        "maximum": 5,
        "rule": "maximum_local_epochs",
    }


def test_excessive_federated_rounds_are_rejected() -> None:
    policy = GlobalPolicy(
        PolicyConfig(
            maximum_federated_rounds=20,
        )
    )

    plan = AnalysisPlan(
        operation=Operation.TRAIN_MODEL,
        model=make_logistic_model(
            federated_rounds=21,
        ),
        aggregation_method=AggregationMethod.FEDAVG,
    )

    with pytest.raises(
        PolicyViolation,
        match="federated rounds",
    ) as exception_info:
        policy.validate(plan)

    assert exception_info.value.details == {
        "requested": 21,
        "maximum": 20,
        "rule": "maximum_federated_rounds",
    }


def test_unsupported_hyperparameter_is_rejected() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.TRAIN_MODEL,
        model=make_logistic_model(
            hyperparameters={
                "learning_rate": 0.01,
                "execute_python": "unsafe",
            }
        ),
        aggregation_method=AggregationMethod.FEDAVG,
    )

    with pytest.raises(
        PolicyViolation,
        match="Unsupported hyperparameters",
    ) as exception_info:
        policy.validate(plan)

    assert exception_info.value.details == {
        "model": "logistic_regression",
        "hyperparameters": ["execute_python"],
        "rule": "allowed_model_hyperparameters",
    }


# =====================================================================
# Metadata tests
# =====================================================================


def test_safe_metadata_is_allowed() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.COHORT_COUNT,
        metadata={
            "requested_by": "hackathon-demo",
            "purpose": "cohort feasibility",
        },
    )

    policy.validate(plan)


def test_prohibited_metadata_key_is_rejected() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.COHORT_COUNT,
        metadata={
            "raw_data": [
                {
                    "age": 71,
                    "diabetes": 1,
                }
            ]
        },
    )

    with pytest.raises(
        PolicyViolation,
        match="prohibited metadata",
    ) as exception_info:
        policy.validate(plan)

    assert exception_info.value.details == {
        "keys": ["raw_data"],
        "rule": "prohibited_metadata",
    }


def test_metadata_key_check_is_case_insensitive() -> None:
    policy = GlobalPolicy()

    plan = AnalysisPlan(
        operation=Operation.COHORT_COUNT,
        metadata={
            " Raw_Data ": "not permitted",
        },
    )

    with pytest.raises(
        PolicyViolation,
        match="raw_data",
    ):
        policy.validate(plan)


# =====================================================================
# Policy-configuration tests
# =====================================================================


def test_policy_rejects_invalid_minimum_cell_size() -> None:
    with pytest.raises(
        ValueError,
        match="minimum_cell_size",
    ):
        GlobalPolicy(
            PolicyConfig(
                minimum_cell_size=0,
            )
        )


def test_policy_rejects_invalid_epoch_limit() -> None:
    with pytest.raises(
        ValueError,
        match="maximum_local_epochs",
    ):
        GlobalPolicy(
            PolicyConfig(
                maximum_local_epochs=0,
            )
        )


def test_policy_rejects_allowed_prohibited_overlap() -> None:
    with pytest.raises(
        ValueError,
        match="both allowed and prohibited",
    ):
        GlobalPolicy(
            PolicyConfig(
                allowed_variables=frozenset(
                    {
                        "age",
                        "participant_id",
                    }
                ),
                prohibited_variables=frozenset(
                    {
                        "participant_id",
                    }
                ),
            )
        )