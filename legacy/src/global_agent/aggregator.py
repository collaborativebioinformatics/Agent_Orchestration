# global_agent/aggregator.py

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from global_agent.exceptions import FederatedResultValidationError
from global_agent.flare_gateway import FederatedDispatchResult
from shared.schemas import (
    AnalysisPlan,
    FederatedResult,
    
    Operation,
    ResponseStatus,
    SiteResponse,
)


class Aggregator(ABC):
    """Interface used by the global coordinator."""

    @abstractmethod
    def aggregate(
        self,
        *,
        plan: AnalysisPlan,
        dispatch_result: FederatedDispatchResult,
    ) -> FederatedResult:
        """Combine privacy-safe site responses."""
        raise NotImplementedError


class FederatedAggregator(Aggregator):
    """
    Deterministically combines privacy-safe site results.

    The aggregator never:

    - receives or combines raw patient rows;
    - averages site means directly;
    - averages site standard deviations directly;
    - uses an LLM for numerical calculations;
    - aggregates model parameters itself.

    Model parameter aggregation belongs to the NVFlare workflow.
    """

    DISCOVERY_OPERATIONS = {
        Operation.DISCOVER_DATASETS,
        Operation.DISCOVER_SCHEMA,
        Operation.SEARCH_COLUMNS,
        Operation.PROFILE_COLUMNS,
    }

    def aggregate(
        self,
        *,
        plan: AnalysisPlan,
        dispatch_result: FederatedDispatchResult,
    ) -> FederatedResult:
        responses = self._successful_responses(
            dispatch_result.responses
        )

        if not responses:
            raise FederatedResultValidationError(
                "No successful site responses are available.",
                request_id=dispatch_result.request_id,
            )

        self._validate_response_operations(
            expected_operation=plan.operation,
            responses=responses,
            request_id=dispatch_result.request_id,
        )

        aggregated_result = self._aggregate_operation(
            operation=plan.operation,
            responses=responses,
            request_id=dispatch_result.request_id,
        )

        minimum_cell_size = getattr(
            plan,
            "minimum_cell_size",
            10,
        )

        aggregated_result, suppressed_fields = (
            self._apply_suppression(
                result=aggregated_result,
                minimum_cell_size=minimum_cell_size,
            )
        )

        contributing_sites = [
            response.site_id for response in responses
        ]

        excluded_sites = [
            {
                "site_id": failure.site_id,
                "reason": failure.error_type,
                "message": failure.message,
                "retryable": failure.retryable,
            }
            for failure in dispatch_result.failures
        ]

        warnings = self._collect_warnings(responses)

        if dispatch_result.has_partial_failure:
            warnings.append(
                "The result is based on a subset of the requested sites."
            )

        if suppressed_fields:
            warnings.append(
                f"{len(suppressed_fields)} aggregate value(s) were "
                "suppressed by the minimum-cell-size policy."
            )

        return FederatedResult(
            request_id=dispatch_result.request_id,
            operation=plan.operation,
            status=ResponseStatus.SUCCESS,
            result=aggregated_result,
            contributing_sites=contributing_sites,
            excluded_sites=excluded_sites,
            minimum_cell_size=minimum_cell_size,
            suppressed=bool(suppressed_fields),
            suppressed_fields=suppressed_fields,
            warnings=warnings,
            metadata={
                "requested_site_count": dispatch_result.metadata.get(
                    "requested_site_count"
                ),
                "contributing_site_count": len(contributing_sites),
                "excluded_site_count": len(excluded_sites),
                "dispatch_elapsed_seconds": (
                    dispatch_result.elapsed_seconds
                ),
            },
        )

    def _aggregate_operation(
        self,
        *,
        operation: Operation,
        responses: Sequence[SiteResponse],
        request_id: str,
    ) -> dict[str, Any]:
        if operation in self.DISCOVERY_OPERATIONS:
            return self._aggregate_discovery(responses)

        if operation == Operation.COHORT_COUNT:
            return self._aggregate_count(responses)

        if operation in {
            Operation.NUMERIC_SUMMARY,
            Operation.STANDARDIZATION_STATISTICS,
        }:
            return self._aggregate_numeric_summary(responses)

        if operation == Operation.CATEGORICAL_DISTRIBUTION:
            return self._aggregate_categorical_distribution(responses)

        if operation == Operation.GROUPED_SUMMARY:
            return self._aggregate_grouped_summary(responses)

        if operation == Operation.MISSINGNESS:
            return self._aggregate_missingness(responses)

        if operation == Operation.HISTOGRAM:
            return self._aggregate_histogram(responses)

        if operation == Operation.TWO_DIMENSIONAL_HISTOGRAM:
            return self._aggregate_two_dimensional_histogram(
                responses
            )

        if operation == Operation.CORRELATION_STATISTICS:
            return self._aggregate_correlation_statistics(responses)

        if operation == Operation.EVALUATE_MODEL:
            return self._aggregate_model_metrics(responses)

        if operation == Operation.TRAIN_MODEL:
            return self._aggregate_training_metadata(responses)

        raise FederatedResultValidationError(
            f"Aggregation is not implemented for {operation.value!r}.",
            request_id=request_id,
        )

    @staticmethod
    def _successful_responses(
        responses: Sequence[SiteResponse],
    ) -> list[SiteResponse]:
        successful: list[SiteResponse] = []

        for response in responses:
            status = getattr(
                response,
                "status",
                ResponseStatus.SUCCESS,
            )

            if status == ResponseStatus.SUCCESS:
                successful.append(response)

        return successful

    @staticmethod
    def _validate_response_operations(
        *,
        expected_operation: Operation,
        responses: Sequence[SiteResponse],
        request_id: str,
    ) -> None:
        for response in responses:
            response_operation = getattr(
                response,
                "operation",
                None,
            )

            if response_operation is None:
                continue

            if response_operation != expected_operation:
                raise FederatedResultValidationError(
                    (
                        f"Site {response.site_id!r} returned operation "
                        f"{response_operation!r}; expected "
                        f"{expected_operation.value!r}."
                    ),
                    request_id=request_id,
                    details={"site_id": response.site_id},
                )

    @staticmethod
    def _site_result(
        response: SiteResponse,
    ) -> dict[str, Any]:
        """
        Extract the validated analytical result from a site response.

        SiteResponse uses `result`, not `payload`.
        """
        result = getattr(response, "result", None)

        if not isinstance(result, Mapping):
            raise FederatedResultValidationError(
                f"Site {response.site_id!r} returned an invalid result.",
                request_id=getattr(response, "request_id", None),
                details={"site_id": response.site_id},
            )

        return dict(result)

    def _aggregate_discovery(
        self,
        responses: Sequence[SiteResponse],
    ) -> dict[str, Any]:
        """
        Keep discovery metadata separated by site.

        Local schemas may differ, so discovery results should not be
        collapsed into a single assumed schema.
        """
        return {
            "sites": {
                response.site_id: self._site_result(response)
                for response in responses
            }
        }

    def _aggregate_count(
        self,
        responses: Sequence[SiteResponse],
    ) -> dict[str, Any]:
        """
        Required local result:

        {
            "count": 120
        }
        """
        total = 0

        for response in responses:
            result = self._site_result(response)

            total += self._nonnegative_int(
                result.get("count"),
                field="count",
                site_id=response.site_id,
            )

        return {
            "count": total,
            "site_count": len(responses),
        }

    def _aggregate_numeric_summary(
        self,
        responses: Sequence[SiteResponse],
    ) -> dict[str, Any]:
        """
        Required local result:

        {
            "columns": {
                "age": {
                    "count": 100,
                    "sum": 7025.0,
                    "sum_squares": 498125.0,
                    "min": 50.0,
                    "max": 91.0
                }
            }
        }
        """
        accumulators: dict[str, dict[str, Any]] = {}

        for response in responses:
            result = self._site_result(response)

            columns = self._require_mapping(
                result.get("columns"),
                field="columns",
                site_id=response.site_id,
            )

            for column, raw_statistics in columns.items():
                statistics = self._require_mapping(
                    raw_statistics,
                    field=f"columns.{column}",
                    site_id=response.site_id,
                )

                count = self._nonnegative_int(
                    statistics.get("count"),
                    field=f"{column}.count",
                    site_id=response.site_id,
                )

                column_sum = self._finite_float(
                    statistics.get("sum"),
                    field=f"{column}.sum",
                    site_id=response.site_id,
                )

                sum_squares = self._finite_float(
                    statistics.get("sum_squares"),
                    field=f"{column}.sum_squares",
                    site_id=response.site_id,
                )

                accumulator = accumulators.setdefault(
                    str(column),
                    {
                        "count": 0,
                        "sum": 0.0,
                        "sum_squares": 0.0,
                        "min": None,
                        "max": None,
                    },
                )

                accumulator["count"] += count
                accumulator["sum"] += column_sum
                accumulator["sum_squares"] += sum_squares

                if count > 0:
                    local_min = self._finite_float(
                        statistics.get("min"),
                        field=f"{column}.min",
                        site_id=response.site_id,
                    )

                    local_max = self._finite_float(
                        statistics.get("max"),
                        field=f"{column}.max",
                        site_id=response.site_id,
                    )

                    accumulator["min"] = (
                        local_min
                        if accumulator["min"] is None
                        else min(
                            accumulator["min"],
                            local_min,
                        )
                    )

                    accumulator["max"] = (
                        local_max
                        if accumulator["max"] is None
                        else max(
                            accumulator["max"],
                            local_max,
                        )
                    )

        return {
            "columns": {
                column: self._finalize_numeric_statistics(
                    statistics
                )
                for column, statistics in sorted(
                    accumulators.items()
                )
            }
        }

    def _aggregate_categorical_distribution(
        self,
        responses: Sequence[SiteResponse],
    ) -> dict[str, Any]:
        """
        Required local result:

        {
            "columns": {
                "diagnosis": {
                    "counts": {
                        "control": 60,
                        "case": 40
                    }
                }
            }
        }
        """
        combined: dict[str, defaultdict[str, int]] = {}

        for response in responses:
            result = self._site_result(response)

            columns = self._require_mapping(
                result.get("columns"),
                field="columns",
                site_id=response.site_id,
            )

            for column, raw_details in columns.items():
                details = self._require_mapping(
                    raw_details,
                    field=f"columns.{column}",
                    site_id=response.site_id,
                )

                counts = self._require_mapping(
                    details.get("counts"),
                    field=f"columns.{column}.counts",
                    site_id=response.site_id,
                )

                column_counts = combined.setdefault(
                    str(column),
                    defaultdict(int),
                )

                for category, raw_count in counts.items():
                    column_counts[str(category)] += (
                        self._nonnegative_int(
                            raw_count,
                            field=(
                                f"{column}.counts.{category}"
                            ),
                            site_id=response.site_id,
                        )
                    )

        return {
            "columns": {
                column: {
                    "counts": dict(sorted(counts.items())),
                    "total": sum(counts.values()),
                }
                for column, counts in sorted(combined.items())
            }
        }

    def _aggregate_grouped_summary(
        self,
        responses: Sequence[SiteResponse],
    ) -> dict[str, Any]:
        """
        Required local result:

        {
            "groups": {
                "control": {
                    "count": 50,
                    "columns": {
                        "age": {
                            "count": 50,
                            "sum": 3400,
                            "sum_squares": 233000,
                            "min": 50,
                            "max": 85
                        }
                    }
                }
            }
        }
        """
        group_counts: defaultdict[str, int] = defaultdict(int)

        group_accumulators: dict[
            str,
            dict[str, dict[str, Any]],
        ] = defaultdict(dict)

        for response in responses:
            result = self._site_result(response)

            groups = self._require_mapping(
                result.get("groups"),
                field="groups",
                site_id=response.site_id,
            )

            for raw_group_name, raw_group in groups.items():
                group_name = str(raw_group_name)

                group = self._require_mapping(
                    raw_group,
                    field=f"groups.{group_name}",
                    site_id=response.site_id,
                )

                group_counts[group_name] += (
                    self._nonnegative_int(
                        group.get("count"),
                        field=f"groups.{group_name}.count",
                        site_id=response.site_id,
                    )
                )

                columns = self._require_mapping(
                    group.get("columns"),
                    field=f"groups.{group_name}.columns",
                    site_id=response.site_id,
                )

                for raw_column, raw_statistics in columns.items():
                    column = str(raw_column)

                    statistics = self._require_mapping(
                        raw_statistics,
                        field=(
                            f"groups.{group_name}."
                            f"columns.{column}"
                        ),
                        site_id=response.site_id,
                    )

                    count = self._nonnegative_int(
                        statistics.get("count"),
                        field=f"{group_name}.{column}.count",
                        site_id=response.site_id,
                    )

                    column_sum = self._finite_float(
                        statistics.get("sum"),
                        field=f"{group_name}.{column}.sum",
                        site_id=response.site_id,
                    )

                    sum_squares = self._finite_float(
                        statistics.get("sum_squares"),
                        field=(
                            f"{group_name}.{column}."
                            "sum_squares"
                        ),
                        site_id=response.site_id,
                    )

                    accumulator = group_accumulators[
                        group_name
                    ].setdefault(
                        column,
                        {
                            "count": 0,
                            "sum": 0.0,
                            "sum_squares": 0.0,
                            "min": None,
                            "max": None,
                        },
                    )

                    accumulator["count"] += count
                    accumulator["sum"] += column_sum
                    accumulator["sum_squares"] += sum_squares

                    if count > 0:
                        local_min = self._finite_float(
                            statistics.get("min"),
                            field=(
                                f"{group_name}.{column}.min"
                            ),
                            site_id=response.site_id,
                        )

                        local_max = self._finite_float(
                            statistics.get("max"),
                            field=(
                                f"{group_name}.{column}.max"
                            ),
                            site_id=response.site_id,
                        )

                        accumulator["min"] = (
                            local_min
                            if accumulator["min"] is None
                            else min(
                                accumulator["min"],
                                local_min,
                            )
                        )

                        accumulator["max"] = (
                            local_max
                            if accumulator["max"] is None
                            else max(
                                accumulator["max"],
                                local_max,
                            )
                        )

        groups_result: dict[str, Any] = {}

        for group_name in sorted(group_accumulators):
            groups_result[group_name] = {
                "count": group_counts[group_name],
                "columns": {
                    column: self._finalize_numeric_statistics(
                        statistics
                    )
                    for column, statistics in sorted(
                        group_accumulators[
                            group_name
                        ].items()
                    )
                },
            }

        return {"groups": groups_result}

    def _aggregate_missingness(
        self,
        responses: Sequence[SiteResponse],
    ) -> dict[str, Any]:
        """
        Required local result:

        {
            "columns": {
                "age": {
                    "missing": 5,
                    "observed": 95
                }
            }
        }
        """
        combined: dict[str, dict[str, int]] = {}

        for response in responses:
            result = self._site_result(response)

            columns = self._require_mapping(
                result.get("columns"),
                field="columns",
                site_id=response.site_id,
            )

            for raw_column, raw_counts in columns.items():
                column = str(raw_column)

                counts = self._require_mapping(
                    raw_counts,
                    field=f"columns.{column}",
                    site_id=response.site_id,
                )

                accumulator = combined.setdefault(
                    column,
                    {
                        "missing": 0,
                        "observed": 0,
                    },
                )

                accumulator["missing"] += self._nonnegative_int(
                    counts.get("missing"),
                    field=f"{column}.missing",
                    site_id=response.site_id,
                )

                accumulator["observed"] += self._nonnegative_int(
                    counts.get("observed"),
                    field=f"{column}.observed",
                    site_id=response.site_id,
                )

        for counts in combined.values():
            total = counts["missing"] + counts["observed"]

            counts["total"] = total
            counts["missing_fraction"] = (
                counts["missing"] / total
                if total > 0
                else None
            )

        return {"columns": combined}

    def _aggregate_histogram(
        self,
        responses: Sequence[SiteResponse],
    ) -> dict[str, Any]:
        """
        Required local result:

        {
            "column": "age",
            "bin_edges": [40, 50, 60, 70],
            "counts": [10, 20, 30]
        }
        """
        first = self._site_result(responses[0])

        column = first.get("column")

        bin_edges = self._float_list(
            first.get("bin_edges"),
            field="bin_edges",
            site_id=responses[0].site_id,
        )

        if len(bin_edges) < 2:
            raise FederatedResultValidationError(
                "A histogram requires at least two bin edges."
            )

        self._validate_increasing_edges(bin_edges)

        aggregated_counts = [0] * (len(bin_edges) - 1)

        for response in responses:
            result = self._site_result(response)

            if result.get("column") != column:
                self._incompatible(
                    response.site_id,
                    "Histogram columns differ between sites.",
                )

            local_edges = self._float_list(
                result.get("bin_edges"),
                field="bin_edges",
                site_id=response.site_id,
            )

            if not self._lists_close(
                bin_edges,
                local_edges,
            ):
                self._incompatible(
                    response.site_id,
                    "Histogram bin edges differ between sites.",
                )

            local_counts = self._integer_list(
                result.get("counts"),
                field="counts",
                site_id=response.site_id,
            )

            if len(local_counts) != len(aggregated_counts):
                self._incompatible(
                    response.site_id,
                    "Histogram dimensions differ between sites.",
                )

            aggregated_counts = [
                global_count + local_count
                for global_count, local_count in zip(
                    aggregated_counts,
                    local_counts,
                    strict=True,
                )
            ]

        return {
            "column": column,
            "bin_edges": bin_edges,
            "counts": aggregated_counts,
            "total": sum(aggregated_counts),
        }

    def _aggregate_two_dimensional_histogram(
        self,
        responses: Sequence[SiteResponse],
    ) -> dict[str, Any]:
        """
        Required local result:

        {
            "x_column": "age",
            "y_column": "score",
            "x_bin_edges": [...],
            "y_bin_edges": [...],
            "counts": [[...], [...]]
        }
        """
        first = self._site_result(responses[0])

        x_column = first.get("x_column")
        y_column = first.get("y_column")

        x_edges = self._float_list(
            first.get("x_bin_edges"),
            field="x_bin_edges",
            site_id=responses[0].site_id,
        )

        y_edges = self._float_list(
            first.get("y_bin_edges"),
            field="y_bin_edges",
            site_id=responses[0].site_id,
        )

        self._validate_increasing_edges(x_edges)
        self._validate_increasing_edges(y_edges)

        row_count = len(x_edges) - 1
        column_count = len(y_edges) - 1

        aggregated_counts = [
            [0 for _ in range(column_count)]
            for _ in range(row_count)
        ]

        for response in responses:
            result = self._site_result(response)

            if (
                result.get("x_column") != x_column
                or result.get("y_column") != y_column
            ):
                self._incompatible(
                    response.site_id,
                    "Two-dimensional histogram columns differ.",
                )

            local_x_edges = self._float_list(
                result.get("x_bin_edges"),
                field="x_bin_edges",
                site_id=response.site_id,
            )

            local_y_edges = self._float_list(
                result.get("y_bin_edges"),
                field="y_bin_edges",
                site_id=response.site_id,
            )

            if not self._lists_close(
                x_edges,
                local_x_edges,
            ):
                self._incompatible(
                    response.site_id,
                    "X-axis bin edges differ between sites.",
                )

            if not self._lists_close(
                y_edges,
                local_y_edges,
            ):
                self._incompatible(
                    response.site_id,
                    "Y-axis bin edges differ between sites.",
                )

            matrix = result.get("counts")

            if not isinstance(matrix, list):
                self._incompatible(
                    response.site_id,
                    "Two-dimensional counts must be a matrix.",
                )

            if len(matrix) != row_count:
                self._incompatible(
                    response.site_id,
                    "Two-dimensional histogram rows differ.",
                )

            for row_index, raw_row in enumerate(matrix):
                row = self._integer_list(
                    raw_row,
                    field=f"counts.{row_index}",
                    site_id=response.site_id,
                )

                if len(row) != column_count:
                    self._incompatible(
                        response.site_id,
                        "Two-dimensional histogram columns differ.",
                    )

                for column_index, value in enumerate(row):
                    aggregated_counts[row_index][
                        column_index
                    ] += value

        return {
            "x_column": x_column,
            "y_column": y_column,
            "x_bin_edges": x_edges,
            "y_bin_edges": y_edges,
            "counts": aggregated_counts,
            "total": sum(
                sum(row) for row in aggregated_counts
            ),
        }

    def _aggregate_correlation_statistics(
        self,
        responses: Sequence[SiteResponse],
    ) -> dict[str, Any]:
        """
        Required local result:

        {
            "columns": ["age", "score"],
            "count": 100,
            "sums": [7000, 4200],
            "cross_products": [
                [500000, 300000],
                [300000, 190000]
            ]
        }
        """
        first = self._site_result(responses[0])
        columns = list(first.get("columns") or [])

        if len(columns) < 2:
            raise FederatedResultValidationError(
                "Correlation requires at least two columns."
            )

        dimension = len(columns)
        total_count = 0
        sums = [0.0] * dimension

        cross_products = [
            [0.0] * dimension
            for _ in range(dimension)
        ]

        for response in responses:
            result = self._site_result(response)

            if list(result.get("columns") or []) != columns:
                self._incompatible(
                    response.site_id,
                    "Correlation column order differs between sites.",
                )

            count = self._nonnegative_int(
                result.get("count"),
                field="count",
                site_id=response.site_id,
            )

            local_sums = self._float_list(
                result.get("sums"),
                field="sums",
                site_id=response.site_id,
            )

            if len(local_sums) != dimension:
                self._incompatible(
                    response.site_id,
                    "Correlation sum dimensions differ.",
                )

            local_cross_products = result.get(
                "cross_products"
            )

            if (
                not isinstance(local_cross_products, list)
                or len(local_cross_products) != dimension
            ):
                self._incompatible(
                    response.site_id,
                    "Cross-product dimensions differ.",
                )

            total_count += count

            for index, value in enumerate(local_sums):
                sums[index] += value

            for row_index, raw_row in enumerate(
                local_cross_products
            ):
                row = self._float_list(
                    raw_row,
                    field=f"cross_products.{row_index}",
                    site_id=response.site_id,
                )

                if len(row) != dimension:
                    self._incompatible(
                        response.site_id,
                        "Cross-product dimensions differ.",
                    )

                for column_index, value in enumerate(row):
                    cross_products[row_index][
                        column_index
                    ] += value

        correlation_matrix = self._correlation_matrix(
            count=total_count,
            sums=sums,
            cross_products=cross_products,
        )

        return {
            "columns": columns,
            "count": total_count,
            "sums": sums,
            "cross_products": cross_products,
            "correlation_matrix": correlation_matrix,
        }

    def _aggregate_model_metrics(
        self,
        responses: Sequence[SiteResponse],
    ) -> dict[str, Any]:
        """
        Required local result:

        {
            "sample_count": 100,
            "metrics": {
                "accuracy": 0.84,
                "loss": 0.42
            }
        }

        Site metrics are weighted by sample count.
        """
        total_samples = 0

        weighted_sums: defaultdict[str, float] = defaultdict(
            float
        )

        metric_weights: defaultdict[str, int] = defaultdict(
            int
        )

        for response in responses:
            result = self._site_result(response)

            sample_count = self._nonnegative_int(
                result.get("sample_count"),
                field="sample_count",
                site_id=response.site_id,
            )

            metrics = self._require_mapping(
                result.get("metrics"),
                field="metrics",
                site_id=response.site_id,
            )

            total_samples += sample_count

            for raw_metric_name, raw_value in metrics.items():
                metric_name = str(raw_metric_name)

                value = self._finite_float(
                    raw_value,
                    field=f"metrics.{metric_name}",
                    site_id=response.site_id,
                )

                weighted_sums[metric_name] += (
                    value * sample_count
                )

                metric_weights[metric_name] += sample_count

        aggregated_metrics = {
            metric_name: (
                weighted_sums[metric_name]
                / metric_weights[metric_name]
                if metric_weights[metric_name] > 0
                else None
            )
            for metric_name in sorted(weighted_sums)
        }

        return {
            "sample_count": total_samples,
            "metrics": aggregated_metrics,
        }

    def _aggregate_training_metadata(
        self,
        responses: Sequence[SiteResponse],
    ) -> dict[str, Any]:
        """
        Aggregate only safe training metrics.

        NVFlare is responsible for aggregating model updates.
        """
        metrics = self._aggregate_model_metrics(responses)

        return {
            **metrics,
            "training_sites": [
                response.site_id for response in responses
            ],
            "message": (
                "Model parameters were aggregated by the federated "
                "training workflow. This result contains only safe "
                "training metrics and metadata."
            ),
        }

    @staticmethod
    def _finalize_numeric_statistics(
        statistics: Mapping[str, Any],
    ) -> dict[str, Any]:
        count = int(statistics["count"])
        column_sum = float(statistics["sum"])
        sum_squares = float(statistics["sum_squares"])

        if count == 0:
            return {
                "count": 0,
                "sum": 0.0,
                "sum_squares": 0.0,
                "mean": None,
                "variance": None,
                "standard_deviation": None,
                "min": None,
                "max": None,
            }

        mean = column_sum / count

        if count > 1:
            variance_numerator = (
                sum_squares
                - (column_sum * column_sum / count)
            )

            # Floating-point rounding can produce a very small
            # negative number for theoretically zero variance.
            variance_numerator = max(
                variance_numerator,
                0.0,
            )

            variance = variance_numerator / (count - 1)
            standard_deviation = math.sqrt(variance)
        else:
            variance = None
            standard_deviation = None

        return {
            "count": count,
            "sum": column_sum,
            "sum_squares": sum_squares,
            "mean": mean,
            "variance": variance,
            "standard_deviation": standard_deviation,
            "min": statistics["min"],
            "max": statistics["max"],
        }

    @staticmethod
    def _correlation_matrix(
        *,
        count: int,
        sums: Sequence[float],
        cross_products: Sequence[Sequence[float]],
    ) -> list[list[float | None]]:
        dimension = len(sums)

        if count < 2:
            return [
                [None for _ in range(dimension)]
                for _ in range(dimension)
            ]

        centered_cross_products = [
            [
                cross_products[row][column]
                - (
                    sums[row]
                    * sums[column]
                    / count
                )
                for column in range(dimension)
            ]
            for row in range(dimension)
        ]

        correlation_matrix: list[
            list[float | None]
        ] = []

        for row in range(dimension):
            correlation_row: list[float | None] = []

            for column in range(dimension):
                denominator = math.sqrt(
                    max(
                        centered_cross_products[row][row],
                        0.0,
                    )
                    * max(
                        centered_cross_products[column][column],
                        0.0,
                    )
                )

                if denominator == 0:
                    correlation_row.append(None)
                    continue

                correlation = (
                    centered_cross_products[row][column]
                    / denominator
                )

                # Protect against floating-point results such as
                # 1.0000000000000002.
                correlation_row.append(
                    max(-1.0, min(1.0, correlation))
                )

            correlation_matrix.append(correlation_row)

        return correlation_matrix

    def _apply_suppression(
        self,
        *,
        result: dict[str, Any],
        minimum_cell_size: int,
    ) -> tuple[dict[str, Any], list[str]]:
        """
        Suppress non-zero count values below the minimum cell size.

        Zero counts are retained because zero does not expose a small
        non-zero cohort.
        """
        suppressed_fields: list[str] = []

        ordinary_count_fields = {
            "count",
            "missing",
            "observed",
            "sample_count",
            "site_count",
            "total",
        }

        def suppress_count(
            value: Any,
            path: str,
        ) -> Any:
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 < value < minimum_cell_size
            ):
                suppressed_fields.append(path)
                return None

            return value

        def visit_histogram_counts(
            value: Any,
            path: str,
        ) -> Any:
            if isinstance(value, list):
                return [
                    visit_histogram_counts(
                        child,
                        f"{path}.{index}",
                    )
                    for index, child in enumerate(value)
                ]

            return suppress_count(value, path)

        def visit(
            value: Any,
            path: str = "",
            parent_key: str = "",
        ) -> Any:
            if isinstance(value, dict):
                output: dict[str, Any] = {}

                for key, child in value.items():
                    child_path = (
                        f"{path}.{key}"
                        if path
                        else str(key)
                    )

                    if key == "counts":
                        if isinstance(child, dict):
                            output[key] = {
                                category: suppress_count(
                                    category_count,
                                    (
                                        f"{child_path}."
                                        f"{category}"
                                    ),
                                )
                                for category, category_count
                                in child.items()
                            }
                        else:
                            output[key] = (
                                visit_histogram_counts(
                                    child,
                                    child_path,
                                )
                            )

                    elif (
                        key in ordinary_count_fields
                        and isinstance(child, int)
                        and not isinstance(child, bool)
                    ):
                        output[key] = suppress_count(
                            child,
                            child_path,
                        )

                    else:
                        output[key] = visit(
                            child,
                            child_path,
                            str(key),
                        )

                return output

            if isinstance(value, list):
                return [
                    visit(
                        child,
                        f"{path}.{index}",
                        parent_key,
                    )
                    for index, child in enumerate(value)
                ]

            return value

        suppressed_result = visit(result)

        return suppressed_result, suppressed_fields

    @staticmethod
    def _collect_warnings(
        responses: Sequence[SiteResponse],
    ) -> list[str]:
        warnings: list[str] = []

        for response in responses:
            site_warnings = (
                getattr(response, "warnings", None) or []
            )

            for warning in site_warnings:
                message = f"{response.site_id}: {warning}"

                if message not in warnings:
                    warnings.append(message)

        return warnings

    @staticmethod
    def _require_mapping(
        value: Any,
        *,
        field: str,
        site_id: str,
    ) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise FederatedResultValidationError(
                (
                    f"Site {site_id!r}: {field!r} "
                    "must be an object."
                ),
                details={
                    "site_id": site_id,
                    "field": field,
                },
            )

        return value

    @staticmethod
    def _nonnegative_int(
        value: Any,
        *,
        field: str,
        site_id: str,
    ) -> int:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise FederatedResultValidationError(
                (
                    f"Site {site_id!r}: {field!r} must be "
                    "a non-negative integer."
                ),
                details={
                    "site_id": site_id,
                    "field": field,
                },
            )

        return value

    @staticmethod
    def _finite_float(
        value: Any,
        *,
        field: str,
        site_id: str,
    ) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise FederatedResultValidationError(
                (
                    f"Site {site_id!r}: {field!r} "
                    "must be numeric."
                ),
                details={
                    "site_id": site_id,
                    "field": field,
                },
            )

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise FederatedResultValidationError(
                (
                    f"Site {site_id!r}: {field!r} "
                    "must be finite."
                ),
                details={
                    "site_id": site_id,
                    "field": field,
                },
            )

        return numeric_value

    def _float_list(
        self,
        value: Any,
        *,
        field: str,
        site_id: str,
    ) -> list[float]:
        if not isinstance(value, list):
            raise FederatedResultValidationError(
                (
                    f"Site {site_id!r}: {field!r} "
                    "must be a list."
                ),
                details={
                    "site_id": site_id,
                    "field": field,
                },
            )

        return [
            self._finite_float(
                item,
                field=f"{field}.{index}",
                site_id=site_id,
            )
            for index, item in enumerate(value)
        ]

    def _integer_list(
        self,
        value: Any,
        *,
        field: str,
        site_id: str,
    ) -> list[int]:
        if not isinstance(value, list):
            raise FederatedResultValidationError(
                (
                    f"Site {site_id!r}: {field!r} "
                    "must be a list."
                ),
                details={
                    "site_id": site_id,
                    "field": field,
                },
            )

        return [
            self._nonnegative_int(
                item,
                field=f"{field}.{index}",
                site_id=site_id,
            )
            for index, item in enumerate(value)
        ]

    @staticmethod
    def _validate_increasing_edges(
        edges: Sequence[float],
    ) -> None:
        if len(edges) < 2:
            raise FederatedResultValidationError(
                "Histogram requires at least two bin edges."
            )

        if any(
            right <= left
            for left, right in zip(
                edges,
                edges[1:],
                strict=True,
            )
        ):
            raise FederatedResultValidationError(
                "Histogram bin edges must be strictly increasing."
            )

    @staticmethod
    def _lists_close(
        left: Sequence[float],
        right: Sequence[float],
    ) -> bool:
        return (
            len(left) == len(right)
            and all(
                math.isclose(
                    left_value,
                    right_value,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
                for left_value, right_value in zip(
                    left,
                    right,
                    strict=True,
                )
            )
        )

    @staticmethod
    def _incompatible(
        site_id: str,
        message: str,
    ) -> None:
        raise FederatedResultValidationError(
            f"Site {site_id!r}: {message}",
            details={"site_id": site_id},
        )