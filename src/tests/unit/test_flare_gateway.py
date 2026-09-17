# tests/unit/test_flare_gateway.py

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from global_agent.exceptions import (
    FederatedExecutionError,
    NoEligibleSitesError,
)
from global_agent.flare_gateway import (
    MockFederatedGateway,
    TransportFederatedGateway,
)
from shared.schemas import AnalysisPlan, Operation, SiteResponse


def run(coroutine):
    """
    Run async gateway methods without requiring pytest-asyncio.
    """
    return asyncio.run(coroutine)


def make_plan() -> AnalysisPlan:
    """
    Construct a minimal plan for gateway unit tests.

    AnalysisPlan validation is tested separately in test_schemas.py.
    """
    return AnalysisPlan.model_construct(
        operation=Operation.COHORT_COUNT,
        dataset_id="demo_dataset",
        columns=[],
        filters=[],
        group_by=[],
        requested_sites=None,
        minimum_cell_size=10,
        metadata={},
    )


def make_site_response(
    site_id: str,
    request_id: str = "request-001",
    count: int = 100,
) -> SiteResponse:
    """
    Construct a minimal response for testing gateway orchestration.

    The aggregator tests will verify the detailed result payload.
    """
    return SiteResponse.model_construct(
        site_id=site_id,
        request_id=request_id,
        payload={"count": count},
    )


def test_mock_gateway_dispatches_to_three_sites() -> None:
    plan = make_plan()

    gateway = MockFederatedGateway(
        handlers={
            "biobank-a": lambda plan, request_id: make_site_response(
                "biobank-a", request_id, 100
            ),
            "biobank-b": lambda plan, request_id: make_site_response(
                "biobank-b", request_id, 200
            ),
            "biobank-c": lambda plan, request_id: make_site_response(
                "biobank-c", request_id, 300
            ),
        }
    )

    result = run(
        gateway.execute(
            plan=plan,
            site_ids=["biobank-a", "biobank-b", "biobank-c"],
            request_id="request-001",
        )
    )

    assert result.request_id == "request-001"
    assert len(result.responses) == 3
    assert result.failures == ()

    assert result.successful_site_ids == (
        "biobank-a",
        "biobank-b",
        "biobank-c",
    )
    assert result.failed_site_ids == ()
    assert result.has_partial_failure is False
    assert result.all_sites_failed is False

    assert result.metadata["requested_site_count"] == 3
    assert result.metadata["successful_site_count"] == 3
    assert result.metadata["failed_site_count"] == 0


def test_mock_gateway_passes_same_plan_and_request_id_to_handler() -> None:
    plan = make_plan()
    received: dict[str, Any] = {}

    def handler(
        received_plan: AnalysisPlan,
        received_request_id: str,
    ) -> SiteResponse:
        received["plan"] = received_plan
        received["request_id"] = received_request_id

        return make_site_response(
            site_id="biobank-a",
            request_id=received_request_id,
        )

    gateway = MockFederatedGateway(
        handlers={"biobank-a": handler}
    )

    run(
        gateway.execute(
            plan=plan,
            site_ids=["biobank-a"],
            request_id="fixed-request-id",
        )
    )

    assert received["plan"] is plan
    assert received["request_id"] == "fixed-request-id"


def test_mock_gateway_generates_request_id_when_missing() -> None:
    captured_request_ids: list[str] = []

    def handler(
        plan: AnalysisPlan,
        request_id: str,
    ) -> SiteResponse:
        captured_request_ids.append(request_id)
        return make_site_response("biobank-a", request_id)

    gateway = MockFederatedGateway(
        handlers={"biobank-a": handler}
    )

    result = run(
        gateway.execute(
            plan=make_plan(),
            site_ids=["biobank-a"],
        )
    )

    assert result.request_id
    assert captured_request_ids == [result.request_id]


def test_mock_gateway_removes_duplicate_site_ids() -> None:
    call_counts = {
        "biobank-a": 0,
        "biobank-b": 0,
    }

    def site_a(
        plan: AnalysisPlan,
        request_id: str,
    ) -> SiteResponse:
        call_counts["biobank-a"] += 1
        return make_site_response("biobank-a", request_id)

    def site_b(
        plan: AnalysisPlan,
        request_id: str,
    ) -> SiteResponse:
        call_counts["biobank-b"] += 1
        return make_site_response("biobank-b", request_id)

    gateway = MockFederatedGateway(
        handlers={
            "biobank-a": site_a,
            "biobank-b": site_b,
        }
    )

    result = run(
        gateway.execute(
            plan=make_plan(),
            site_ids=[
                "biobank-a",
                "biobank-a",
                "biobank-b",
                "biobank-b",
            ],
        )
    )

    assert len(result.responses) == 2
    assert call_counts == {
        "biobank-a": 1,
        "biobank-b": 1,
    }


def test_mock_gateway_strips_site_id_whitespace() -> None:
    gateway = MockFederatedGateway(
        handlers={
            "biobank-a": lambda plan, request_id: make_site_response(
                "biobank-a",
                request_id,
            )
        }
    )

    result = run(
        gateway.execute(
            plan=make_plan(),
            site_ids=["  biobank-a  "],
        )
    )

    assert result.successful_site_ids == ("biobank-a",)


def test_mock_gateway_supports_async_site_handlers() -> None:
    async def async_handler(
        plan: AnalysisPlan,
        request_id: str,
    ) -> SiteResponse:
        await asyncio.sleep(0)
        return make_site_response("biobank-a", request_id)

    gateway = MockFederatedGateway(
        handlers={"biobank-a": async_handler}
    )

    result = run(
        gateway.execute(
            plan=make_plan(),
            site_ids=["biobank-a"],
        )
    )

    assert result.successful_site_ids == ("biobank-a",)


def test_mock_gateway_records_partial_failure() -> None:
    def failing_handler(
        plan: AnalysisPlan,
        request_id: str,
    ) -> SiteResponse:
        raise ConnectionError("Site is temporarily unavailable")

    gateway = MockFederatedGateway(
        handlers={
            "biobank-a": lambda plan, request_id: make_site_response(
                "biobank-a", request_id, 100
            ),
            "biobank-b": failing_handler,
            "biobank-c": lambda plan, request_id: make_site_response(
                "biobank-c", request_id, 300
            ),
        },
        fail_on_partial_error=False,
    )

    result = run(
        gateway.execute(
            plan=make_plan(),
            site_ids=["biobank-a", "biobank-b", "biobank-c"],
        )
    )

    assert result.successful_site_ids == (
        "biobank-a",
        "biobank-c",
    )
    assert result.failed_site_ids == ("biobank-b",)
    assert result.has_partial_failure is True
    assert result.all_sites_failed is False

    failure = result.failures[0]

    assert failure.site_id == "biobank-b"
    assert failure.error_type == "ConnectionError"
    assert "temporarily unavailable" in failure.message


def test_mock_gateway_can_reject_partial_failure() -> None:
    def failing_handler(
        plan: AnalysisPlan,
        request_id: str,
    ) -> SiteResponse:
        raise RuntimeError("Local agent failed")

    gateway = MockFederatedGateway(
        handlers={
            "biobank-a": lambda plan, request_id: make_site_response(
                "biobank-a", request_id
            ),
            "biobank-b": failing_handler,
        },
        fail_on_partial_error=True,
    )

    with pytest.raises(
        FederatedExecutionError,
        match="One or more mock sites failed",
    ):
        run(
            gateway.execute(
                plan=make_plan(),
                site_ids=["biobank-a", "biobank-b"],
            )
        )


def test_mock_gateway_records_unregistered_site_as_failure() -> None:
    gateway = MockFederatedGateway(
        handlers={
            "biobank-a": lambda plan, request_id: make_site_response(
                "biobank-a", request_id
            )
        }
    )

    result = run(
        gateway.execute(
            plan=make_plan(),
            site_ids=["biobank-a", "unknown-biobank"],
        )
    )

    assert result.successful_site_ids == ("biobank-a",)
    assert result.failed_site_ids == ("unknown-biobank",)

    failure = result.failures[0]

    assert failure.error_type == "site_not_registered"
    assert failure.retryable is False


def test_mock_gateway_records_async_timeout() -> None:
    async def slow_handler(
        plan: AnalysisPlan,
        request_id: str,
    ) -> SiteResponse:
        await asyncio.sleep(0.1)
        return make_site_response("slow-biobank", request_id)

    gateway = MockFederatedGateway(
        handlers={
            "fast-biobank": lambda plan, request_id: make_site_response(
                "fast-biobank", request_id
            ),
            "slow-biobank": slow_handler,
        },
        timeout_seconds=0.01,
    )

    result = run(
        gateway.execute(
            plan=make_plan(),
            site_ids=["fast-biobank", "slow-biobank"],
        )
    )

    assert result.successful_site_ids == ("fast-biobank",)
    assert result.failed_site_ids == ("slow-biobank",)

    failure = result.failures[0]

    assert failure.error_type == "timeout"
    assert failure.retryable is True


def test_mock_gateway_rejects_mismatched_response_site_id() -> None:
    gateway = MockFederatedGateway(
        handlers={
            "biobank-a": lambda plan, request_id: make_site_response(
                "different-biobank",
                request_id,
            ),
            "biobank-b": lambda plan, request_id: make_site_response(
                "biobank-b",
                request_id,
            ),
        }
    )

    result = run(
        gateway.execute(
            plan=make_plan(),
            site_ids=["biobank-a", "biobank-b"],
        )
    )

    assert result.successful_site_ids == ("biobank-b",)
    assert result.failed_site_ids == ("biobank-a",)

    failure = result.failures[0]

    assert failure.error_type == "ValueError"
    assert "does not match" in failure.message


def test_mock_gateway_rejects_malformed_response() -> None:
    gateway = MockFederatedGateway(
        handlers={
            "malformed-site": lambda plan, request_id: {},
            "valid-site": lambda plan, request_id: make_site_response(
                "valid-site",
                request_id,
            ),
        }
    )

    result = run(
        gateway.execute(
            plan=make_plan(),
            site_ids=["malformed-site", "valid-site"],
        )
    )

    assert result.successful_site_ids == ("valid-site",)
    assert result.failed_site_ids == ("malformed-site",)

    failure = result.failures[0]

    assert failure.error_type == "invalid_response"
    assert "schema validation" in failure.message


def test_mock_gateway_raises_when_all_sites_fail() -> None:
    def failing_handler(
        plan: AnalysisPlan,
        request_id: str,
    ) -> SiteResponse:
        raise RuntimeError("Local execution failed")

    gateway = MockFederatedGateway(
        handlers={
            "biobank-a": failing_handler,
            "biobank-b": failing_handler,
            "biobank-c": failing_handler,
        }
    )

    with pytest.raises(
        FederatedExecutionError,
        match="Every selected mock site failed",
    ):
        run(
            gateway.execute(
                plan=make_plan(),
                site_ids=["biobank-a", "biobank-b", "biobank-c"],
            )
        )


@pytest.mark.parametrize(
    "site_ids",
    [
        [],
        [""],
        ["   "],
        ["", "  "],
    ],
)
def test_mock_gateway_rejects_empty_site_list(
    site_ids: list[str],
) -> None:
    gateway = MockFederatedGateway()

    with pytest.raises(
        NoEligibleSitesError,
        match="No eligible sites",
    ):
        run(
            gateway.execute(
                plan=make_plan(),
                site_ids=site_ids,
            )
        )


def test_register_and_unregister_mock_site() -> None:
    gateway = MockFederatedGateway()

    handler = lambda plan, request_id: make_site_response(
        "biobank-a",
        request_id,
    )

    gateway.register_site("biobank-a", handler)

    assert gateway.registered_site_ids == ("biobank-a",)

    gateway.unregister_site("biobank-a")

    assert gateway.registered_site_ids == ()


def test_registering_duplicate_mock_site_raises_error() -> None:
    gateway = MockFederatedGateway()

    handler = lambda plan, request_id: make_site_response(
        "biobank-a",
        request_id,
    )

    gateway.register_site("biobank-a", handler)

    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        gateway.register_site("biobank-a", handler)


def test_replace_mock_site_handler() -> None:
    gateway = MockFederatedGateway(
        handlers={
            "biobank-a": lambda plan, request_id: make_site_response(
                "biobank-a",
                request_id,
                count=10,
            )
        }
    )

    gateway.replace_site(
        "biobank-a",
        lambda plan, request_id: make_site_response(
            "biobank-a",
            request_id,
            count=99,
        ),
    )

    result = run(
        gateway.execute(
            plan=make_plan(),
            site_ids=["biobank-a"],
        )
    )

    assert len(result.responses) == 1
    assert result.responses[0].payload["count"] == 99


class FakeFlareTransport:
    """Small fake implementing the FlareTransport protocol."""

    def __init__(
        self,
        responses: Mapping[str, SiteResponse | Exception],
    ) -> None:
        self.responses = dict(responses)
        self.calls: list[dict[str, Any]] = []

    async def submit(
        self,
        *,
        site_id: str,
        task_name: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "site_id": site_id,
                "task_name": task_name,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )

        response = self.responses[site_id]

        if isinstance(response, Exception):
            raise response

        # SiteResponse.model_validate accepts an existing SiteResponse.
        return response  # type: ignore[return-value]


def test_transport_gateway_builds_flare_task_payload() -> None:
    transport = FakeFlareTransport(
        {
            "biobank-a": make_site_response(
                "biobank-a",
                "request-transport",
            )
        }
    )

    gateway = TransportFederatedGateway(
        transport=transport,
        task_name="federated_analysis",
        timeout_seconds=30.0,
    )

    result = run(
        gateway.execute(
            plan=make_plan(),
            site_ids=["biobank-a"],
            request_id="request-transport",
        )
    )

    assert result.successful_site_ids == ("biobank-a",)
    assert len(transport.calls) == 1

    call = transport.calls[0]

    assert call["site_id"] == "biobank-a"
    assert call["task_name"] == "federated_analysis"
    assert call["timeout_seconds"] == 30.0

    assert call["payload"]["protocol_version"] == "1.0"
    assert call["payload"]["request_id"] == "request-transport"
    assert call["payload"]["plan"]["operation"] == (
        Operation.COHORT_COUNT.value
    )


def test_transport_gateway_records_connection_failure() -> None:
    transport = FakeFlareTransport(
        {
            "biobank-a": make_site_response("biobank-a"),
            "biobank-b": ConnectionError("FLARE client unavailable"),
        }
    )

    gateway = TransportFederatedGateway(
        transport=transport,
        fail_on_partial_error=False,
    )

    result = run(
        gateway.execute(
            plan=make_plan(),
            site_ids=["biobank-a", "biobank-b"],
        )
    )

    assert result.successful_site_ids == ("biobank-a",)
    assert result.failed_site_ids == ("biobank-b",)

    failure = result.failures[0]

    assert failure.error_type == "ConnectionError"
    assert failure.retryable is True


def test_gateway_constructor_validation() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        MockFederatedGateway(timeout_seconds=0)

    with pytest.raises(ValueError, match="cannot be negative"):
        MockFederatedGateway(simulated_latency_seconds=-1)

    transport = FakeFlareTransport({})

    with pytest.raises(ValueError, match="greater than zero"):
        TransportFederatedGateway(
            transport=transport,
            timeout_seconds=0,
        )

    with pytest.raises(ValueError, match="at least 1"):
        TransportFederatedGateway(
            transport=transport,
            max_concurrency=0,
        )