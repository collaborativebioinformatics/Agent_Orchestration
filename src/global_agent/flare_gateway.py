# global_agent/flare_gateway.py

from __future__ import annotations

import asyncio
import inspect
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from pydantic import ValidationError

from global_agent.exceptions import (
    FederatedExecutionError,
    NoEligibleSitesError,
)
from shared.schemas import AnalysisPlan, SiteResponse

"""
as a boundary between the global agent and NVFlare. 
The actual NVFlare API is hidden behind a small FlareTransport protocol, so the 
coordinator and tests do not depend directly on NVFlare.
"""

# A local mock handler can be synchronous or asynchronous.
MockSiteHandler = Callable[
    [AnalysisPlan, str],
    SiteResponse | Mapping[str, Any] | Awaitable[SiteResponse | Mapping[str, Any]],
]


@dataclass(frozen=True, slots=True)
class SiteExecutionFailure:
    """A safe description of a failed site request."""

    site_id: str
    error_type: str
    message: str
    retryable: bool = False

    def model_dump(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "error_type": self.error_type,
            "message": self.message,
            "retryable": self.retryable,
        }


@dataclass(frozen=True, slots=True)
class FederatedDispatchResult:
    """
    Result of dispatching one analysis plan to multiple sites.

    This object contains site-level responses. It does not aggregate their
    statistical payloads; aggregation belongs in aggregator.py.
    """

    request_id: str
    responses: tuple[SiteResponse, ...]
    failures: tuple[SiteExecutionFailure, ...] = ()
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def successful_site_ids(self) -> tuple[str, ...]:
        return tuple(response.site_id for response in self.responses)

    @property
    def failed_site_ids(self) -> tuple[str, ...]:
        return tuple(failure.site_id for failure in self.failures)

    @property
    def has_partial_failure(self) -> bool:
        return bool(self.responses and self.failures)

    @property
    def all_sites_failed(self) -> bool:
        return not self.responses and bool(self.failures)


class FlareTransport(Protocol):
    """
    Low-level transport used to submit a task to one NVFlare client.

    A real implementation can wrap an NVFlare controller, job API, message
    bus, or another deployment-specific mechanism.
    """

    async def submit(
        self,
        *,
        site_id: str,
        task_name: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        """
        Submit a task and return its JSON-compatible response payload.

        The returned mapping must be compatible with SiteResponse.
        """
        ...


class FederatedGateway(ABC):
    """Interface used by the global coordinator."""

    @abstractmethod
    async def execute(
        self,
        *,
        plan: AnalysisPlan,
        site_ids: Sequence[str],
        request_id: str | None = None,
    ) -> FederatedDispatchResult:
        """Execute a validated plan at the selected sites."""
        raise NotImplementedError


class TransportFederatedGateway(FederatedGateway):
    """
    Production-oriented gateway using a supplied NVFlare transport.

    It serializes the shared AnalysisPlan contract, executes requests
    concurrently, validates every response, and records site failures.
    """

    def __init__(
        self,
        transport: FlareTransport,
        *,
        task_name: str = "federated_analysis",
        timeout_seconds: float = 120.0,
        max_concurrency: int = 8,
        fail_on_partial_error: bool = False,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")

        self._transport = transport
        self._task_name = task_name
        self._timeout_seconds = timeout_seconds
        self._max_concurrency = max_concurrency
        self._fail_on_partial_error = fail_on_partial_error

    async def execute(
        self,
        *,
        plan: AnalysisPlan,
        site_ids: Sequence[str],
        request_id: str | None = None,
    ) -> FederatedDispatchResult:
        clean_site_ids = _normalize_site_ids(site_ids)

        if not clean_site_ids:
            raise NoEligibleSitesError(
                "No eligible sites were supplied to the FLARE gateway.",
                request_id=request_id,
            )

        execution_id = request_id or str(uuid4())
        started_at = time.monotonic()
        semaphore = asyncio.Semaphore(self._max_concurrency)

        # JSON mode converts enums and other Pydantic types into transport-safe
        # primitive values.
        plan_payload = plan.model_dump(mode="json")

        payload = {
            "protocol_version": "1.0",
            "request_id": execution_id,
            "plan": plan_payload,
        }

        async def run_site(
            site_id: str,
        ) -> SiteResponse | SiteExecutionFailure:
            async with semaphore:
                try:
                    raw_response = await asyncio.wait_for(
                        self._transport.submit(
                            site_id=site_id,
                            task_name=self._task_name,
                            payload=payload,
                            timeout_seconds=self._timeout_seconds,
                        ),
                        timeout=self._timeout_seconds,
                    )

                    response = SiteResponse.model_validate(raw_response)

                    if response.site_id != site_id:
                        raise ValueError(
                            "Response site_id does not match the requested site: "
                            f"expected {site_id!r}, received "
                            f"{response.site_id!r}"
                        )

                    return response

                except asyncio.TimeoutError:
                    return SiteExecutionFailure(
                        site_id=site_id,
                        error_type="timeout",
                        message=(
                            f"Site did not respond within "
                            f"{self._timeout_seconds} seconds."
                        ),
                        retryable=True,
                    )

                except ValidationError as exc:
                    return SiteExecutionFailure(
                        site_id=site_id,
                        error_type="invalid_response",
                        message=_safe_validation_message(exc),
                        retryable=False,
                    )

                except Exception as exc:
                    return SiteExecutionFailure(
                        site_id=site_id,
                        error_type=type(exc).__name__,
                        message=_safe_error_message(exc),
                        retryable=_is_retryable_exception(exc),
                    )

        results = await asyncio.gather(
            *(run_site(site_id) for site_id in clean_site_ids)
        )

        responses = tuple(
            item for item in results if isinstance(item, SiteResponse)
        )
        failures = tuple(
            item
            for item in results
            if isinstance(item, SiteExecutionFailure)
        )

        elapsed_seconds = time.monotonic() - started_at

        if not responses:
            raise FederatedExecutionError(
                "Every selected site failed during federated execution.",
                request_id=execution_id,
                details={
                    "failures": [
                        failure.model_dump() for failure in failures
                    ],
                    "elapsed_seconds": elapsed_seconds,
                },
            )

        if failures and self._fail_on_partial_error:
            raise FederatedExecutionError(
                "One or more sites failed during federated execution.",
                request_id=execution_id,
                details={
                    "successful_sites": [
                        response.site_id for response in responses
                    ],
                    "failures": [
                        failure.model_dump() for failure in failures
                    ],
                    "elapsed_seconds": elapsed_seconds,
                },
            )

        return FederatedDispatchResult(
            request_id=execution_id,
            responses=responses,
            failures=failures,
            elapsed_seconds=elapsed_seconds,
            metadata={
                "task_name": self._task_name,
                "requested_site_count": len(clean_site_ids),
                "successful_site_count": len(responses),
                "failed_site_count": len(failures),
            },
        )


class MockFederatedGateway(FederatedGateway):
    """
    In-process gateway for unit tests and hackathon development.

    Each simulated biobank registers a handler. The handler receives the same
    AnalysisPlan that a real local agent would receive.
    """

    def __init__(
        self,
        handlers: Mapping[str, MockSiteHandler] | None = None,
        *,
        timeout_seconds: float = 10.0,
        simulated_latency_seconds: float = 0.0,
        fail_on_partial_error: bool = False,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        if simulated_latency_seconds < 0:
            raise ValueError(
                "simulated_latency_seconds cannot be negative"
            )

        self._handlers: dict[str, MockSiteHandler] = dict(handlers or {})
        self._timeout_seconds = timeout_seconds
        self._simulated_latency_seconds = simulated_latency_seconds
        self._fail_on_partial_error = fail_on_partial_error

    def register_site(
        self,
        site_id: str,
        handler: MockSiteHandler,
    ) -> None:
        normalized_site_id = site_id.strip()

        if not normalized_site_id:
            raise ValueError("site_id cannot be empty")

        if normalized_site_id in self._handlers:
            raise ValueError(
                f"A handler is already registered for {normalized_site_id!r}"
            )

        self._handlers[normalized_site_id] = handler

    def replace_site(
        self,
        site_id: str,
        handler: MockSiteHandler,
    ) -> None:
        normalized_site_id = site_id.strip()

        if not normalized_site_id:
            raise ValueError("site_id cannot be empty")

        self._handlers[normalized_site_id] = handler

    def unregister_site(self, site_id: str) -> None:
        self._handlers.pop(site_id, None)

    @property
    def registered_site_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))

    async def execute(
        self,
        *,
        plan: AnalysisPlan,
        site_ids: Sequence[str],
        request_id: str | None = None,
    ) -> FederatedDispatchResult:
        clean_site_ids = _normalize_site_ids(site_ids)

        if not clean_site_ids:
            raise NoEligibleSitesError(
                "No eligible sites were supplied to the mock gateway.",
                request_id=request_id,
            )

        execution_id = request_id or str(uuid4())
        started_at = time.monotonic()

        async def run_site(
            site_id: str,
        ) -> SiteResponse | SiteExecutionFailure:
            handler = self._handlers.get(site_id)

            if handler is None:
                return SiteExecutionFailure(
                    site_id=site_id,
                    error_type="site_not_registered",
                    message=f"No mock handler is registered for {site_id!r}.",
                    retryable=False,
                )

            try:
                if self._simulated_latency_seconds:
                    await asyncio.sleep(self._simulated_latency_seconds)

                result = handler(plan, execution_id)

                if inspect.isawaitable(result):
                    result = await asyncio.wait_for(
                        result,
                        timeout=self._timeout_seconds,
                    )

                response = (
                    result
                    if isinstance(result, SiteResponse)
                    else SiteResponse.model_validate(result)
                )

                if response.site_id != site_id:
                    raise ValueError(
                        "Mock response site_id does not match its registered "
                        f"site: expected {site_id!r}, received "
                        f"{response.site_id!r}"
                    )

                return response

            except asyncio.TimeoutError:
                return SiteExecutionFailure(
                    site_id=site_id,
                    error_type="timeout",
                    message=(
                        f"Mock site did not respond within "
                        f"{self._timeout_seconds} seconds."
                    ),
                    retryable=True,
                )

            except ValidationError as exc:
                return SiteExecutionFailure(
                    site_id=site_id,
                    error_type="invalid_response",
                    message=_safe_validation_message(exc),
                    retryable=False,
                )

            except Exception as exc:
                return SiteExecutionFailure(
                    site_id=site_id,
                    error_type=type(exc).__name__,
                    message=_safe_error_message(exc),
                    retryable=False,
                )

        results = await asyncio.gather(
            *(run_site(site_id) for site_id in clean_site_ids)
        )

        responses = tuple(
            item for item in results if isinstance(item, SiteResponse)
        )
        failures = tuple(
            item
            for item in results
            if isinstance(item, SiteExecutionFailure)
        )

        elapsed_seconds = time.monotonic() - started_at

        if not responses:
            raise FederatedExecutionError(
                "Every selected mock site failed.",
                request_id=execution_id,
                details={
                    "failures": [
                        failure.model_dump() for failure in failures
                    ]
                },
            )

        if failures and self._fail_on_partial_error:
            raise FederatedExecutionError(
                "One or more mock sites failed.",
                request_id=execution_id,
                details={
                    "successful_sites": [
                        response.site_id for response in responses
                    ],
                    "failures": [
                        failure.model_dump() for failure in failures
                    ],
                },
            )

        return FederatedDispatchResult(
            request_id=execution_id,
            responses=responses,
            failures=failures,
            elapsed_seconds=elapsed_seconds,
            metadata={
                "gateway": "mock",
                "requested_site_count": len(clean_site_ids),
                "successful_site_count": len(responses),
                "failed_site_count": len(failures),
            },
        )


def _normalize_site_ids(site_ids: Sequence[str]) -> tuple[str, ...]:
    """Remove whitespace and duplicates while preserving order."""

    result: list[str] = []
    seen: set[str] = set()

    for raw_site_id in site_ids:
        site_id = raw_site_id.strip()

        if site_id and site_id not in seen:
            result.append(site_id)
            seen.add(site_id)

    return tuple(result)


def _safe_validation_message(exc: ValidationError) -> str:
    """
    Return validation information without copying arbitrary site payloads
    into global logs or error messages.
    """

    locations = [
        ".".join(str(part) for part in error["loc"])
        for error in exc.errors(include_input=False)
    ]

    fields = ", ".join(locations[:5]) or "unknown fields"
    return f"Site response failed schema validation at: {fields}."


def _safe_error_message(exc: Exception) -> str:
    """
    Avoid exposing raw payloads or sensitive records through exception text.
    """

    message = str(exc).strip()

    if not message:
        return "Site execution failed without an error message."

    # Keep failure information reasonably small and log-safe.
    return message[:500]


def _is_retryable_exception(exc: Exception) -> bool:
    return isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        ),
    )