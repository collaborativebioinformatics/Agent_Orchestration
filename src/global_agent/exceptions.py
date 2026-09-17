"""Domain-specific exceptions for the global agent.

These exceptions describe failures occurring during the global-agent
workflow:

    planning
        -> policy validation
        -> federated execution
        -> result validation
        -> explanation

The API layer can later translate these exceptions into HTTP responses.
The coordinator and other domain modules should not depend on FastAPI.
"""

from __future__ import annotations

from typing import Any, Mapping
from uuid import UUID


class GlobalAgentError(Exception):
    """Base exception for all global-agent errors.

    Parameters
    ----------
    message:
        Human-readable description of the failure.

    request_id:
        Optional request identifier used to trace the failure.

    details:
        Optional structured, non-sensitive diagnostic information.

    Notes
    -----
    The ``details`` dictionary must never contain patient-level data.
    """

    error_code = "global_agent_error"

    def __init__(
        self,
        message: str,
        *,
        request_id: UUID | str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)

        self.message = message

        self.request_id = (
            str(request_id)
            if request_id is not None
            else None
        )

        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the error."""

        return {
            "error_code": self.error_code,
            "message": self.message,
            "request_id": self.request_id,
            "details": self.details,
        }


class PlanningError(GlobalAgentError):
    """Raised when a user query cannot be converted into a plan."""

    error_code = "planning_error"


class PolicyViolation(GlobalAgentError):
    """Raised when an analysis plan violates governance rules."""

    error_code = "policy_violation"


class FederatedExecutionError(GlobalAgentError):
    """Raised when federated execution cannot be completed.

    Examples include:

    - FLARE job submission failure;
    - communication failure;
    - workflow timeout;
    - invalid response from the federated runtime.
    """

    error_code = "federated_execution_error"


class NoEligibleSitesError(FederatedExecutionError):
    """Raised when no biobank can execute the analysis plan."""

    error_code = "no_eligible_sites"


class FederatedResultValidationError(
    FederatedExecutionError
):
    """Raised when a federated result does not match its plan.

    Examples include:

    - mismatched request identifiers;
    - mismatched operation;
    - malformed or inconsistent site results.
    """

    error_code = "federated_result_validation_error"


class ExplanationError(GlobalAgentError):
    """Raised when a result cannot be converted into an explanation."""

    error_code = "explanation_error"


class ConfigurationError(GlobalAgentError):
    """Raised when global-agent configuration is invalid."""

    error_code = "configuration_error"


class CatalogueError(GlobalAgentError):
    """Base error for federated catalogue operations."""

    error_code = "catalogue_error"


class UnknownSiteError(CatalogueError):
    """Raised when a requested site is absent from the catalogue."""

    error_code = "unknown_site"


class UnknownDatasetError(CatalogueError):
    """Raised when a requested dataset cannot be found."""

    error_code = "unknown_dataset"


class UnknownColumnError(CatalogueError):
    """Raised when a requested column cannot be resolved."""

    error_code = "unknown_column"


class UnconfirmedMappingError(CatalogueError):
    """Raised when a variable mapping requires confirmation."""

    error_code = "unconfirmed_mapping"


class ColumnPermissionError(CatalogueError):
    """Raised when a column is unavailable for the requested use."""

    error_code = "column_permission_error"

class SessionError(GlobalAgentError):
    """Base error for global-agent session operations."""

    error_code = "session_error"


class SessionNotFoundError(SessionError):
    """Raised when a requested session does not exist."""

    error_code = "session_not_found"


class SessionExpiredError(SessionError):
    """Raised when a requested session has expired."""

    error_code = "session_expired"


class SessionConflictError(SessionError):
    """Raised when concurrent updates conflict."""

    error_code = "session_conflict"