"""Session state for the conversational global agent.

A session remembers privacy-safe workflow state between user requests:

- discovered federated catalogue;
- selected dataset;
- pending variable mappings;
- last analysis plan;
- last federated result;
- downloadable artifact references.

The session must never store:

- patient-level records;
- raw database rows;
- direct identifiers;
- model-training batches;
- unapproved local data.

The initial implementation uses an in-memory store. It can later be
replaced with Redis or another persistent store without changing the
global-agent coordinator.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from global_agent.exceptions import (
    SessionConflictError,
    SessionExpiredError,
    SessionNotFoundError,
)
from shared.schemas import (
    AnalysisPlan,
    ArtifactReference,
    FederatedCatalogue,
    FederatedResult,
    VariableMapping,
)


# =====================================================================
# Time helper
# =====================================================================


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""

    return datetime.now(timezone.utc)


# =====================================================================
# Session model
# =====================================================================


class AgentSession(BaseModel):
    """Privacy-safe conversational state for one user session."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    session_id: UUID = Field(default_factory=uuid4)

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    updated_at: datetime = Field(
        default_factory=utc_now,
    )

    revision: int = Field(default=0, ge=0)

    catalogue: FederatedCatalogue | None = None

    selected_dataset_id: str | None = None

    pending_mappings: list[VariableMapping] = Field(
        default_factory=list,
    )

    last_plan: AnalysisPlan | None = None

    last_result: FederatedResult | None = None

    artifacts: list[ArtifactReference] = Field(
        default_factory=list,
    )

    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_session_state(
        self,
    ) -> "AgentSession":
        """Validate relationships among session fields."""

        if (
            self.selected_dataset_id is not None
            and self.catalogue is None
        ):
            raise ValueError(
                "A dataset cannot be selected before "
                "a catalogue has been discovered"
            )

        if (
            self.last_plan is not None
            and self.last_result is not None
            and self.last_plan.request_id
            != self.last_result.request_id
        ):
            raise ValueError(
                "last_plan and last_result must use "
                "the same request_id"
            )

        return self

    def with_catalogue(
        self,
        catalogue: FederatedCatalogue,
    ) -> "AgentSession":
        """Return a session updated with a new catalogue.

        A previously selected dataset is retained only if it still
        exists in the new catalogue.
        """

        selected_dataset = (
            self.selected_dataset_id
        )

        available_datasets = {
            dataset.dataset_id
            for site_catalogue
            in catalogue.site_catalogues
            for dataset
            in site_catalogue.datasets
        }

        if (
            selected_dataset is not None
            and selected_dataset
            not in available_datasets
        ):
            selected_dataset = None

        return self._updated_copy(
            catalogue=catalogue,
            selected_dataset_id=selected_dataset,
            pending_mappings=[
                mapping
                for mapping
                in catalogue.variable_mappings
                if any(
                    not site_column.confirmed
                    for site_column
                    in mapping.site_columns
                )
            ],
        )

    def with_selected_dataset(
        self,
        dataset_id: str,
    ) -> "AgentSession":
        """Return a session with a selected dataset."""

        if self.catalogue is None:
            raise ValueError(
                "A catalogue must be discovered before "
                "selecting a dataset"
            )

        normalized_dataset = (
            dataset_id.strip().lower()
        )

        if not normalized_dataset:
            raise ValueError(
                "dataset_id cannot be empty"
            )

        available_datasets = {
            dataset.dataset_id
            for site_catalogue
            in self.catalogue.site_catalogues
            for dataset
            in site_catalogue.datasets
        }

        if (
            normalized_dataset
            not in available_datasets
        ):
            raise ValueError(
                f"Dataset '{normalized_dataset}' is not "
                "available in the session catalogue"
            )

        return self._updated_copy(
            selected_dataset_id=(
                normalized_dataset
            ),
        )

    def with_pending_mappings(
        self,
        mappings: list[VariableMapping],
    ) -> "AgentSession":
        """Return a session with pending mapping suggestions."""

        unconfirmed = [
            mapping
            for mapping in mappings
            if any(
                not site_column.confirmed
                for site_column
                in mapping.site_columns
            )
        ]

        return self._updated_copy(
            pending_mappings=unconfirmed,
        )

    def with_plan(
        self,
        plan: AnalysisPlan,
    ) -> "AgentSession":
        """Record the most recent analysis plan."""

        return self._updated_copy(
            last_plan=plan,
            last_result=None,
            artifacts=[],
        )

    def with_result(
        self,
        result: FederatedResult,
    ) -> "AgentSession":
        """Record a result corresponding to the last plan."""

        if self.last_plan is None:
            raise ValueError(
                "An analysis plan must be recorded before "
                "recording its result"
            )

        if (
            result.request_id
            != self.last_plan.request_id
        ):
            raise ValueError(
                "Federated result request_id does not match "
                "the last analysis plan"
            )

        if (
            result.operation
            != self.last_plan.operation
        ):
            raise ValueError(
                "Federated result operation does not match "
                "the last analysis plan"
            )

        return self._updated_copy(
            last_result=result,
            artifacts=list(result.artifacts),
        )

    def add_warning(
        self,
        warning: str,
    ) -> "AgentSession":
        """Add a unique privacy-safe warning."""

        normalized_warning = warning.strip()

        if not normalized_warning:
            return self

        warnings = list(self.warnings)

        if normalized_warning not in warnings:
            warnings.append(normalized_warning)

        return self._updated_copy(
            warnings=warnings,
        )

    def clear_analysis_state(
        self,
    ) -> "AgentSession":
        """Clear the last analysis while retaining the catalogue."""

        return self._updated_copy(
            last_plan=None,
            last_result=None,
            artifacts=[],
        )

    def _updated_copy(
        self,
        **updates,
    ) -> "AgentSession":
        """Create and fully validate an updated session."""

        payload = self.model_dump(mode="python")

        payload.update(updates)

        payload["updated_at"] = utc_now()

        # The store owns revision increments. Session methods preserve
        # the current revision until save() is called.
        payload["revision"] = self.revision

        return AgentSession.model_validate(
            payload
        )


# =====================================================================
# Abstract session store
# =====================================================================


class SessionStore(ABC):
    """Storage interface used by the global-agent coordinator."""

    @abstractmethod
    async def create(
        self,
    ) -> AgentSession:
        """Create and persist a new session."""
        raise NotImplementedError

    @abstractmethod
    async def get(
        self,
        session_id: UUID | str,
    ) -> AgentSession:
        """Retrieve an existing session."""
        raise NotImplementedError

    @abstractmethod
    async def save(
        self,
        session: AgentSession,
        *,
        expected_revision: int | None = None,
    ) -> AgentSession:
        """Persist an updated session."""
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        session_id: UUID | str,
    ) -> None:
        """Delete one session."""
        raise NotImplementedError

    @abstractmethod
    async def exists(
        self,
        session_id: UUID | str,
    ) -> bool:
        """Return whether a non-expired session exists."""
        raise NotImplementedError


# =====================================================================
# In-memory store
# =====================================================================


class InMemorySessionStore(SessionStore):
    """Concurrency-safe in-memory session storage.

    This implementation is suitable for:

    - unit tests;
    - local development;
    - a single-process hackathon demonstration.

    It is not appropriate for multiple API worker processes because
    each worker would have a separate in-memory dictionary.
    """

    def __init__(
        self,
        *,
        ttl: timedelta = timedelta(hours=2),
        maximum_sessions: int = 1000,
    ) -> None:
        if ttl.total_seconds() <= 0:
            raise ValueError(
                "Session TTL must be positive"
            )

        if maximum_sessions < 1:
            raise ValueError(
                "maximum_sessions must be at least 1"
            )

        self.ttl = ttl
        self.maximum_sessions = maximum_sessions

        self._sessions: dict[
            UUID,
            AgentSession,
        ] = {}

        self._lock = asyncio.Lock()

    async def create(
        self,
    ) -> AgentSession:
        """Create and store a new session."""

        async with self._lock:
            self._remove_expired_locked()

            if (
                len(self._sessions)
                >= self.maximum_sessions
            ):
                self._remove_oldest_locked()

            session = AgentSession()

            self._sessions[
                session.session_id
            ] = self._clone(session)

            return self._clone(session)

    async def get(
        self,
        session_id: UUID | str,
    ) -> AgentSession:
        """Retrieve a non-expired session."""

        normalized_id = self._parse_session_id(
            session_id
        )

        async with self._lock:
            session = self._sessions.get(
                normalized_id
            )

            if session is None:
                raise SessionNotFoundError(
                    f"Session '{normalized_id}' "
                    "was not found.",
                    details={
                        "session_id":
                            str(normalized_id),
                    },
                )

            if self._is_expired(session):
                del self._sessions[
                    normalized_id
                ]

                raise SessionExpiredError(
                    f"Session '{normalized_id}' "
                    "has expired.",
                    details={
                        "session_id":
                            str(normalized_id),
                    },
                )

            return self._clone(session)

    async def save(
        self,
        session: AgentSession,
        *,
        expected_revision: int | None = None,
    ) -> AgentSession:
        """Persist a session using optimistic concurrency control."""

        async with self._lock:
            current = self._sessions.get(
                session.session_id
            )

            if current is None:
                raise SessionNotFoundError(
                    f"Session '{session.session_id}' "
                    "was not found.",
                    details={
                        "session_id":
                            str(session.session_id),
                    },
                )

            if self._is_expired(current):
                del self._sessions[
                    session.session_id
                ]

                raise SessionExpiredError(
                    f"Session '{session.session_id}' "
                    "has expired.",
                    details={
                        "session_id":
                            str(session.session_id),
                    },
                )

            required_revision = (
                session.revision
                if expected_revision is None
                else expected_revision
            )

            if (
                current.revision
                != required_revision
            ):
                raise SessionConflictError(
                    "The session was modified by another "
                    "request.",
                    details={
                        "session_id":
                            str(session.session_id),
                        "expected_revision":
                            required_revision,
                        "current_revision":
                            current.revision,
                    },
                )

            payload = session.model_dump(
                mode="python"
            )

            payload["revision"] = (
                current.revision + 1
            )

            payload["updated_at"] = utc_now()

            saved = AgentSession.model_validate(
                payload
            )

            self._sessions[
                saved.session_id
            ] = self._clone(saved)

            return self._clone(saved)

    async def delete(
        self,
        session_id: UUID | str,
    ) -> None:
        """Delete one session."""

        normalized_id = self._parse_session_id(
            session_id
        )

        async with self._lock:
            if normalized_id not in self._sessions:
                raise SessionNotFoundError(
                    f"Session '{normalized_id}' "
                    "was not found.",
                    details={
                        "session_id":
                            str(normalized_id),
                    },
                )

            del self._sessions[normalized_id]

    async def exists(
        self,
        session_id: UUID | str,
    ) -> bool:
        """Return whether a session exists and has not expired."""

        try:
            normalized_id = (
                self._parse_session_id(
                    session_id
                )
            )
        except ValueError:
            return False

        async with self._lock:
            session = self._sessions.get(
                normalized_id
            )

            if session is None:
                return False

            if self._is_expired(session):
                del self._sessions[
                    normalized_id
                ]

                return False

            return True

    async def clear_expired(
        self,
    ) -> int:
        """Remove expired sessions and return the number removed."""

        async with self._lock:
            before = len(self._sessions)

            self._remove_expired_locked()

            return before - len(
                self._sessions
            )

    async def count(
        self,
    ) -> int:
        """Return the number of active sessions."""

        async with self._lock:
            self._remove_expired_locked()

            return len(self._sessions)

    # =================================================================
    # Internal helpers
    # =================================================================

    def _is_expired(
        self,
        session: AgentSession,
    ) -> bool:
        return (
            utc_now() - session.updated_at
            > self.ttl
        )

    def _remove_expired_locked(
        self,
    ) -> None:
        expired_ids = [
            session_id
            for session_id, session
            in self._sessions.items()
            if self._is_expired(session)
        ]

        for session_id in expired_ids:
            del self._sessions[session_id]

    def _remove_oldest_locked(
        self,
    ) -> None:
        if not self._sessions:
            return

        oldest_id = min(
            self._sessions,
            key=lambda session_id:
                self._sessions[
                    session_id
                ].updated_at,
        )

        del self._sessions[oldest_id]

    @staticmethod
    def _parse_session_id(
        session_id: UUID | str,
    ) -> UUID:
        if isinstance(session_id, UUID):
            return session_id

        try:
            return UUID(str(session_id))

        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid session ID: {session_id}"
            ) from error

    @staticmethod
    def _clone(
        session: AgentSession,
    ) -> AgentSession:
        """Return a deep validated copy."""

        return AgentSession.model_validate(
            session.model_dump(
                mode="python"
            )
        )