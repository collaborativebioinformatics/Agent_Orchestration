"""Global-agent workflow coordinator.

The coordinator connects the main global-agent components:

    user query
        -> planner
        -> policy validation
        -> federated execution gateway
        -> explainer
        -> chat response

It does not:

- interpret natural language itself;
- implement privacy rules;
- access local biobank data;
- calculate statistics;
- communicate directly with individual biobanks;
- implement NVIDIA FLARE workflows.
"""

from __future__ import annotations

from typing import Protocol

from shared.schemas import (
    AnalysisPlan,
    ChatRequest,
    ChatResponse,
    FederatedResult,
)


# ---------------------------------------------------------------------
# Dependency interfaces
# ---------------------------------------------------------------------


class PlannerProtocol(Protocol):
    """Interface required from a planner implementation."""

    async def create_plan(
        self,
        query: str,
    ) -> AnalysisPlan:
        """Convert a natural-language query into an analysis plan."""
        ...


class PolicyProtocol(Protocol):
    """Interface required from the policy validator."""

    def validate(
        self,
        plan: AnalysisPlan,
    ) -> None:
        """Raise an exception if the plan is not permitted."""
        ...


class FederatedGatewayProtocol(Protocol):
    """Interface required from a federated execution gateway."""

    async def execute(
        self,
        plan: AnalysisPlan,
    ) -> FederatedResult:
        """Execute an approved plan through the federation."""
        ...


class ExplainerProtocol(Protocol):
    """Interface required from a result explainer."""

    async def explain(
        self,
        plan: AnalysisPlan,
        result: FederatedResult,
    ) -> str:
        """Create a user-facing explanation of aggregate results."""
        ...


# ---------------------------------------------------------------------
# Global-agent coordinator
# ---------------------------------------------------------------------


class GlobalAgent:
    """Coordinate one complete federated-analysis request.

    Parameters
    ----------
    planner:
        Converts the user's natural-language question into an
        ``AnalysisPlan``.

    policy:
        Applies deterministic governance and privacy rules to the
        proposed plan.

    gateway:
        Sends an approved plan to a mock federation or NVIDIA FLARE
        workflow and returns the aggregate result.

    explainer:
        Converts the approved plan and aggregate result into a
        user-facing explanation.
    """

    def __init__(
        self,
        planner: PlannerProtocol,
        policy: PolicyProtocol,
        gateway: FederatedGatewayProtocol,
        explainer: ExplainerProtocol,
    ) -> None:
        self._planner = planner
        self._policy = policy
        self._gateway = gateway
        self._explainer = explainer

    async def handle(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        """Process one user request through the complete workflow.

        Exceptions are intentionally not converted into API responses
        here. Domain exceptions should be handled later by the FastAPI
        layer, where they can be mapped to appropriate HTTP responses.

        Parameters
        ----------
        request:
            Validated natural-language chat request.

        Returns
        -------
        ChatResponse
            The validated plan, federated result, and explanation.

        Raises
        ------
        PlanningError
            If the planner cannot interpret the question.

        PolicyViolation
            If the proposed analysis is not permitted.

        FederatedExecutionError
            If federated execution fails.

        ExplanationError
            If the result cannot be explained.
        """

        # Stage 1: Convert natural language into a structured plan.
        plan = await self._planner.create_plan(request.query)

        # Stage 2: Apply deterministic governance and privacy checks.
        #
        # The policy validator returns nothing when the plan is valid
        # and raises PolicyViolation when it is not valid.
        self._policy.validate(plan)

        # Stage 3: Execute the approved plan through the configured
        # gateway. During unit testing this can be a mock gateway.
        # During the final demonstration it can be an NVFlareGateway.
        federated_result = await self._gateway.execute(plan)

        # Stage 4: Perform a defensive consistency check.
        self._validate_execution_result(
            plan=plan,
            result=federated_result,
        )

        # Stage 5: Generate a user-facing explanation using only the
        # plan and privacy-approved aggregate results.
        answer = await self._explainer.explain(
            plan=plan,
            result=federated_result,
        )

        # Stage 6: Return the complete traceable response.
        return ChatResponse(
            request_id=plan.request_id,
            plan=plan,
            federated_result=federated_result,
            answer=answer,
            clarification_required=False,
            clarification_question=None,
        )

    async def handle_query(
        self,
        query: str,
    ) -> ChatResponse:
        """Convenience method for callers that have a plain string.

        FastAPI will normally construct ``ChatRequest`` automatically,
        but this method is convenient for command-line demonstrations
        and integration tests.
        """

        request = ChatRequest(query=query)
        return await self.handle(request)

    @staticmethod
    def _validate_execution_result(
        plan: AnalysisPlan,
        result: FederatedResult,
    ) -> None:
        """Check that the result belongs to the submitted plan.

        Pydantic validates the structures independently. This method
        validates their relationship.
        """

        if result.request_id != plan.request_id:
            raise ValueError(
                "Federated result request_id does not match "
                "the analysis plan request_id"
            )

        if result.operation != plan.operation:
            raise ValueError(
                "Federated result operation does not match "
                "the analysis plan operation"
            )


#######Test##########

# ---------------------------------------------------------------------
# Manual smoke-test components
# ---------------------------------------------------------------------


class _DemoPlanner:
    """Small planner used only when this file is run directly."""

    async def create_plan(
        self,
        query: str,
    ) -> AnalysisPlan:
        from shared.schemas import (
            AggregationMethod,
            FilterCondition,
            Operation,
        )

        print("1. Planner received:", query)

        return AnalysisPlan(
            operation=Operation.COHORT_COUNT,
            variables=["bmi"],
            filters=[
                FilterCondition(
                    variable="age",
                    operator="between",
                    value=[60, 80],
                )
            ],
            minimum_cell_size=10,
            aggregation_method=AggregationMethod.SUM,
        )


class _DemoPolicy:
    """Small policy implementation for the manual test."""

    def validate(
        self,
        plan: AnalysisPlan,
    ) -> None:
        print("2. Policy validated:", plan.operation.value)

        if plan.minimum_cell_size < 10:
            raise ValueError(
                "Minimum cell size cannot be below 10"
            )


class _DemoGateway:
    """Simulate a federated result from three biobanks."""

    async def execute(
        self,
        plan: AnalysisPlan,
    ) -> FederatedResult:
        from shared.schemas import (
            AggregationMethod,
            PrivacyMetadata,
            ResponseStatus,
            SiteResponse,
        )

        print("3. Gateway executing request:", plan.request_id)

        site_counts = {
            "biobank-a": 120,
            "biobank-b": 180,
            "biobank-c": 210,
        }

        site_responses = []

        for site_id, count in site_counts.items():
            site_responses.append(
                SiteResponse(
                    request_id=plan.request_id,
                    site_id=site_id,
                    status=ResponseStatus.SUCCESS,
                    result={
                        "eligible_n": count,
                    },
                    privacy=PrivacyMetadata(
                        minimum_cell_size=(
                            plan.minimum_cell_size
                        ),
                        suppressed=False,
                        disclosure_checks_passed=True,
                    ),
                )
            )

        total = sum(site_counts.values())

        return FederatedResult(
            request_id=plan.request_id,
            operation=plan.operation,
            aggregation_method=AggregationMethod.SUM,
            participating_sites=list(site_counts),
            unavailable_sites=[],
            suppressed_sites=[],
            result={
                "eligible_n": total,
            },
            site_results=site_responses,
            warnings=[],
        )


class _DemoExplainer:
    """Generate a deterministic explanation for the manual test."""

    async def explain(
        self,
        plan: AnalysisPlan,
        result: FederatedResult,
    ) -> str:
        print("4. Explainer received aggregate result")

        total = result.result["eligible_n"]
        number_of_sites = len(result.participating_sites)

        return (
            f"Across {number_of_sites} participating biobanks, "
            f"{total} participants aged 60–80 had available "
            f"BMI data."
        )


# ---------------------------------------------------------------------
# Manual entry point
# ---------------------------------------------------------------------


async def _main() -> None:
    """Run a local coordinator smoke test."""

    agent = GlobalAgent(
        planner=_DemoPlanner(),
        policy=_DemoPolicy(),
        gateway=_DemoGateway(),
        explainer=_DemoExplainer(),
    )

    response = await agent.handle_query(
        "How many participants aged 60–80 have BMI data?"
    )

    print("\n--- Final response ---")
    print(
        response.model_dump_json(
            indent=2,
        )
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())