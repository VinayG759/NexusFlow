"""Task executor for NexusFlow.

Routes tasks from an orchestrator plan to the correct agent, respects
``depends_on`` ordering, and accumulates results into a shared context dict
that later tasks can read.

Usage::

    from src.utils.task_executor import task_executor

    # Execute a single task
    result = await task_executor.execute_task(
        task={"task_id": "t1", "description": "Research FastAPI", "assigned_agent": "ResearchAgent", "depends_on": []},
        context={},
    )

    # Execute a full plan from orchestrator.orchestrate()
    outcome = await task_executor.execute_plan(tasks, problem_statement="Build a REST API")
"""

import asyncio
import re

from src.agents.api_agent import api_agent
from src.agents.builder_agent import builder_agent
from src.agents.deploy_agent import deploy_agent
from src.agents.file_agent import file_agent
from src.agents.research_agent import research_agent
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Keywords used to pick a DeployAgent method when no explicit method is named.
_DEPS_KEYWORDS = {"dependencies", "requirements", "packages", "install"}
_TEST_KEYWORDS = {"test", "pytest", "spec", "verify", "validate"}


def _derive_filepath(description: str) -> str:
    """Derive a save path for generated code based on the task description."""
    words = description.lower().split()
    slug = "_".join(re.sub(r"[^a-z0-9]", "", w) for w in words[:5] if w)
    slug = slug or "generated"

    desc_lower = description.lower()
    if any(kw in desc_lower for kw in ("frontend", "react", "typescript")):
        return f"weather-app/frontend/src/{slug}.tsx"
    if any(kw in desc_lower for kw in ("database", "schema", "postgresql")):
        return f"weather-app/backend/database/{slug}.py"
    if any(kw in desc_lower for kw in ("api", "endpoint", "route")):
        return f"weather-app/backend/routes/{slug}.py"
    return f"weather-app/backend/{slug}.py"


class TaskExecutor:
    """Routes individual tasks to the appropriate agent and executes full plans.

    Maintains a registry of agent singletons keyed by the same names the
    Orchestrator assigns via ``assigned_agent`` in the task plan, so routing is
    a simple dict lookup.

    Attributes:
        _agents: Mapping of agent name strings to their singleton instances.
    """

    def __init__(self) -> None:
        """Initialise the TaskExecutor with the full agent registry."""
        self._agents = {
            "ResearchAgent": research_agent,
            "BuilderAgent": builder_agent,
            "FileAgent": file_agent,
            "APIAgent": api_agent,
            "DeployAgent": deploy_agent,
        }
        logger.info("TaskExecutor initialised with agents: %s", list(self._agents))

    # ── Internal routing ──────────────────────────────────────────────────────

    async def _dispatch(self, task: dict, context: dict) -> dict:
        """Route a single task to the appropriate agent method.

        Args:
            task: Task dict with ``task_id``, ``description``, ``assigned_agent``,
                and ``depends_on`` keys.
            context: Results from previously completed tasks keyed by ``task_id``.

        Returns:
            The raw result dict returned by the called agent method.

        Raises:
            ValueError: If ``assigned_agent`` is not in the registry.
        """
        agent_name = task.get("assigned_agent", "")
        description = task.get("description", "")

        if agent_name not in self._agents:
            raise ValueError(
                f"Unknown agent '{agent_name}'. "
                f"Registered agents: {list(self._agents)}."
            )

        if agent_name == "ResearchAgent":
            return research_agent.research(description)

        if agent_name == "BuilderAgent":
            await asyncio.sleep(2)
            result = await builder_agent.generate_code(description)
            if result.get("status") == "success":
                filepath = _derive_filepath(description)
                logger.info("Saving generated code to %r", filepath)
                save_result = file_agent.create_project_file(filepath, result["code"])
                if save_result.get("status") == "success":
                    result["saved_to"] = filepath
                else:
                    logger.warning("Failed to save code to %r: %s", filepath, save_result.get("error"))
            return result

        if agent_name == "FileAgent":
            # Use the first completed context result as file content if available.
            content = ""
            if context:
                last_result = next(reversed(context.values()), {})
                content = (
                    last_result.get("result", {}).get("code")
                    or last_result.get("result", {}).get("summary")
                    or last_result.get("result", {}).get("content")
                    or ""
                )
            # Derive a sanitised filename from the task description.
            filename = description.lower().replace(" ", "_")[:40].rstrip("_") + ".txt"
            return file_agent.create_project_file(f"output/{filename}", content or description)

        if agent_name == "APIAgent":
            await asyncio.sleep(2)
            return await api_agent.query_llm(description)

        if agent_name == "DeployAgent":
            desc_lower = description.lower()
            words = set(desc_lower.split())
            import os
            if "frontend" in desc_lower and os.path.isdir("weather-app/frontend"):
                project_path = "weather-app/frontend"
            elif os.path.isdir("weather-app/backend"):
                project_path = "weather-app/backend"
            else:
                project_path = "weather-app"
            if words & _TEST_KEYWORDS:
                return deploy_agent.run_tests(project_path)
            if words & _DEPS_KEYWORDS:
                return deploy_agent.check_dependencies(project_path)
            return deploy_agent.build_project(project_path)

        # Should never reach here given the registry check above.
        raise ValueError(f"No dispatch handler for agent '{agent_name}'.")  # pragma: no cover

    # ── Public methods ────────────────────────────────────────────────────────

    async def execute_task(self, task: dict, context: dict) -> dict:
        """Execute a single task by routing it to the assigned agent.

        Args:
            task: Task dict produced by the Orchestrator, containing:
                - ``task_id`` (str): unique identifier.
                - ``description`` (str): plain-language work description.
                - ``assigned_agent`` (str): name of the agent to invoke.
                - ``depends_on`` (list[str]): task IDs this task waits for
                  (ordering is the caller's responsibility).
            context: Dict of ``{task_id: execute_task result}`` for all
                previously completed tasks. Passed to ``_dispatch`` so agents
                can consume upstream outputs.

        Returns:
            ::

                {
                    "task_id": str,
                    "agent":   str,
                    "status":  "success" | "error",
                    "result":  dict,   # agent's return value on success
                    "error":   str,    # error message on failure, else None
                }
        """
        task_id = task.get("task_id", "unknown")
        agent_name = task.get("assigned_agent", "unknown")
        logger.info("Executing task_id=%r agent=%r", task_id, agent_name)

        try:
            result = await self._dispatch(task, context)
            status = result.get("status", "success")
            error = result.get("error") if status == "error" else None
            logger.info("task_id=%r completed — status=%s", task_id, status)
            return {
                "task_id": task_id,
                "agent": agent_name,
                "status": status,
                "result": result,
                "error": error,
            }
        except Exception as exc:
            logger.exception("task_id=%r failed: %s", task_id, exc)
            return {
                "task_id": task_id,
                "agent": agent_name,
                "status": "error",
                "result": {},
                "error": str(exc),
            }

    async def execute_plan(self, tasks: list, problem_statement: str) -> dict:
        """Execute all tasks in a plan, respecting ``depends_on`` ordering.

        Iterates through *tasks* in the order returned by the Orchestrator.
        Before executing each task all of its dependencies must already appear
        in the completed results — tasks whose deps are not yet met are skipped
        on the first pass and retried until no forward progress is possible.

        Args:
            tasks: Ordered list of task dicts from ``orchestrator.orchestrate()``.
            problem_statement: The original problem, included in the return value
                for traceability.

        Returns:
            ::

                {
                    "status":          "success" | "partial" | "error",
                    "problem":         str,
                    "completed_tasks": list[str],   # task_ids that succeeded
                    "failed_tasks":    list[str],   # task_ids that failed
                    "results":         dict,         # {task_id: execute_task result}
                }

            ``status`` is ``"success"`` when every task completed, ``"partial"``
            when at least one succeeded but others failed or were blocked, and
            ``"error"`` when nothing completed.
        """
        logger.info(
            "Executing plan with %d task(s) for problem=%r", len(tasks), problem_statement,
        )

        context: dict = {}           # task_id → execute_task result
        pending: list = list(tasks)  # tasks not yet executed
        completed: list[str] = []
        failed: list[str] = []

        # Keep making passes until no task can be advanced.
        while pending:
            advanced = False
            still_pending: list = []

            for task in pending:
                task_id = task.get("task_id", "unknown")
                depends_on: list = task.get("depends_on") or []

                # Check all dependencies are completed (not just present in context).
                unmet = [dep for dep in depends_on if dep not in completed]
                if unmet:
                    logger.debug("task_id=%r waiting on deps: %s", task_id, unmet)
                    still_pending.append(task)
                    continue

                outcome = await self.execute_task(task, context)
                context[task_id] = outcome

                if outcome["status"] == "success":
                    completed.append(task_id)
                else:
                    failed.append(task_id)
                    logger.warning("task_id=%r failed — skipping dependents.", task_id)
                    # Mark any tasks that depend on this one as permanently blocked.
                    blocked = {
                        t.get("task_id")
                        for t in still_pending
                        if task_id in (t.get("depends_on") or [])
                    }
                    still_pending = [
                        t for t in still_pending if t.get("task_id") not in blocked
                    ]
                    for b in blocked:
                        failed.append(b)
                        logger.warning("task_id=%r blocked by failed dep %r.", b, task_id)

                advanced = True

            pending = still_pending

            if not advanced and pending:
                # Circular or unresolvable dependencies — abort remaining tasks.
                blocked_ids = [t.get("task_id") for t in pending]
                logger.error("Unresolvable dependencies for tasks: %s", blocked_ids)
                failed.extend(blocked_ids)
                break

        if failed and not completed:
            overall_status = "error"
        elif failed:
            overall_status = "partial"
        else:
            overall_status = "success"

        logger.info(
            "Plan finished — status=%s completed=%d failed=%d",
            overall_status, len(completed), len(failed),
        )
        return {
            "status": overall_status,
            "problem": problem_statement,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "results": context,
        }


# Module-level singleton — import this directly instead of instantiating TaskExecutor yourself.
task_executor = TaskExecutor()
