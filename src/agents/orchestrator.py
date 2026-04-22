"""Orchestrator Agent for NexusFlow.

The master controller of the multi-agent system: decomposes problems into
subtasks via a Groq LLM call, delegates work to specialised agents, monitors
progress, and synthesizes final results.
"""

import json
import uuid
from datetime import datetime, timezone

from src.config.settings import settings
from src.tools.api_connector import api_connector
from src.utils.logger import get_logger

logger = get_logger(__name__)

_VALID_AGENTS = {"ResearchAgent", "BuilderAgent", "FileAgent", "APIAgent", "DeployAgent"}

_ORCHESTRATE_SYSTEM_PROMPT = """You are a project planning AI for a multi-agent software system.
When given a problem statement you must decompose it into discrete subtasks.

Return ONLY a valid JSON array — no explanation, no markdown, no code fences.
Each element must have exactly these keys:
  "task_id"        : a short slug like "task-001"
  "description"    : what the agent must do (one sentence)
  "assigned_agent" : one of ResearchAgent | BuilderAgent | FileAgent | APIAgent | DeployAgent
  "depends_on"     : a JSON array of task_id strings this task must wait for (empty if none)

Example output:
[
  {"task_id": "task-001", "description": "Research best practices", "assigned_agent": "ResearchAgent", "depends_on": []},
  {"task_id": "task-002", "description": "Scaffold FastAPI project", "assigned_agent": "BuilderAgent", "depends_on": ["task-001"]}
]"""


class OrchestratorAgent:
    """Master controller that coordinates all agents in the NexusFlow system.

    Calls the Groq LLM to decompose a problem into a structured task plan, then
    delegates each task to the appropriate specialised agent. Progress monitoring
    and result synthesis are stubs pending database integration.

    Attributes:
        agent_name: Human-readable identifier for this orchestrator instance.
        available_agents: Names of agents this orchestrator may delegate to.
    """

    def __init__(self, agent_name: str, available_agents: list) -> None:
        """Initialise the OrchestratorAgent.

        Args:
            agent_name: Unique name for this orchestrator instance, used in
                logging and task attribution.
            available_agents: List of agent name strings the orchestrator may
                delegate to (e.g. ``["ResearchAgent", "BuilderAgent"]``).
        """
        self.agent_name = agent_name
        self.available_agents = available_agents
        logger.info("OrchestratorAgent '%s' initialised with agents: %s", agent_name, available_agents)

    async def orchestrate(self, problem_statement: str) -> dict:
        """Break down a problem statement into a structured task plan via the Groq LLM.

        Sends the problem to Groq with a strict JSON-only system prompt. The
        response is parsed into a list of task dicts that the caller can iterate
        over and pass to :meth:`delegate_task`.

        Args:
            problem_statement: The high-level goal or question provided by the user.

        Returns:
            On success::

                {
                    "problem": str,
                    "tasks": [
                        {
                            "task_id":        str,
                            "description":    str,
                            "assigned_agent": str,
                            "depends_on":     list[str],
                        },
                        ...
                    ],
                    "status": "success",
                }

            On LLM failure or JSON parse error::

                {
                    "problem": str,
                    "tasks":   [],
                    "status":  "error",
                    "error":   str,
                }
        """
        logger.info("[%s] Orchestrating problem: %r", self.agent_name, problem_statement)

        user_prompt = (
            f"Break down this problem into subtasks for a multi-agent system:\n\n{problem_statement}"
        )

        llm_result = await api_connector.call_groq(
            prompt=user_prompt,
            system_prompt=_ORCHESTRATE_SYSTEM_PROMPT,
        )

        if llm_result["status"] != "success":
            error = llm_result.get("error", "LLM call failed.")
            logger.error("[%s] Groq call failed during orchestrate: %s", self.agent_name, error)
            return {"problem": problem_statement, "tasks": [], "status": "error", "error": error}

        raw_content = llm_result["content"].strip()
        try:
            tasks = json.loads(raw_content)
            if not isinstance(tasks, list):
                raise ValueError("LLM returned JSON but not a list.")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(
                "[%s] Failed to parse LLM task plan: %s — raw: %.200s",
                self.agent_name, exc, raw_content,
            )
            return {
                "problem": problem_statement,
                "tasks": [],
                "status": "error",
                "error": f"JSON parse error: {exc}",
            }

        logger.info("[%s] Orchestration produced %d task(s).", self.agent_name, len(tasks))
        return {"problem": problem_statement, "tasks": tasks, "status": "success"}

    def delegate_task(self, task: str, agent_name: str) -> dict:
        """Assign a task to a specific agent and return a task tracking object.

        Validates that *agent_name* is in :attr:`available_agents`, generates a
        UUID task identifier, and returns a tracking dict the caller can pass to
        :meth:`monitor_progress`.

        Args:
            task: Plain-language description of the work to be done.
            agent_name: Name of the agent to assign the task to. Must be present
                in :attr:`available_agents`.

        Returns:
            On success::

                {
                    "task_id":    str,   # UUID4 string
                    "task":       str,   # original task description
                    "agent":      str,   # assigned agent name
                    "status":     "pending",
                    "created_at": str,   # ISO-8601 UTC timestamp
                }

            On validation failure::

                {
                    "status": "error",
                    "error":  str,
                }
        """
        logger.info("[%s] Delegating task to '%s': %s", self.agent_name, agent_name, task)

        if agent_name not in self.available_agents:
            error = (
                f"Agent '{agent_name}' is not in available_agents: {self.available_agents}."
            )
            logger.error("[%s] %s", self.agent_name, error)
            return {"status": "error", "error": error}

        task_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        logger.info("[%s] Task '%s' created for agent '%s'.", self.agent_name, task_id, agent_name)
        # TODO: Persist the task record to the database via src.database.models.Task.
        # TODO: Dispatch the task to the agent via a message queue or direct call.
        return {
            "task_id": task_id,
            "task": task,
            "agent": agent_name,
            "status": "pending",
            "created_at": created_at,
        }

    def monitor_progress(self, task_id: str) -> dict:
        """Return the current execution status of a delegated task.

        Args:
            task_id: The unique identifier returned by :meth:`delegate_task`.

        Returns:
            A dict with keys ``task_id``, ``status``, ``progress``, ``result``,
            ``error``, and ``updated_at``. Currently returns an empty stub
            pending database integration.
        """
        logger.info("[%s] Monitoring progress for task_id='%s'", self.agent_name, task_id)
        # TODO: Look up task_id in the database or in-memory task store.
        # TODO: If task is running, poll the assigned agent for a live status update.
        # TODO: Handle stale/timed-out tasks and mark them as "failed".
        return {}

    def synthesize_results(self, results: dict) -> str:
        """Combine all agent outputs into a single, coherent final response.

        Args:
            results: Mapping of task_id to the output dict returned by each
                agent, e.g. ``{"task-001": {"answer": "..."}, ...}``.

        Returns:
            A string containing the final synthesised answer. Currently returns
            an empty stub pending LLM synthesis integration.
        """
        logger.info("[%s] Synthesizing %d task result(s).", self.agent_name, len(results))
        # TODO: Format results into a structured prompt for the LLM synthesis step.
        # TODO: Call api_connector.call_groq() to merge and summarise outputs.
        # TODO: Post-process the response (strip artefacts, enforce length limits).
        return ""
