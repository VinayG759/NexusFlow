"""Research Agent for NexusFlow.

Performs web-based research on behalf of the orchestrator. All searches are
delegated to :class:`~src.tools.web_search.WebSearchTool` so the agent itself
stays decoupled from the underlying search API.

Usage::

    from src.agents.research_agent import research_agent

    result = research_agent.research("transformer architecture")
    apis   = research_agent.find_apis("Stripe")
    tools  = research_agent.find_resources("parse PDF files in Python")
"""

from src.tools.web_search import WebSearchTool
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResearchAgent:
    """Agent responsible for gathering information from the web.

    Wraps :class:`~src.tools.web_search.WebSearchTool` and returns structured
    result dicts with a consistent schema so the orchestrator can process
    outputs from any agent uniformly.

    Attributes:
        agent_name: Human-readable identifier used in log messages and result dicts.
        _search_tool: The :class:`~src.tools.web_search.WebSearchTool` instance
            used for all searches.
    """

    def __init__(self, agent_name: str) -> None:
        """Initialise the ResearchAgent.

        Args:
            agent_name: A unique name for this agent instance, included in
                every result dict so the orchestrator can attribute outputs.
        """
        self.agent_name = agent_name
        self._search_tool = WebSearchTool()
        logger.info("ResearchAgent '%s' initialised.", agent_name)

    def research(self, topic: str) -> dict:
        """Research a topic and return findings with a plain-text summary.

        Calls :meth:`~src.tools.web_search.WebSearchTool.search_and_summarize`
        so the returned dict includes both individual results and a combined
        summary of the top three sources.

        Args:
            topic: The subject to research (e.g. ``"BERT embeddings"``).

        Returns:
            On success::

                {
                    "agent":    str,          # agent_name
                    "topic":    str,          # the original topic
                    "findings": list[dict],   # list of {title, url, content, score}
                    "summary":  str,          # combined content from top 3 results
                    "status":   "success",
                }

            On failure::

                {
                    "agent":   str,
                    "topic":   str,
                    "status":  "error",
                    "error":   str,
                }
        """
        logger.info("[%s] Starting research on topic=%r", self.agent_name, topic)
        try:
            result = self._search_tool.search_and_summarize(topic)
            if "error" in result:
                raise RuntimeError(result["error"])
            logger.info(
                "[%s] Research complete for topic=%r — %d finding(s).",
                self.agent_name, topic, result["total_results"],
            )
            return {
                "agent": self.agent_name,
                "topic": topic,
                "findings": result["results"],
                "summary": result["summary"],
                "status": "success",
            }
        except Exception as exc:
            logger.exception("[%s] Research failed for topic=%r: %s", self.agent_name, topic, exc)
            return {
                "agent": self.agent_name,
                "topic": topic,
                "status": "error",
                "error": str(exc),
            }

    def find_apis(self, service_name: str) -> dict:
        """Find API documentation and references for a named service.

        Constructs a targeted query (``"<service_name> API documentation"``)
        and returns raw search results so the caller can inspect endpoints,
        auth schemes, and SDK links.

        Args:
            service_name: The name of the service or platform to look up
                (e.g. ``"Stripe"``, ``"OpenAI"``, ``"GitHub"``).

        Returns:
            On success::

                {
                    "agent":   str,
                    "service": str,           # the original service_name
                    "apis":    list[dict],    # list of {title, url, content, score}
                    "status":  "success",
                }

            On failure::

                {
                    "agent":   str,
                    "service": str,
                    "status":  "error",
                    "error":   str,
                }
        """
        query = f"{service_name} API documentation"
        logger.info("[%s] Searching for API docs — service=%r", self.agent_name, service_name)
        try:
            result = self._search_tool.search(query)
            if "error" in result:
                raise RuntimeError(result["error"])
            logger.info(
                "[%s] find_apis complete for service=%r — %d result(s).",
                self.agent_name, service_name, result["total_results"],
            )
            return {
                "agent": self.agent_name,
                "service": service_name,
                "apis": result["results"],
                "status": "success",
            }
        except Exception as exc:
            logger.exception(
                "[%s] find_apis failed for service=%r: %s", self.agent_name, service_name, exc,
            )
            return {
                "agent": self.agent_name,
                "service": service_name,
                "status": "error",
                "error": str(exc),
            }

    def find_resources(self, task_description: str) -> dict:
        """Find libraries, tools, or examples relevant to a task description.

        Constructs a search query from the task description and returns results
        that the orchestrator or builder agent can use to select dependencies
        and implementation approaches.

        Args:
            task_description: A plain-language description of the task for which
                resources are needed (e.g. ``"parse PDF files in Python"``).

        Returns:
            On success::

                {
                    "agent":     str,
                    "task":      str,          # the original task_description
                    "resources": list[dict],   # list of {title, url, content, score}
                    "status":    "success",
                }

            On failure::

                {
                    "agent":  str,
                    "task":   str,
                    "status": "error",
                    "error":  str,
                }
        """
        logger.info(
            "[%s] Searching for resources — task=%r", self.agent_name, task_description,
        )
        try:
            result = self._search_tool.search(task_description)
            if "error" in result:
                raise RuntimeError(result["error"])
            logger.info(
                "[%s] find_resources complete for task=%r — %d result(s).",
                self.agent_name, task_description, result["total_results"],
            )
            return {
                "agent": self.agent_name,
                "task": task_description,
                "resources": result["results"],
                "status": "success",
            }
        except Exception as exc:
            logger.exception(
                "[%s] find_resources failed for task=%r: %s",
                self.agent_name, task_description, exc,
            )
            return {
                "agent": self.agent_name,
                "task": task_description,
                "status": "error",
                "error": str(exc),
            }


# Module-level singleton — import this directly instead of instantiating ResearchAgent yourself.
research_agent = ResearchAgent("ResearchAgent")
