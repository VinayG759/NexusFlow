"""API Agent for NexusFlow.

Handles all external API interactions and LLM queries on behalf of the
orchestrator. Wraps :class:`~src.tools.api_connector.APIConnectorTool` and
exposes higher-level methods that agents and the orchestrator can call without
knowing the HTTP or LLM API details.

Usage::

    from src.agents.api_agent import api_agent

    data    = await api_agent.fetch_api_data("https://api.example.com/users")
    result  = await api_agent.post_api_data("https://api.example.com/items", {"name": "x"})
    reply   = await api_agent.query_llm("Explain async/await in Python")
    code    = await api_agent.integrate_service("Stripe", "process a payment")
"""

from src.tools.api_connector import APIConnectorTool
from src.utils.logger import get_logger

logger = get_logger(__name__)


class APIAgent:
    """Agent responsible for external API calls and LLM-powered tasks.

    Delegates every network operation to
    :class:`~src.tools.api_connector.APIConnectorTool` and enriches results
    with the agent's name so the orchestrator can attribute outputs uniformly.

    Attributes:
        agent_name: Human-readable identifier included in every result dict.
        _connector: The :class:`~src.tools.api_connector.APIConnectorTool`
            instance used for all HTTP and LLM calls.
    """

    def __init__(self, agent_name: str) -> None:
        """Initialise the APIAgent.

        Args:
            agent_name: Unique name for this agent, included in every result
                dict for traceability.
        """
        self.agent_name = agent_name
        self._connector = APIConnectorTool()
        logger.info("APIAgent '%s' initialised.", agent_name)

    async def fetch_api_data(
        self,
        url: str,
        headers: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """Fetch data from an external REST API via HTTP GET.

        Args:
            url: Full URL of the endpoint to call.
            headers: Optional HTTP headers to include in the request.
            params: Optional query-string parameters.

        Returns:
            On success::

                {
                    "agent":  str,
                    "status": "success",
                    "url":    str,
                    "data":   dict | str,  # parsed JSON or raw text
                }

            On failure::

                {
                    "agent":  str,
                    "status": "error",
                    "url":    str,
                    "error":  str,
                }
        """
        logger.info("[%s] GET %s", self.agent_name, url)
        try:
            result = await self._connector.get(url, headers=headers, params=params)
            logger.info("[%s] GET %s → status=%s", self.agent_name, url, result["status"])
            return {
                "agent": self.agent_name,
                "status": result["status"],
                "url": url,
                "data": result.get("data", {}),
                **( {"error": result["error"]} if result["status"] == "error" and "error" in result else {} ),
            }
        except Exception as exc:
            logger.exception("[%s] fetch_api_data failed for %s: %s", self.agent_name, url, exc)
            return {"agent": self.agent_name, "status": "error", "url": url, "error": str(exc)}

    async def post_api_data(
        self,
        url: str,
        payload: dict,
        headers: dict | None = None,
    ) -> dict:
        """Send data to an external REST API via HTTP POST.

        Args:
            url: Full URL of the endpoint to call.
            payload: Dict serialised as the JSON request body.
            headers: Optional HTTP headers to include in the request.

        Returns:
            On success::

                {
                    "agent":  str,
                    "status": "success",
                    "url":    str,
                    "data":   dict | str,
                }

            On failure::

                {
                    "agent":  str,
                    "status": "error",
                    "url":    str,
                    "error":  str,
                }
        """
        logger.info("[%s] POST %s payload_keys=%s", self.agent_name, url, list(payload.keys()))
        try:
            result = await self._connector.post(url, headers=headers, payload=payload)
            logger.info("[%s] POST %s → status=%s", self.agent_name, url, result["status"])
            return {
                "agent": self.agent_name,
                "status": result["status"],
                "url": url,
                "data": result.get("data", {}),
                **( {"error": result["error"]} if result["status"] == "error" and "error" in result else {} ),
            }
        except Exception as exc:
            logger.exception("[%s] post_api_data failed for %s: %s", self.agent_name, url, exc)
            return {"agent": self.agent_name, "status": "error", "url": url, "error": str(exc)}

    async def query_llm(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> dict:
        """Send a prompt to the Groq LLM and return the response.

        Args:
            prompt: The user message to send to the model.
            system_prompt: Optional system-level instruction that shapes the
                model's behaviour (e.g. ``"You are a security expert."``).

        Returns:
            On success::

                {
                    "agent":   str,
                    "status":  "success",
                    "content": str,   # model's reply text
                    "model":   str,   # model ID used
                    "usage":   dict,  # token counts
                }

            On failure::

                {
                    "agent":  str,
                    "status": "error",
                    "error":  str,
                }
        """
        logger.info("[%s] Querying LLM — prompt_len=%d", self.agent_name, len(prompt))
        try:
            if system_prompt is None:
                system_prompt = (
                    "You are an expert software engineer. Return ONLY raw code using FastAPI for backend "
                    "and React+TypeScript for frontend. Never use Flask or Django. "
                    "No explanations or markdown."
                )
            result = await self._connector.call_groq(prompt=prompt, system_prompt=system_prompt)
            if result["status"] != "success":
                raise RuntimeError(result.get("error", "LLM call returned non-success status."))
            logger.info(
                "[%s] LLM responded — model=%s tokens=%s",
                self.agent_name, result.get("model"), result.get("usage", {}).get("completion_tokens"),
            )
            return {
                "agent": self.agent_name,
                "status": "success",
                "content": result["content"],
                "model": result.get("model", ""),
                "usage": result.get("usage", {}),
            }
        except Exception as exc:
            logger.exception("[%s] query_llm failed: %s", self.agent_name, exc)
            return {"agent": self.agent_name, "status": "error", "error": str(exc)}

    async def integrate_service(
        self,
        service_name: str,
        task_description: str,
    ) -> dict:
        """Research a service and generate integration code for a given task.

        Performs two steps:
        1. Fetches the service's developer docs page to gather API context.
        2. Calls the LLM with that context to produce ready-to-use integration
           code tailored to *task_description*.

        Args:
            service_name: Name of the third-party service to integrate with
                (e.g. ``"Stripe"``, ``"Twilio"``, ``"SendGrid"``).
            task_description: What the integration should accomplish
                (e.g. ``"process a one-time payment"``).

        Returns:
            On success::

                {
                    "agent":            str,
                    "status":           "success",
                    "service":          str,
                    "integration_code": str,   # LLM-generated code snippet
                    "resources":        list,  # search results used as context
                }

            On failure::

                {
                    "agent":   str,
                    "status":  "error",
                    "service": str,
                    "error":   str,
                }
        """
        logger.info(
            "[%s] Integrating service=%r for task=%r", self.agent_name, service_name, task_description,
        )
        try:
            # Step 1 — collect API reference material via web search.
            from src.tools.web_search import WebSearchTool  # local import avoids circular deps
            search_tool = WebSearchTool()
            search_result = search_tool.search(f"{service_name} API documentation Python")
            resources = search_result.get("results", [])
            context_snippets = "\n\n".join(
                f"Source: {r['url']}\n{r['content']}" for r in resources[:3] if r.get("content")
            )

            # Step 2 — ask the LLM to write integration code using that context.
            system_prompt = (
                "You are an expert software engineer specialising in third-party API integrations. "
                "Return ONLY raw Python code with no explanation or markdown formatting."
            )
            user_prompt = (
                f"Write Python code to integrate with the {service_name} API to accomplish:\n"
                f"{task_description}\n\n"
                f"Use the following reference material:\n{context_snippets or '(no reference material found)'}"
            )
            llm_result = await self._connector.call_groq(
                prompt=user_prompt, system_prompt=system_prompt,
            )
            if llm_result["status"] != "success":
                raise RuntimeError(llm_result.get("error", "LLM call failed."))

            logger.info(
                "[%s] Integration code generated for service=%r", self.agent_name, service_name,
            )
            return {
                "agent": self.agent_name,
                "status": "success",
                "service": service_name,
                "integration_code": llm_result["content"],
                "resources": resources,
            }
        except Exception as exc:
            logger.exception(
                "[%s] integrate_service failed for service=%r: %s", self.agent_name, service_name, exc,
            )
            return {
                "agent": self.agent_name,
                "status": "error",
                "service": service_name,
                "error": str(exc),
            }


# Module-level singleton — import this directly instead of instantiating APIAgent yourself.
api_agent = APIAgent("APIAgent")
