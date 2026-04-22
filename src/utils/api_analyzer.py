"""API Analyzer — identifies external APIs required by a problem statement.

Makes a single LLM call to extract every third-party service a project will
need, enriches results with free-tier information, and can fall back to a web
search to surface free alternatives for paid-only APIs.
"""

import json
import re

from src.tools.api_connector import api_connector
from src.tools.web_search import web_search_tool
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a senior software architect. Analyze the problem statement and "
    "identify ALL external APIs, services or ML models needed to build it. "
    "Return ONLY a valid JSON array. Each item must have exactly these keys: "
    "name (str), purpose (str), free_tier (bool), signup_url (str), "
    "docs_url (str), key_steps (list of max 5 strings), is_paid_only (bool), "
    "free_alternatives (list of objects with name/signup_url/note, only if "
    "is_paid_only is true). "
    "If no external APIs are needed return empty array []."
)


class APIAnalyzer:
    """Identify and describe all external APIs required for a given project.

    Uses the Groq LLM to analyse a free-text problem statement and return a
    structured list of every third-party service the resulting application will
    need, including signup links, setup steps, and free-tier availability.

    Attributes:
        agent_name: Label used in log messages.
    """

    def __init__(self, agent_name: str = "APIAnalyzer") -> None:
        self.agent_name = agent_name
        logger.info("%s initialised.", self.agent_name)

    # ── Public methods ────────────────────────────────────────────────────────

    async def analyze(self, problem_statement: str) -> dict:
        """Identify all external APIs required to build the described project.

        Makes a single LLM call with a structured JSON schema enforced via the
        system prompt, then parses and validates the response.

        Args:
            problem_statement: Free-text description of the project to build.

        Returns:
            On success::

                {
                    "status":       "success",
                    "apis":         list[dict],   # one entry per API/service
                    "total":        int,
                    "has_paid_only": bool,        # True if any api is paid-only
                }

            On failure::

                {
                    "status":  "error",
                    "error":   str,
                    "apis":    [],
                    "total":   0,
                    "has_paid_only": False,
                }
        """
        logger.info(
            "%s.analyze — problem=%r", self.agent_name, problem_statement[:120],
        )

        user_prompt = (
            f"Analyze this problem and list all required external APIs: {problem_statement}"
        )

        llm_result = await api_connector.call_groq(
            prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
        )

        if llm_result.get("status") != "success":
            error_msg = llm_result.get("error", "LLM call failed.")
            logger.error("%s.analyze — LLM error: %s", self.agent_name, error_msg)
            return {"status": "error", "error": error_msg, "apis": [], "total": 0, "has_paid_only": False}

        raw = llm_result.get("content", "")
        logger.debug("%s.analyze — raw LLM response length=%d", self.agent_name, len(raw))

        try:
            apis: list[dict] = json.loads(self._clean_json(raw))
        except json.JSONDecodeError as exc:
            logger.error(
                "%s.analyze — JSON parse failed: %s | raw=%r", self.agent_name, exc, raw[:200],
            )
            return {
                "status": "error",
                "error": f"Failed to parse LLM response as JSON: {exc}",
                "apis": [],
                "total": 0,
                "has_paid_only": False,
            }

        if not isinstance(apis, list):
            logger.error("%s.analyze — expected list, got %s", self.agent_name, type(apis).__name__)
            return {
                "status": "error",
                "error": "LLM returned a non-list JSON value.",
                "apis": [],
                "total": 0,
                "has_paid_only": False,
            }

        has_paid_only = any(api.get("is_paid_only", False) for api in apis)
        logger.info(
            "%s.analyze — found %d API(s), has_paid_only=%s",
            self.agent_name, len(apis), has_paid_only,
        )
        return {
            "status": "success",
            "apis": apis,
            "total": len(apis),
            "has_paid_only": has_paid_only,
        }

    async def search_alternatives(self, api_name: str) -> list[dict]:
        """Search the web for free alternatives to a paid API.

        Args:
            api_name: Name of the API for which to find alternatives
                (e.g. ``"OpenAI GPT-4"``).

        Returns:
            List of up to 3 results, each::

                {
                    "name":        str,   # result title
                    "url":         str,
                    "description": str,   # content snippet
                }

            Returns an empty list if the search fails.
        """
        query = f"free alternative to {api_name} API developer"
        logger.info(
            "%s.search_alternatives — query=%r", self.agent_name, query,
        )

        result = web_search_tool.search(query, max_results=3)

        if result.get("error"):
            logger.warning(
                "%s.search_alternatives — search error: %s", self.agent_name, result["error"],
            )
            return []

        alternatives = [
            {
                "name":        item.get("title", ""),
                "url":         item.get("url", ""),
                "description": item.get("content", ""),
            }
            for item in result.get("results", [])[:3]
        ]
        logger.info(
            "%s.search_alternatives — found %d alternative(s) for %r",
            self.agent_name, len(alternatives), api_name,
        )
        return alternatives

    # ── Private helpers ───────────────────────────────────────────────────────

    def _clean_json(self, raw: str) -> str:
        """Strip markdown code fences from an LLM response.

        Handles both annotated fences (```json … ```) and plain fences
        (``` … ```).

        Args:
            raw: Raw string from the LLM, possibly wrapped in markdown fences.

        Returns:
            The inner content with fences removed and surrounding whitespace
            stripped.
        """
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        return cleaned.strip()


# ── Module-level singleton ────────────────────────────────────────────────────

api_analyzer = APIAnalyzer()
