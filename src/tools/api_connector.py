"""API connector tool for NexusFlow.

Provides a shared async HTTP client for generic REST calls and a dedicated
method for calling the Anthropic Claude API. All network I/O is async so
agents can await these methods without blocking the event loop.

Usage::

    from src.tools.api_connector import api_connector

    # Generic requests
    result = await api_connector.get("https://api.example.com/data")
    result = await api_connector.post("https://api.example.com/items", payload={"name": "x"})

    # Claude LLM
    result = await api_connector.call_llm("Summarise this text: ...")

    # Shutdown (call once at app teardown)
    await api_connector.close()
"""

import asyncio
import httpx

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_API_VERSION = "2023-06-01"
_ANTHROPIC_DEFAULT_MAX_TOKENS = 1024

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
_GROQ_DEFAULT_MAX_TOKENS = 4096


class APIConnectorTool:
    """Async HTTP client tool for REST API calls and LLM access.

    Maintains a single :class:`httpx.AsyncClient` instance across requests
    so connections are reused (keep-alive) rather than opened per call.

    Attributes:
        timeout: Request timeout in seconds applied to all outgoing calls.
        _client: The shared :class:`httpx.AsyncClient` instance.
    """

    def __init__(self, timeout: int = 30) -> None:
        """Initialise the APIConnectorTool with a shared async HTTP client.

        Args:
            timeout: Seconds before a request is aborted. Applied as both
                connect and read timeout. Defaults to 30 seconds.
        """
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        logger.info("APIConnectorTool initialised with timeout=%ds.", timeout)

    async def get(
        self,
        url: str,
        headers: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """Make an async HTTP GET request.

        Args:
            url: Full URL to request.
            headers: Optional request headers merged with client defaults.
            params: Optional query-string parameters.

        Returns:
            On success::

                {
                    "status":      "success",
                    "status_code": int,
                    "data":        dict | str,  # parsed JSON or raw text
                    "url":         str,
                }

            On failure::

                {
                    "status":      "error",
                    "status_code": int | None,
                    "url":         str,
                    "error":       str,
                }
        """
        logger.info("GET %s params=%s", url, params)
        try:
            response = await self._client.get(url, headers=headers or {}, params=params or {})
            data = _parse_response_body(response)
            logger.info("GET %s → %d", url, response.status_code)
            return {
                "status": "success" if response.is_success else "error",
                "status_code": response.status_code,
                "data": data,
                "url": url,
            }
        except Exception as exc:
            logger.exception("GET %s failed: %s", url, exc)
            return {"status": "error", "status_code": None, "url": url, "error": str(exc)}

    async def post(
        self,
        url: str,
        headers: dict | None = None,
        payload: dict | None = None,
    ) -> dict:
        """Make an async HTTP POST request with a JSON body.

        Args:
            url: Full URL to POST to.
            headers: Optional request headers merged with client defaults.
            payload: Dict serialised as the JSON request body. Defaults to ``{}``.

        Returns:
            On success::

                {
                    "status":      "success",
                    "status_code": int,
                    "data":        dict | str,
                    "url":         str,
                }

            On failure::

                {
                    "status":      "error",
                    "status_code": int | None,
                    "url":         str,
                    "error":       str,
                }
        """
        logger.info("POST %s payload_keys=%s", url, list((payload or {}).keys()))
        try:
            response = await self._client.post(url, headers=headers or {}, json=payload or {})
            data = _parse_response_body(response)
            logger.info("POST %s → %d", url, response.status_code)
            return {
                "status": "success" if response.is_success else "error",
                "status_code": response.status_code,
                "data": data,
                "url": url,
            }
        except Exception as exc:
            logger.exception("POST %s failed: %s", url, exc)
            return {"status": "error", "status_code": None, "url": url, "error": str(exc)}

    async def call_llm(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> dict:
        """Call the Anthropic Claude API and return the assistant's response.

        Sends a single user turn with an optional system prompt. Uses
        ``settings.ANTHROPIC_API_KEY`` for authentication and falls back to
        ``settings.DEFAULT_MODEL`` when *model* is not specified.

        Args:
            prompt: The user message text to send to Claude.
            system_prompt: Optional system-level instruction prepended to the
                conversation (e.g. ``"You are a code review assistant."``).
            model: Anthropic model ID to use. Defaults to
                ``settings.DEFAULT_MODEL``.

        Returns:
            On success::

                {
                    "status":  "success",
                    "content": str,   # text of the first content block
                    "model":   str,   # model ID echo'd from the response
                    "usage":   dict,  # {"input_tokens": int, "output_tokens": int}
                }

            On failure::

                {
                    "status": "error",
                    "error":  str,
                }
        """
        if not settings.ANTHROPIC_API_KEY:
            return {
                "status": "error",
                "error": "ANTHROPIC_API_KEY is not set in settings.",
            }

        resolved_model = model or settings.DEFAULT_MODEL
        logger.info("Calling Claude model=%r prompt_len=%d", resolved_model, len(prompt))

        headers = {
            "x-api-key": settings.ANTHROPIC_API_KEY,
            "anthropic-version": _ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
        body: dict = {
            "model": resolved_model,
            "max_tokens": _ANTHROPIC_DEFAULT_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            body["system"] = system_prompt

        try:
            response = await self._client.post(
                _ANTHROPIC_MESSAGES_URL, headers=headers, json=body,
            )
            if not response.is_success:
                error_text = response.text
                logger.error("Claude API error %d: %s", response.status_code, error_text)
                return {
                    "status": "error",
                    "error": f"HTTP {response.status_code}: {error_text}",
                }

            data = response.json()
            content = data.get("content", [{}])[0].get("text", "")
            usage = data.get("usage", {})
            logger.info(
                "Claude responded — model=%r input_tokens=%s output_tokens=%s",
                data.get("model"), usage.get("input_tokens"), usage.get("output_tokens"),
            )
            return {
                "status": "success",
                "content": content,
                "model": data.get("model", resolved_model),
                "usage": usage,
            }
        except Exception as exc:
            logger.exception("call_llm failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    async def call_groq(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        max_tokens: int | None = None,
    ) -> dict:
        """Call the Groq inference API using an OpenAI-compatible chat endpoint.

        Sends a chat completion request to Groq with an optional system message.
        Uses ``settings.GROQ_API_KEY`` for Bearer token authentication and
        defaults to ``"llama-3.3-70b-versatile"`` when *model* is not specified.

        Args:
            prompt: The user message text to send to the model.
            system_prompt: Optional system-level instruction added as the first
                message in the conversation (e.g. ``"You are a code assistant."``).
            model: Groq model ID to use. Defaults to ``"llama-3.3-70b-versatile"``.

        Returns:
            On success::

                {
                    "status":  "success",
                    "content": str,   # text of the first choice's message
                    "model":   str,   # model ID echo'd from the response
                    "usage":   dict,  # {"prompt_tokens": int, "completion_tokens": int, ...}
                }

            On failure::

                {
                    "status": "error",
                    "error":  str,
                }
        """
        if not settings.GROQ_API_KEY:
            return {
                "status": "error",
                "error": "GROQ_API_KEY is not set in settings.",
            }

        resolved_model = model or _GROQ_DEFAULT_MODEL
        logger.info("Calling Groq model=%r prompt_len=%d", resolved_model, len(prompt))

        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "content-type": "application/json",
        }
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: dict = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else _GROQ_DEFAULT_MAX_TOKENS,
        }

        delay = retry_delay
        last_error: str = ""
        for attempt in range(max_retries):
            try:
                response = await self._client.post(_GROQ_CHAT_URL, headers=headers, json=body)

                if response.status_code == 429:
                    last_error = response.text
                    if attempt < max_retries - 1:
                        logger.warning(
                            "Groq rate limit (429) on attempt %d/%d — retrying in %.1fs",
                            attempt + 1, max_retries, delay,
                        )
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    logger.error("Groq rate limit exhausted after %d attempts.", max_retries)
                    return {"status": "error", "error": f"Rate limit exceeded: {last_error}"}

                if not response.is_success:
                    error_text = response.text
                    logger.error("Groq API error %d: %s", response.status_code, error_text)
                    return {
                        "status": "error",
                        "error": f"HTTP {response.status_code}: {error_text}",
                    }

                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})
                logger.info(
                    "Groq responded — model=%r prompt_tokens=%s completion_tokens=%s",
                    data.get("model"), usage.get("prompt_tokens"), usage.get("completion_tokens"),
                )
                return {
                    "status": "success",
                    "content": content,
                    "model": data.get("model", resolved_model),
                    "usage": usage,
                }
            except Exception as exc:
                logger.exception("call_groq attempt %d failed: %s", attempt + 1, exc)
                return {"status": "error", "error": str(exc)}

        return {"status": "error", "error": f"Rate limit exceeded after {max_retries} retries: {last_error}"}

    async def close(self) -> None:
        """Close the underlying HTTP client and release all connections.

        Should be called once during application shutdown, e.g. in the FastAPI
        lifespan teardown handler.
        """
        await self._client.aclose()
        logger.info("APIConnectorTool HTTP client closed.")


# ── Internal helper ───────────────────────────────────────────────────────────


def _parse_response_body(response: httpx.Response) -> dict | str:
    """Return parsed JSON if the response is JSON, otherwise raw text."""
    try:
        return response.json()
    except Exception:
        return response.text


# Module-level singleton — import this directly instead of instantiating APIConnectorTool yourself.
api_connector = APIConnectorTool()
