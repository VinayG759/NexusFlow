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

# Primary Groq model. llama-3.1-8b-instant removed: its free-tier TPM cap (6000)
# is lower than a typical build prompt (~9k tokens), so it can never serve as
# a useful fallback. On 429 we retry the same model after the TPM window resets.
MODELS_IN_ORDER = [
    "llama-3.3-70b-versatile",   # 128k ctx, ~14k TPM free tier
]

# Global semaphore: serialise all Groq calls so concurrent builds don't exhaust
# the 12k TPM limit simultaneously (thundering-herd on 429 retries).
# Lazily initialised inside call_groq() so it binds to the running event loop.
_groq_semaphore: asyncio.Semaphore | None = None


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

    async def call_llm(self, prompt: str, model: str = "claude-3-5-sonnet-20240620", max_tokens: int = 4096) -> dict:
        """A wrapper to call an LLM, currently configured for Anthropic Claude."""
        return await self.call_anthropic(prompt, model, max_tokens)

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

    async def call_anthropic(
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
        model: str = _GROQ_DEFAULT_MODEL,
        max_tokens: int = _GROQ_DEFAULT_MAX_TOKENS,
        timeout: int = 120,
        max_retries: int = 3,
        initial_backoff: int = 65,
    ) -> dict:
        """Serialise all Groq calls through a single semaphore, then delegate to _call_groq_inner."""
        if not settings.GROQ_API_KEY:
            return {"status": "error", "error": "GROQ_API_KEY is not set in settings."}

        global _groq_semaphore
        if _groq_semaphore is None:
            _groq_semaphore = asyncio.Semaphore(1)

        async with _groq_semaphore:
            return await self._call_groq_inner(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                max_retries=max_retries,
                max_tokens=max_tokens,
            )

    async def _call_groq_inner(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        max_tokens: int | None = None,
    ) -> dict:
        """Inner Groq call — runs under the global semaphore acquired by call_groq."""
        requested_model = model or _GROQ_DEFAULT_MODEL

        # Build cascade: requested model first, then remaining fallbacks in order.
        models_to_try = [requested_model] + [
            m for m in MODELS_IN_ORDER if m != requested_model
        ]

        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "content-type": "application/json",
        }
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resolved_max_tokens = max_tokens if max_tokens is not None else _GROQ_DEFAULT_MAX_TOKENS
        last_error: str = ""

        for model_name in models_to_try:
            body: dict = {
                "model": model_name,
                "messages": messages,
                "max_tokens": resolved_max_tokens,
            }
            logger.info("Calling Groq model=%r prompt_len=%d", model_name, len(prompt))

            delay = retry_delay
            for attempt in range(max_retries):
                try:
                    logger.info("Calling Groq with model %s, timeout=%s", model_name, 120)
                    response = await self._client.post(
                        _GROQ_CHAT_URL, headers=headers, json=body,
                        timeout=120,
                    )
                    logger.info("Groq response status: %s", response.status_code)

                    # 429 = TPM rate limit — wait for the 1-minute window to reset,
                    # then retry the SAME model (don't cascade: fallback models have
                    # lower TPM caps and can't handle large prompts on the free tier).
                    if response.status_code == 429:
                        last_error = response.text
                        if attempt < max_retries - 1:
                            logger.warning(
                                "Groq 429 on model=%r — waiting 65s for TPM window reset (attempt %d/%d)",
                                model_name, attempt + 1, max_retries,
                            )
                            await asyncio.sleep(65)
                            continue  # retry same model
                        logger.warning(
                            "Groq 429 on model=%r — all %d attempts exhausted, cascading",
                            model_name, max_retries,
                        )
                        await asyncio.sleep(2)
                        break  # try next model only after all retries

                    # 413 = request too large for this model, 400 = model decommissioned
                    # Cascade immediately to the next model.
                    if response.status_code in (400, 413):
                        last_error = response.text
                        logger.warning(
                            "Groq %d on model=%r — error body: %s — cascading to next model",
                            response.status_code, model_name, last_error,
                        )
                        await asyncio.sleep(2)
                        break  # exit retry loop, move to next model

                    if not response.is_success:
                        error_text = response.text
                        logger.error(
                            "Groq API error %d on model=%r: %s",
                            response.status_code, model_name, error_text,
                        )
                        return {
                            "status": "error",
                            "error": f"HTTP {response.status_code}: {error_text}",
                        }

                    data = response.json()
                    content = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    usage = data.get("usage", {})
                    used_model = data.get("model", model_name)

                    logger.info(
                        "Groq responded — model=%r prompt_tokens=%s completion_tokens=%s",
                        used_model,
                        usage.get("prompt_tokens"),
                        usage.get("completion_tokens"),
                    )
                    if used_model != requested_model:
                        logger.warning(
                            "Groq fallback active: generated with %r instead of %r",
                            used_model, requested_model,
                        )

                    return {
                        "status": "success",
                        "content": content,
                        "model": used_model,
                        "model_used": used_model,
                        "usage": usage,
                    }

                except Exception as exc:
                    if attempt < max_retries - 1:
                        logger.warning(
                            "call_groq model=%r attempt %d/%d failed (%s) — retrying in %.1fs",
                            model_name, attempt + 1, max_retries, exc, delay,
                        )
                        await asyncio.sleep(delay)
                        delay *= 2
                    else:
                        logger.exception(
                            "call_groq model=%r all %d attempts failed: %s",
                            model_name, max_retries, exc,
                        )
                        return {"status": "error", "error": str(exc)}

        # Try Claude as last resort
        logger.warning("All Groq models exhausted, trying Claude API fallback...")
        claude_result = await self._call_claude_fallback(
            prompt,
            system_prompt or "",
            max_tokens if max_tokens is not None else _GROQ_DEFAULT_MAX_TOKENS,
        )
        if claude_result["status"] == "success":
            return claude_result
        return {"status": "error", "error": "All LLMs exhausted"}

    async def _call_claude_fallback(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 8000,
    ) -> dict:
        """Use Claude API as fallback when Groq TPD is exhausted."""
        api_key = settings.ANTHROPIC_API_KEY
        if not api_key:
            return {"status": "error", "error": "No ANTHROPIC_API_KEY set"}

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": max_tokens,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    content = data["content"][0]["text"]
                    logger.info("Claude fallback succeeded — model=claude-sonnet-4-20250514")
                    return {
                        "status": "success",
                        "content": content,
                        "model_used": "claude-sonnet-fallback",
                    }
                else:
                    logger.error("Claude fallback error %d: %s", response.status_code, response.text)
                    return {"status": "error", "error": f"Claude API error: {response.text}"}
        except Exception as exc:
            logger.exception("Claude fallback failed: %s", exc)
            return {"status": "error", "error": str(exc)}

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
