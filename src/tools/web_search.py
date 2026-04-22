"""Web search tool for NexusFlow.

Wraps the Tavily search API and exposes two methods agents can call:
- :meth:`WebSearchTool.search` — raw search results.
- :meth:`WebSearchTool.search_and_summarize` — results plus a combined summary.

Usage::

    from src.tools.web_search import web_search_tool

    result = web_search_tool.search("latest AI research papers")
    summary = web_search_tool.search_and_summarize("FastAPI best practices")
"""

from tavily import TavilyClient

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WebSearchTool:
    """Tavily-backed web search tool available to NexusFlow agents.

    Initialises a single :class:`TavilyClient` instance and provides a
    consistent result schema so agents do not need to know Tavily internals.

    Attributes:
        _client: The underlying :class:`TavilyClient` instance.
    """

    def __init__(self) -> None:
        """Initialise the Tavily client using the configured API key.

        Raises:
            ValueError: If ``settings.WEB_SEARCH_API_KEY`` is empty, preventing
                silent failures at search time.
        """
        if not settings.WEB_SEARCH_API_KEY:
            raise ValueError(
                "WEB_SEARCH_API_KEY is not set. "
                "Add it to your .env file before using WebSearchTool."
            )
        self._client = TavilyClient(api_key=settings.WEB_SEARCH_API_KEY)
        logger.info("WebSearchTool initialised.")

    def search(self, query: str, max_results: int | None = None) -> dict:
        """Perform a web search and return structured results.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return. Falls back to
                ``settings.MAX_SEARCH_RESULTS`` when not provided.

        Returns:
            On success::

                {
                    "query": str,
                    "results": [
                        {
                            "title": str,
                            "url": str,
                            "content": str,
                            "score": float,
                        },
                        ...
                    ],
                    "total_results": int,
                }

            On failure::

                {
                    "query": str,
                    "results": [],
                    "total_results": 0,
                    "error": str,
                }
        """
        limit = max_results if max_results is not None else settings.MAX_SEARCH_RESULTS
        logger.info("Searching web for query=%r (max_results=%d)", query, limit)

        try:
            response = self._client.search(query=query, max_results=limit)
            raw_results = response.get("results", [])
            results = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                }
                for item in raw_results
            ]
            logger.info("Search returned %d result(s) for query=%r", len(results), query)
            return {
                "query": query,
                "results": results,
                "total_results": len(results),
            }
        except Exception as exc:
            logger.exception("Web search failed for query=%r: %s", query, exc)
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": str(exc),
            }

    def search_and_summarize(self, query: str) -> dict:
        """Search the web and append a plain-text summary of the top results.

        Calls :meth:`search` internally and concatenates the ``content`` fields
        from the top 3 results into a single summary string.

        Args:
            query: The search query string.

        Returns:
            The same dict returned by :meth:`search` with one additional key::

                {
                    ...,          # all keys from search()
                    "summary": str,  # combined content from top 3 results,
                                     # or an empty string if search failed
                }
        """
        logger.info("Generating search summary for query=%r", query)
        result = self.search(query)

        top_contents = [
            item["content"]
            for item in result["results"][:3]
            if item.get("content")
        ]
        summary = "\n\n".join(top_contents)
        logger.info(
            "Summary generated from %d source(s) for query=%r",
            len(top_contents),
            query,
        )

        result["summary"] = summary
        return result


# Module-level singleton — import this directly instead of instantiating WebSearchTool yourself.
web_search_tool = WebSearchTool()
