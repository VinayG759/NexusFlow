"""Manual smoke tests for ResearchAgent.

Run directly — no test framework required::

    python tests/test_research_agent.py
"""

from src.agents.research_agent import research_agent

SEP = "-" * 60


def test_research() -> None:
    print("TEST: research()")
    result = research_agent.research("FastAPI best practices")
    print(f"  status  : {result['status']}")
    print(f"  topic   : {result.get('topic', 'N/A')}")
    print(f"  summary : {result.get('summary', result.get('error', 'N/A'))[:300]}")


def test_find_apis() -> None:
    print("TEST: find_apis()")
    result = research_agent.find_apis("Stripe")
    print(f"  status : {result['status']}")
    for api in result.get("apis", [])[:2]:
        print(f"  title  : {api.get('title', 'N/A')}")
        print(f"  url    : {api.get('url', 'N/A')}")


def test_find_resources() -> None:
    print("TEST: find_resources()")
    result = research_agent.find_resources("parse PDF files in Python")
    print(f"  status : {result['status']}")
    for resource in result.get("resources", [])[:2]:
        print(f"  title  : {resource.get('title', 'N/A')}")
        print(f"  url    : {resource.get('url', 'N/A')}")


if __name__ == "__main__":
    test_research()
    print(SEP)
    test_find_apis()
    print(SEP)
    test_find_resources()
