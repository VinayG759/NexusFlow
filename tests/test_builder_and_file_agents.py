"""Manual smoke tests for BuilderAgent and FileAgent.

Run directly — no test framework required::

    python tests/test_builder_and_file_agents.py
"""

from src.agents.builder_agent import builder_agent
from src.agents.file_agent import file_agent

SEP = "-" * 60


def test_create_project() -> None:
    print("TEST: builder_agent.create_project()")
    result = builder_agent.create_project("test-project", "fastapi", "A test FastAPI project")
    print(f"  status       : {result['status']}")
    print(f"  project_name : {result.get('project_name', 'N/A')}")
    print(f"  project_type : {result.get('project_type', 'N/A')}")
    print(f"  files created: {len(result.get('files_created', []))}")


def test_generate_code() -> None:
    print("TEST: builder_agent.generate_code()")
    result = builder_agent.generate_code("create a user authentication endpoint", "python")
    print(f"  status   : {result['status']}")
    print(f"  language : {result.get('language', 'N/A')}")
    print(f"  task     : {result.get('task', 'N/A')}")
    print(f"  code     :\n{result.get('code', result.get('error', 'N/A'))}")


def test_file_operations() -> None:
    print("TEST: file_agent CRUD operations")

    created = file_agent.create_project_file("test_output/hello.txt", "Hello from NexusFlow!")
    print(f"  create status : {created['status']}")

    read1 = file_agent.read_project_file("test_output/hello.txt")
    print(f"  initial read  : {read1.get('content', read1.get('error', 'N/A'))}")

    file_agent.update_project_file("test_output/hello.txt", "Updated by NexusFlow!")
    read2 = file_agent.read_project_file("test_output/hello.txt")
    print(f"  updated read  : {read2.get('content', read2.get('error', 'N/A'))}")

    deleted = file_agent.delete_project_file("test_output/hello.txt")
    print(f"  delete status : {deleted['status']}")


def test_setup_project_structure() -> None:
    print("TEST: file_agent.setup_project_structure()")
    result = file_agent.setup_project_structure({
        "test_output/app.py": "print('app')",
        "test_output/config.py": "DEBUG = True",
        "test_output/README.md": "# Test Project",
    })
    print(f"  status  : {result['status']}")
    print(f"  created : {len(result.get('created', []))}")
    print(f"  failed  : {len(result.get('failed', []))}")


if __name__ == "__main__":
    test_create_project()
    print(SEP)
    test_generate_code()
    print(SEP)
    test_file_operations()
    print(SEP)
    test_setup_project_structure()
