"""Builder Agent for NexusFlow.

Responsible for generating code, scaffolding project structures, running
tests, and managing dependencies. Uses :class:`~src.tools.file_manager.FileManagerTool`
for all filesystem work and :class:`~src.tools.code_executor.CodeExecutorTool`
for subprocess execution.

Usage::

    from src.agents.builder_agent import builder_agent

    builder_agent.generate_code("REST API endpoint for user login", language="python")
    builder_agent.create_project("my-api", "fastapi", "User management service")
    builder_agent.run_project_tests("tests/")
    builder_agent.install_dependencies("requirements.txt")
"""

from src.config.settings import settings
from src.tools.api_connector import api_connector
from src.tools.code_executor import CodeExecutorTool
from src.tools.file_manager import FileManagerTool
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Project templates ─────────────────────────────────────────────────────────
# Minimal file skeletons for each supported project_type.
# Keys are relative paths; values are starter content strings.

_FASTAPI_STRUCTURE: dict[str, str] = {
    "main.py": (
        'from fastapi import FastAPI\n\napp = FastAPI()\n\n\n'
        '@app.get("/")\ndef root():\n    return {"message": "Hello from FastAPI"}\n'
    ),
    "requirements.txt": "fastapi==0.115.12\nuvicorn[standard]==0.34.0\n",
    "src/__init__.py": "",
    "src/routes/__init__.py": "",
    "src/models/__init__.py": "",
    "tests/__init__.py": "",
    "tests/test_main.py": (
        "from fastapi.testclient import TestClient\nfrom main import app\n\n"
        "client = TestClient(app)\n\n\ndef test_root():\n"
        '    response = client.get("/")\n    assert response.status_code == 200\n'
    ),
    ".env.example": "# Add environment variables here\n",
}

_REACT_STRUCTURE: dict[str, str] = {
    "package.json": (
        '{\n  "name": "react-app",\n  "version": "0.1.0",\n'
        '  "dependencies": {\n    "react": "^18.0.0",\n    "react-dom": "^18.0.0"\n  }\n}\n'
    ),
    "public/index.html": (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head><meta charset=\"UTF-8\" />"
        "<title>React App</title></head>\n<body><div id=\"root\"></div></body>\n</html>\n"
    ),
    "src/index.jsx": (
        "import React from 'react';\nimport ReactDOM from 'react-dom/client';\n"
        "import App from './App';\n\nReactDOM.createRoot(document.getElementById('root'))"
        ".render(<App />);\n"
    ),
    "src/App.jsx": (
        "export default function App() {\n"
        '  return <h1>Hello from React</h1>;\n}\n'
    ),
    "src/components/.gitkeep": "",
}

_FULLSTACK_STRUCTURE: dict[str, str] = {
    **{f"backend/{k}": v for k, v in _FASTAPI_STRUCTURE.items()},
    **{f"frontend/{k}": v for k, v in _REACT_STRUCTURE.items()},
    "docker-compose.yml": (
        "version: '3.9'\nservices:\n"
        "  backend:\n    build: ./backend\n    ports:\n      - '8000:8000'\n"
        "  frontend:\n    build: ./frontend\n    ports:\n      - '3000:3000'\n"
    ),
    "README.md": "# Fullstack Project\n\nBackend: FastAPI  |  Frontend: React\n",
}

_TEMPLATES: dict[str, dict[str, str]] = {
    "fastapi": _FASTAPI_STRUCTURE,
    "react": _REACT_STRUCTURE,
    "fullstack": _FULLSTACK_STRUCTURE,
}


class BuilderAgent:
    """Agent responsible for code generation and project scaffolding.

    Combines :class:`~src.tools.file_manager.FileManagerTool` for filesystem
    operations and :class:`~src.tools.code_executor.CodeExecutorTool` for
    running tests and installing packages.

    Attributes:
        agent_name: Human-readable identifier included in every result dict.
        _file_manager: Tool instance used for all file I/O.
        _executor: Tool instance used for subprocess execution.
    """

    def __init__(self, agent_name: str) -> None:
        """Initialise the BuilderAgent.

        Args:
            agent_name: Unique name for this agent, used in logs and result dicts.
        """
        self.agent_name = agent_name
        self._file_manager = FileManagerTool()
        self._executor = CodeExecutorTool()
        logger.info("BuilderAgent '%s' initialised.", agent_name)

    async def generate_code(self, task_description: str, language: str = "python") -> dict:
        """Generate source code for a given task description via the Claude LLM.

        Calls the Anthropic Claude API through
        :func:`~src.tools.api_connector.APIConnectorTool.call_llm` with a
        system prompt that instructs the model to return only raw code with no
        explanation or markdown fencing.

        Args:
            task_description: Plain-language description of what the code
                should do (e.g. ``"REST endpoint that validates a JWT token"``).
            language: Target programming language. Defaults to ``"python"``.

        Returns:
            On success::

                {
                    "agent":    str,
                    "status":   "success",
                    "language": str,
                    "task":     str,
                    "code":     str,   # raw code returned by the LLM
                    "message":  str,
                }

            On failure::

                {
                    "agent":    str,
                    "status":   "error",
                    "language": str,
                    "task":     str,
                    "error":    str,
                }
        """
        logger.info(
            "[%s] Generating %s code for task: %r", self.agent_name, language, task_description,
        )
        system_prompt = (
            "You are an expert software engineer. You must follow these rules strictly:\n"
            "1. Return ONLY raw code with no explanation, comments about the task, or markdown formatting\n"
            "2. Do not wrap code in triple backticks\n"
            "3. Always use FastAPI (never Flask) for Python backend code\n"
            "4. Always use async/await patterns in FastAPI\n"
            "5. Always use SQLAlchemy with asyncpg for database operations\n"
            "6. Always use TypeScript for React frontend code\n"
            "7. Write production-ready code with proper error handling\n\n"
            "Tech stack constraints:\n"
            "- Backend: FastAPI + SQLAlchemy + asyncpg + PostgreSQL\n"
            "- Frontend: React + TypeScript + TailwindCSS\n"
            "- Never use Flask, Django, or MongoDB"
        )
        user_prompt = (
            "Tech stack: FastAPI backend, React+TypeScript frontend, PostgreSQL database.\n"
            f"Task: {task_description}"
        )
        try:
            llm_result = await api_connector.call_groq(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
            if llm_result["status"] != "success":
                raise RuntimeError(llm_result.get("error", "LLM call returned non-success status."))

            code = self._clean_code(llm_result["content"])
            logger.info("[%s] Code generation succeeded for task: %r", self.agent_name, task_description)
            return {
                "agent": self.agent_name,
                "status": "success",
                "language": language,
                "task": task_description,
                "code": code,
                "message": f"Code generated successfully using {llm_result.get('model', settings.DEFAULT_MODEL)}.",
            }
        except Exception as exc:
            logger.exception("[%s] generate_code failed: %s", self.agent_name, exc)
            return {
                "agent": self.agent_name,
                "status": "error",
                "language": language,
                "task": task_description,
                "error": str(exc),
            }

    def _clean_code(self, code: str) -> str:
        """Strip markdown fencing and surrounding whitespace from LLM output."""
        code = code.strip()
        for fence in ("```python", "```typescript", "```jsx", "```tsx", "```"):
            if code.startswith(fence):
                code = code[len(fence):]
                break
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    def create_project(
        self,
        project_name: str,
        project_type: str,
        description: str,
    ) -> dict:
        """Scaffold a new project with a standard directory structure.

        Supported *project_type* values: ``"fastapi"``, ``"react"``,
        ``"fullstack"``. Files are written under ``<project_name>/`` relative
        to the file manager's base directory.

        Args:
            project_name: Name of the root directory to create (e.g. ``"my-api"``).
            project_type: Template to use — one of ``"fastapi"``, ``"react"``,
                or ``"fullstack"``.
            description: Human-readable description stored in the README comment.

        Returns:
            On success::

                {
                    "agent":         str,
                    "status":        "success" | "partial",
                    "project_name":  str,
                    "project_type":  str,
                    "files_created": list[str],
                    "message":       str,
                }

            On failure::

                {
                    "agent":        str,
                    "status":       "error",
                    "project_name": str,
                    "project_type": str,
                    "error":        str,
                }
        """
        logger.info(
            "[%s] Creating %r project: %r", self.agent_name, project_type, project_name,
        )
        try:
            template = _TEMPLATES.get(project_type)
            if template is None:
                raise ValueError(
                    f"Unknown project_type '{project_type}'. "
                    f"Choose from: {', '.join(_TEMPLATES)}."
                )

            structure = {
                f"{project_name}/{path}": content
                for path, content in template.items()
            }
            # Inject the description into a top-level README if not already present.
            readme_key = f"{project_name}/README.md"
            if readme_key not in structure:
                structure[readme_key] = f"# {project_name}\n\n{description}\n"

            files_created: list[str] = []
            files_failed: list[str] = []

            for path, content in structure.items():
                result = self._file_manager.create_file(path, content)
                if result["status"] == "success":
                    files_created.append(path)
                else:
                    files_failed.append(path)
                    logger.warning("[%s] Failed to create %s: %s", self.agent_name, path, result.get("error"))

            status = "success" if not files_failed else "partial"
            message = (
                f"Project '{project_name}' scaffolded with {len(files_created)} file(s)."
                if status == "success"
                else f"Project partially created — {len(files_failed)} file(s) failed."
            )
            logger.info("[%s] create_project done — status=%s", self.agent_name, status)
            return {
                "agent": self.agent_name,
                "status": status,
                "project_name": project_name,
                "project_type": project_type,
                "files_created": files_created,
                "message": message,
            }
        except Exception as exc:
            logger.exception("[%s] create_project failed: %s", self.agent_name, exc)
            return {
                "agent": self.agent_name,
                "status": "error",
                "project_name": project_name,
                "project_type": project_type,
                "error": str(exc),
            }

    def run_project_tests(self, test_path: str) -> dict:
        """Run pytest on a given path and return the output.

        Args:
            test_path: Path to the test file or directory to pass to pytest
                (e.g. ``"tests/"`` or ``"tests/test_main.py"``).

        Returns:
            ::

                {
                    "agent":     str,
                    "status":    "success" | "error" | "timeout",
                    "test_path": str,
                    "stdout":    str,
                    "stderr":    str,
                    "exit_code": int | None,
                }
        """
        logger.info("[%s] Running tests at: %r", self.agent_name, test_path)
        try:
            result = self._executor.execute_command(f"py -3.11 -m pytest {test_path} -v")
            logger.info(
                "[%s] Tests finished — status=%s exit_code=%s",
                self.agent_name, result["status"], result["exit_code"],
            )
            return {
                "agent": self.agent_name,
                "status": result["status"],
                "test_path": test_path,
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "exit_code": result["exit_code"],
            }
        except Exception as exc:
            logger.exception("[%s] run_project_tests failed: %s", self.agent_name, exc)
            return {
                "agent": self.agent_name,
                "status": "error",
                "test_path": test_path,
                "stdout": "",
                "stderr": str(exc),
                "exit_code": None,
            }

    def install_dependencies(self, requirements_path: str) -> dict:
        """Install packages listed in a pip requirements file.

        Args:
            requirements_path: Path to the ``requirements.txt`` (or similar)
                file to pass to ``pip install -r``.

        Returns:
            ::

                {
                    "agent":   str,
                    "status":  "success" | "error" | "timeout",
                    "message": str,
                    "stdout":  str,
                    "stderr":  str,
                }
        """
        logger.info("[%s] Installing dependencies from: %r", self.agent_name, requirements_path)
        try:
            result = self._executor.execute_command(
                f"py -3.11 -m pip install -r {requirements_path}"
            )
            if result["status"] == "success":
                message = f"Dependencies from '{requirements_path}' installed successfully."
            elif result["status"] == "timeout":
                message = f"pip install timed out for '{requirements_path}'."
            else:
                message = f"Failed to install dependencies from '{requirements_path}'."

            logger.info(
                "[%s] install_dependencies done — status=%s", self.agent_name, result["status"],
            )
            return {
                "agent": self.agent_name,
                "status": result["status"],
                "message": message,
                "stdout": result["stdout"],
                "stderr": result["stderr"],
            }
        except Exception as exc:
            logger.exception("[%s] install_dependencies failed: %s", self.agent_name, exc)
            return {
                "agent": self.agent_name,
                "status": "error",
                "message": str(exc),
                "stdout": "",
                "stderr": str(exc),
            }


# Module-level singleton — import this directly instead of instantiating BuilderAgent yourself.
builder_agent = BuilderAgent("BuilderAgent")
