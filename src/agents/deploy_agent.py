"""Deploy Agent for NexusFlow.

Handles testing, building, and serving projects on behalf of the orchestrator.
Uses :class:`~src.tools.code_executor.CodeExecutorTool` for subprocess execution
and :class:`~src.tools.file_manager.FileManagerTool` for filesystem inspection.

Usage::

    from src.agents.deploy_agent import deploy_agent

    deploy_agent.run_tests("my-api/")
    deploy_agent.check_dependencies("my-api/")
    deploy_agent.build_project("my-api/")
    deploy_agent.start_dev_server("my-api/", port=8000)
"""

from src.tools.code_executor import CodeExecutorTool
from src.tools.file_manager import FileManagerTool
from src.utils.logger import get_logger

logger = get_logger(__name__)

_REACT_MARKER = "package.json"
_FASTAPI_MARKER = "main.py"


class DeployAgent:
    """Agent responsible for testing, building, and running NexusFlow projects.

    Detects whether a project is FastAPI or React by inspecting the filesystem,
    then dispatches the appropriate tool commands via
    :class:`~src.tools.code_executor.CodeExecutorTool`.

    Attributes:
        agent_name: Human-readable identifier included in every result dict.
        _executor: Tool instance used for all subprocess execution.
        _file_manager: Tool instance used for filesystem inspection.
    """

    def __init__(self, agent_name: str) -> None:
        """Initialise the DeployAgent.

        Args:
            agent_name: Unique name for this agent, used in logs and result dicts.
        """
        self.agent_name = agent_name
        self._executor = CodeExecutorTool()
        self._file_manager = FileManagerTool()
        logger.info("DeployAgent '%s' initialised.", agent_name)

    # ── Internal helper ───────────────────────────────────────────────────────

    def _detect_project_type(self, project_path: str) -> str | None:
        """Return ``"fastapi"``, ``"react"``, or ``None`` based on marker files.

        Args:
            project_path: Root directory of the project to inspect.

        Returns:
            ``"fastapi"`` if ``main.py`` is present, ``"react"`` if
            ``package.json`` is present, ``None`` if neither is found.
        """
        if self._file_manager.file_exists(f"{project_path}/{_FASTAPI_MARKER}"):
            return "fastapi"
        if self._file_manager.file_exists(f"{project_path}/{_REACT_MARKER}"):
            return "react"
        return None

    # ── Public methods ────────────────────────────────────────────────────────

    def run_tests(self, project_path: str) -> dict:
        """Run the pytest test suite for a project.

        Args:
            project_path: Root directory of the project to test. pytest is
                invoked with ``-v`` so individual test names appear in output.

        Returns:
            ::

                {
                    "agent":   str,
                    "status":  "success" | "error" | "timeout",
                    "passed":  bool,    # True when exit_code is 0
                    "failed":  bool,    # True when exit_code is non-zero
                    "output":  str,     # combined stdout + stderr
                }
        """
        logger.info("[%s] Running tests in: %r", self.agent_name, project_path)
        try:
            result = self._executor.execute_command(
                f"py -3.11 -m pytest {project_path} -v"
            )
            passed = result["exit_code"] == 0
            output = result["stdout"] + (f"\n{result['stderr']}" if result["stderr"] else "")
            logger.info(
                "[%s] Tests finished — passed=%s status=%s",
                self.agent_name, passed, result["status"],
            )
            return {
                "agent": self.agent_name,
                "status": result["status"],
                "passed": passed,
                "failed": not passed,
                "output": output,
            }
        except Exception as exc:
            logger.exception("[%s] run_tests failed: %s", self.agent_name, exc)
            return {
                "agent": self.agent_name,
                "status": "error",
                "passed": False,
                "failed": True,
                "output": str(exc),
            }

    def start_dev_server(self, project_path: str, port: int = 8000) -> dict:
        """Start a development server for a FastAPI or React project.

        Detects the project type from the filesystem:
        - FastAPI (``main.py`` present): runs ``uvicorn main:app --reload``
        - React (``package.json`` present): runs ``npm start``

        Args:
            project_path: Root directory of the project to serve.
            port: Port number for the dev server. Used for FastAPI; React uses
                the port configured in its own scripts.

        Returns:
            ::

                {
                    "agent":        str,
                    "status":       "success" | "error",
                    "project_path": str,
                    "port":         int,
                    "message":      str,
                }
        """
        logger.info("[%s] Starting dev server for: %r on port %d", self.agent_name, project_path, port)
        try:
            project_type = self._detect_project_type(project_path)

            if project_type == "fastapi":
                command = (
                    f"py -3.11 -m uvicorn main:app --app-dir {project_path} "
                    f"--host 0.0.0.0 --port {port} --reload"
                )
                message = f"FastAPI dev server starting at http://0.0.0.0:{port}"
            elif project_type == "react":
                command = f"npm start --prefix {project_path}"
                message = f"React dev server starting (default port 3000)."
            else:
                return {
                    "agent": self.agent_name,
                    "status": "error",
                    "project_path": project_path,
                    "port": port,
                    "message": (
                        f"Could not detect project type in '{project_path}'. "
                        f"Expected '{_FASTAPI_MARKER}' or '{_REACT_MARKER}'."
                    ),
                }

            result = self._executor.execute_command(command)
            status = result["status"]
            logger.info("[%s] Dev server command finished — status=%s", self.agent_name, status)
            return {
                "agent": self.agent_name,
                "status": status,
                "project_path": project_path,
                "port": port,
                "message": message if status == "success" else result.get("stderr", "Server failed to start."),
            }
        except Exception as exc:
            logger.exception("[%s] start_dev_server failed: %s", self.agent_name, exc)
            return {
                "agent": self.agent_name,
                "status": "error",
                "project_path": project_path,
                "port": port,
                "message": str(exc),
            }

    def build_project(self, project_path: str) -> dict:
        """Build a production artefact or validate the project runs.

        - **React**: runs ``npm run build`` to produce a ``build/`` directory.
        - **FastAPI**: performs a dry-run import check (``python -c "import main"``)
          to confirm the project imports cleanly before deployment.

        Args:
            project_path: Root directory of the project to build.

        Returns:
            ::

                {
                    "agent":        str,
                    "status":       "success" | "error" | "timeout",
                    "project_path": str,
                    "message":      str,
                    "output":       str,
                }
        """
        logger.info("[%s] Building project: %r", self.agent_name, project_path)
        try:
            project_type = self._detect_project_type(project_path)

            if project_type == "fastapi":
                command = f"py -3.11 -c \"import sys; sys.path.insert(0, '{project_path}'); import main\""
                success_msg = f"FastAPI project '{project_path}' validated successfully."
            elif project_type == "react":
                command = f"npm run build --prefix {project_path}"
                success_msg = f"React project '{project_path}' built successfully."
            else:
                return {
                    "agent": self.agent_name,
                    "status": "error",
                    "project_path": project_path,
                    "message": (
                        f"Could not detect project type in '{project_path}'. "
                        f"Expected '{_FASTAPI_MARKER}' or '{_REACT_MARKER}'."
                    ),
                    "output": "",
                }

            result = self._executor.execute_command(command)
            output = result["stdout"] + (f"\n{result['stderr']}" if result["stderr"] else "")
            message = success_msg if result["status"] == "success" else f"Build failed for '{project_path}'."
            logger.info("[%s] build_project — status=%s", self.agent_name, result["status"])
            return {
                "agent": self.agent_name,
                "status": result["status"],
                "project_path": project_path,
                "message": message,
                "output": output,
            }
        except Exception as exc:
            logger.exception("[%s] build_project failed: %s", self.agent_name, exc)
            return {
                "agent": self.agent_name,
                "status": "error",
                "project_path": project_path,
                "message": str(exc),
                "output": "",
            }

    def check_dependencies(self, project_path: str) -> dict:
        """Inspect a project's dependency file and report missing packages.

        Checks for ``requirements.txt`` (FastAPI/Python) or ``package.json``
        (React). For Python projects, runs ``pip check`` to surface unmet
        dependencies. For React projects, checks whether ``node_modules/``
        exists.

        Args:
            project_path: Root directory of the project to inspect.

        Returns:
            ::

                {
                    "agent":             str,
                    "status":            "success" | "error",
                    "project_path":      str,
                    "dependencies_file": str | None,  # path to found dep file
                    "missing_packages":  list[str],   # names of missing packages
                }
        """
        logger.info("[%s] Checking dependencies for: %r", self.agent_name, project_path)
        try:
            missing_packages: list[str] = []
            dependencies_file: str | None = None

            req_path = f"{project_path}/requirements.txt"
            pkg_path = f"{project_path}/package.json"

            if self._file_manager.file_exists(req_path):
                dependencies_file = req_path
                pip_check = self._executor.execute_command("py -3.11 -m pip check")
                if pip_check["status"] != "success" and pip_check.get("stdout"):
                    missing_packages = [
                        line.split()[0]
                        for line in pip_check["stdout"].splitlines()
                        if line.strip() and "No broken requirements" not in line
                    ]

            elif self._file_manager.file_exists(pkg_path):
                dependencies_file = pkg_path
                node_modules = f"{project_path}/node_modules"
                if not self._file_manager.file_exists(f"{node_modules}/.package-lock.json"):
                    missing_packages = ["node_modules (run npm install)"]

            else:
                logger.warning(
                    "[%s] No dependency file found in '%s'.", self.agent_name, project_path,
                )
                return {
                    "agent": self.agent_name,
                    "status": "error",
                    "project_path": project_path,
                    "dependencies_file": None,
                    "missing_packages": [],
                }

            logger.info(
                "[%s] check_dependencies — file=%s missing=%d",
                self.agent_name, dependencies_file, len(missing_packages),
            )
            return {
                "agent": self.agent_name,
                "status": "success",
                "project_path": project_path,
                "dependencies_file": dependencies_file,
                "missing_packages": missing_packages,
            }
        except Exception as exc:
            logger.exception("[%s] check_dependencies failed: %s", self.agent_name, exc)
            return {
                "agent": self.agent_name,
                "status": "error",
                "project_path": project_path,
                "dependencies_file": None,
                "missing_packages": [],
            }


# Module-level singleton — import this directly instead of instantiating DeployAgent yourself.
deploy_agent = DeployAgent("DeployAgent")
