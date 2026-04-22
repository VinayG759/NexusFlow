"""Code validator utility for NexusFlow.

Provides syntax checking and import validation for generated Python files,
allowing the agent pipeline to catch errors before they reach the user.

Usage::

    from src.utils.code_validator import code_validator

    result = code_validator.validate_python("weather-app/backend/main.py")
    result = code_validator.validate_project("weather-app/backend")
    result = code_validator.run_python_file("weather-app/backend/main.py")
    result = code_validator.check_imports("weather-app/backend/routes.py")
"""

from pathlib import Path

from src.tools.code_executor import CodeExecutorTool
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CodeValidator:
    """Validates generated Python files using the system Python interpreter.

    Wraps :class:`~src.tools.code_executor.CodeExecutorTool` to expose
    higher-level validation operations: syntax checking via ``py_compile``,
    AST-level import parsing, recursive project validation, and direct
    file execution.

    Attributes:
        _executor: The :class:`~src.tools.code_executor.CodeExecutorTool`
            instance used for all subprocess calls.
    """

    def __init__(self) -> None:
        """Initialise the CodeValidator with a CodeExecutorTool instance."""
        self._executor = CodeExecutorTool()
        logger.info("CodeValidator initialised.")

    # ── Public methods ────────────────────────────────────────────────────────

    def validate_python(self, file_path: str) -> dict:
        """Check a single Python file for syntax errors using py_compile.

        Invokes ``py -3.11 -m py_compile`` on the given path. This catches
        ``SyntaxError`` and ``IndentationError`` without executing the file.

        Args:
            file_path: Absolute or relative path to the ``.py`` file to check.

        Returns:
            On success::

                {
                    "status":    "valid",
                    "file_path": str,
                    "errors":    [],
                }

            On failure::

                {
                    "status":    "invalid",
                    "file_path": str,
                    "errors":    [str],   # stderr output from py_compile
                }
        """
        logger.info("Validating Python syntax: %r", file_path)
        result = self._executor.execute_command(f"py -3.11 -m py_compile {file_path}")

        if result["exit_code"] == 0:
            logger.info("Syntax OK: %r", file_path)
            return {"status": "valid", "file_path": file_path, "errors": []}

        errors = [result["stderr"]] if result["stderr"] else ["Unknown syntax error."]
        logger.warning("Syntax error in %r: %s", file_path, errors[0])
        return {"status": "invalid", "file_path": file_path, "errors": errors}

    def validate_project(self, project_path: str) -> dict:
        """Recursively validate all Python files under a directory.

        Walks *project_path* with :mod:`pathlib` to discover every ``*.py``
        file and runs :meth:`validate_python` on each one.

        Args:
            project_path: Path to the project root directory to scan.

        Returns:
            ::

                {
                    "status":        "valid" | "invalid",
                    "project_path":  str,
                    "valid_files":   list[str],
                    "invalid_files": list[{"file_path": str, "errors": list[str]}],
                }

            ``status`` is ``"valid"`` only when every file passes; otherwise
            ``"invalid"``.
        """
        logger.info("Validating project at: %r", project_path)
        root = Path(project_path)
        py_files = sorted(root.rglob("*.py"))

        if not py_files:
            logger.warning("No .py files found under %r", project_path)
            return {
                "status": "valid",
                "project_path": project_path,
                "valid_files": [],
                "invalid_files": [],
            }

        valid_files: list[str] = []
        invalid_files: list[dict] = []

        for py_file in py_files:
            path_str = str(py_file)
            result = self.validate_python(path_str)
            if result["status"] == "valid":
                valid_files.append(path_str)
            else:
                invalid_files.append({"file_path": path_str, "errors": result["errors"]})

        overall = "valid" if not invalid_files else "invalid"
        logger.info(
            "Project validation done — status=%s valid=%d invalid=%d",
            overall, len(valid_files), len(invalid_files),
        )
        return {
            "status": overall,
            "project_path": project_path,
            "valid_files": valid_files,
            "invalid_files": invalid_files,
        }

    def run_python_file(self, file_path: str, timeout: int = 10) -> dict:
        """Execute a Python file and capture its output.

        Runs ``py -3.11 {file_path}`` as a subprocess. Useful for smoke-testing
        a script or confirming a module-level singleton initialises without
        crashing.

        Args:
            file_path: Path to the ``.py`` file to execute.
            timeout: Maximum seconds to wait before aborting. Defaults to 10.

        Returns:
            ::

                {
                    "status":    "success" | "error" | "timeout",
                    "stdout":    str,
                    "stderr":    str,
                    "exit_code": int | None,
                }
        """
        logger.info("Running Python file: %r (timeout=%ds)", file_path, timeout)
        original_timeout = self._executor.timeout
        self._executor.timeout = timeout
        try:
            result = self._executor.execute_command(f"py -3.11 {file_path}")
        finally:
            self._executor.timeout = original_timeout

        logger.info(
            "run_python_file %r — status=%s exit_code=%s",
            file_path, result["status"], result["exit_code"],
        )
        return {
            "status": result["status"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "exit_code": result["exit_code"],
        }

    def check_imports(self, file_path: str) -> dict:
        """Parse a Python file's AST to verify it is syntactically well-formed.

        Uses ``ast.parse`` rather than ``py_compile`` — this is a pure-Python
        parse that does not require the file's imports to be resolvable, making
        it useful for validating generated code before its dependencies are
        installed.

        Args:
            file_path: Path to the ``.py`` file to parse.

        Returns:
            On success::

                {
                    "status":    "valid",
                    "file_path": str,
                    "errors":    [],
                }

            On failure::

                {
                    "status":    "invalid",
                    "file_path": str,
                    "errors":    [str],
                }
        """
        logger.info("AST-checking imports in: %r", file_path)
        # Use escaped braces so the path is embedded as a string literal.
        escaped = file_path.replace("\\", "\\\\").replace("'", "\\'")
        command = f"py -3.11 -c \"import ast; ast.parse(open('{escaped}').read())\""
        result = self._executor.execute_command(command)

        if result["exit_code"] == 0:
            logger.info("AST parse OK: %r", file_path)
            return {"status": "valid", "file_path": file_path, "errors": []}

        errors = [result["stderr"]] if result["stderr"] else ["AST parse failed."]
        logger.warning("AST parse error in %r: %s", file_path, errors[0])
        return {"status": "invalid", "file_path": file_path, "errors": errors}


# Module-level singleton — import this directly instead of instantiating CodeValidator yourself.
code_validator = CodeValidator()
