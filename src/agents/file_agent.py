"""File Agent for NexusFlow.

Wraps :class:`~src.tools.file_manager.FileManagerTool` and exposes a
higher-level interface that the orchestrator and other agents can use to
read, write, and manage project files without dealing with filesystem details.

Usage::

    from src.agents.file_agent import file_agent

    file_agent.create_project_file("src/main.py", "print('hello')")
    result = file_agent.read_project_file("src/main.py")
    file_agent.setup_project_structure({
        "src/__init__.py": "",
        "src/app.py": "# app entry point",
        "README.md": "# My Project",
    })
"""

from src.tools.file_manager import FileManagerTool
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FileAgent:
    """Agent responsible for all file system operations in a NexusFlow project.

    Delegates every I/O operation to :class:`~src.tools.file_manager.FileManagerTool`
    and enriches each result with the agent's name so the orchestrator can
    attribute outputs uniformly across agents.

    Attributes:
        agent_name: Human-readable identifier included in every result dict.
        _file_manager: The :class:`~src.tools.file_manager.FileManagerTool`
            instance used for all filesystem operations.
    """

    def __init__(self, agent_name: str) -> None:
        """Initialise the FileAgent.

        Args:
            agent_name: A unique name for this agent instance, included in
                every result dict for traceability.
        """
        self.agent_name = agent_name
        self._file_manager = FileManagerTool()
        logger.info("FileAgent '%s' initialised.", agent_name)

    # ── Internal helper ───────────────────────────────────────────────────────

    def _enrich(self, result: dict) -> dict:
        """Prepend the agent name to a result dict returned by FileManagerTool."""
        return {"agent": self.agent_name, **result}

    # ── Public methods ────────────────────────────────────────────────────────

    def create_project_file(self, relative_path: str, content: str) -> dict:
        """Create a file at *relative_path* with the given content.

        Missing parent directories are created automatically. Existing files
        are silently overwritten.

        Args:
            relative_path: Destination path relative to the tool's base directory.
            content: UTF-8 text to write into the file.

        Returns:
            On success::

                {"agent": str, "status": "success", "path": str, "message": str}

            On failure::

                {"agent": str, "status": "error", "path": str, "error": str}
        """
        logger.info("[%s] Creating file: %s", self.agent_name, relative_path)
        result = self._file_manager.create_file(relative_path, content)
        return self._enrich(result)

    def read_project_file(self, relative_path: str) -> dict:
        """Read and return the full text content of a file.

        Args:
            relative_path: Path of the file to read, relative to the tool's
                base directory.

        Returns:
            On success::

                {"agent": str, "status": "success", "path": str, "content": str}

            On failure::

                {"agent": str, "status": "error", "path": str, "error": str}
        """
        logger.info("[%s] Reading file: %s", self.agent_name, relative_path)
        result = self._file_manager.read_file(relative_path)
        return self._enrich(result)

    def update_project_file(self, relative_path: str, content: str) -> dict:
        """Overwrite an existing file with new content.

        Fails with an error dict if the file does not already exist, keeping
        the intent distinct from :meth:`create_project_file`.

        Args:
            relative_path: Path of the file to update, relative to the tool's
                base directory.
            content: New UTF-8 text content that replaces the entire file.

        Returns:
            On success::

                {"agent": str, "status": "success", "path": str, "message": str}

            On failure::

                {"agent": str, "status": "error", "path": str, "error": str}
        """
        logger.info("[%s] Updating file: %s", self.agent_name, relative_path)
        result = self._file_manager.update_file(relative_path, content)
        return self._enrich(result)

    def delete_project_file(self, relative_path: str) -> dict:
        """Delete a file if it exists (idempotent — succeeds if already absent).

        Args:
            relative_path: Path of the file to delete, relative to the tool's
                base directory.

        Returns:
            On success::

                {"agent": str, "status": "success", "path": str, "message": str}

            On failure::

                {"agent": str, "status": "error", "path": str, "error": str}
        """
        logger.info("[%s] Deleting file: %s", self.agent_name, relative_path)
        result = self._file_manager.delete_file(relative_path)
        return self._enrich(result)

    def list_project_files(self, relative_dir: str = "") -> dict:
        """List all files recursively under a directory.

        Args:
            relative_dir: Directory to inspect, relative to the tool's base
                directory. Defaults to the base directory itself when empty.

        Returns:
            On success::

                {
                    "agent":     str,
                    "status":    "success",
                    "directory": str,        # absolute path of the listed dir
                    "files":     list[str],  # paths relative to base_path, sorted
                }

            On failure::

                {"agent": str, "status": "error", "directory": str, "error": str}
        """
        logger.info("[%s] Listing files in: %r", self.agent_name, relative_dir or "(base)")
        result = self._file_manager.list_files(relative_dir)
        return self._enrich(result)

    def setup_project_structure(self, structure: dict[str, str]) -> dict:
        """Create multiple files in a single call from a path-to-content mapping.

        Iterates over every entry in *structure*, calling
        :meth:`create_project_file` for each one. The operation is
        best-effort: failures for individual files are collected in ``failed``
        without stopping the remaining creates.

        Args:
            structure: Mapping of relative file paths to their text content,
                e.g.::

                    {
                        "src/__init__.py": "",
                        "src/app.py": "# entry point",
                        "README.md": "# My Project",
                    }

        Returns:
            ::

                {
                    "agent":   str,
                    "status":  "success" | "partial" | "error",
                    "created": list[str],  # paths that were created successfully
                    "failed":  list[dict], # [{"path": str, "error": str}, ...]
                }

            ``status`` is ``"success"`` when all files were created,
            ``"partial"`` when at least one failed, and ``"error"`` when all
            failed or *structure* was empty.
        """
        logger.info(
            "[%s] Setting up project structure (%d file(s)).",
            self.agent_name, len(structure),
        )
        created: list[str] = []
        failed: list[dict] = []

        for path, content in structure.items():
            result = self._file_manager.create_file(path, content)
            if result["status"] == "success":
                created.append(result["path"])
                logger.info("[%s] Created: %s", self.agent_name, path)
            else:
                failed.append({"path": path, "error": result.get("error", "unknown error")})
                logger.warning("[%s] Failed to create: %s — %s", self.agent_name, path, result.get("error"))

        if failed and not created:
            status = "error"
        elif failed:
            status = "partial"
        else:
            status = "success"

        logger.info(
            "[%s] setup_project_structure done — %d created, %d failed.",
            self.agent_name, len(created), len(failed),
        )
        return {
            "agent": self.agent_name,
            "status": status,
            "created": created,
            "failed": failed,
        }


# Module-level singleton — import this directly instead of instantiating FileAgent yourself.
file_agent = FileAgent("FileAgent")
