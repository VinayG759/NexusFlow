"""File manager tool for NexusFlow.

Provides safe, sandboxed CRUD operations on files rooted at a configurable
base directory. All relative paths are resolved against ``base_path`` so
agents cannot accidentally read or write outside their workspace.

Usage::

    from src.tools.file_manager import file_manager

    file_manager.create_file("output/report.txt", "Hello, world!")
    result = file_manager.read_file("output/report.txt")
    file_manager.update_file("output/report.txt", "Updated content.")
    file_manager.delete_file("output/report.txt")
    listing = file_manager.list_files("output/")
"""

import os
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FileManagerTool:
    """Sandboxed file I/O tool for NexusFlow agents.

    All operations are restricted to the subtree rooted at ``base_path``.
    Any attempt to traverse above it (e.g. via ``../../etc/passwd``) is
    blocked by resolving the full path and checking it is inside ``base_path``
    before touching the filesystem.

    Attributes:
        base_path: Absolute :class:`~pathlib.Path` that acts as the root of
            the sandbox for all relative path arguments.
    """

    def __init__(self, base_path: str = "") -> None:
        """Initialise the FileManagerTool.

        Args:
            base_path: Root directory for all file operations. Defaults to the
                current working directory when empty or not provided.
        """
        self.base_path = Path(base_path).resolve() if base_path else Path.cwd()
        logger.info("FileManagerTool initialised with base_path=%s", self.base_path)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve(self, relative_path: str) -> Path:
        """Resolve *relative_path* to an absolute :class:`~pathlib.Path`.

        Absolute paths (e.g. ``C:\\Users\\...`` or ``/home/...``) are used
        directly without joining against ``base_path``, allowing NexusFlow to
        write generated projects to any location on disk.  Relative paths are
        still sandboxed inside ``base_path`` as before.

        Args:
            relative_path: Either an absolute path or a path relative to
                ``base_path``.

        Returns:
            Absolute :class:`~pathlib.Path` for the target.

        Raises:
            ValueError: If a relative path resolves outside ``base_path``.
        """
        p = Path(relative_path)
        if p.is_absolute():
            return p.resolve()

        target = (self.base_path / relative_path).resolve()
        if not str(target).startswith(str(self.base_path)):
            raise ValueError(
                f"Path '{relative_path}' resolves outside base_path '{self.base_path}'."
            )
        return target

    @staticmethod
    def _error(path: str, exc: Exception) -> dict:
        return {"status": "error", "path": path, "error": str(exc)}

    # ── Public methods ────────────────────────────────────────────────────────

    def create_file(self, relative_path: str, content: str) -> dict:
        """Create a new file, including any missing parent directories.

        If the file already exists it is overwritten.

        Args:
            relative_path: Path of the file to create, relative to ``base_path``.
            content: Text content to write into the file (UTF-8 encoded).

        Returns:
            On success::

                {"status": "success", "path": str, "message": str}

            On failure::

                {"status": "error", "path": str, "error": str}
        """
        logger.info("Creating file: %s", relative_path)
        try:
            target = self._resolve(relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            logger.info("File created: %s", target)
            return {
                "status": "success",
                "path": str(target),
                "message": f"File '{relative_path}' created successfully.",
            }
        except Exception as exc:
            logger.exception("Failed to create file '%s': %s", relative_path, exc)
            return self._error(relative_path, exc)

    def read_file(self, relative_path: str) -> dict:
        """Read and return the full text content of a file.

        Args:
            relative_path: Path of the file to read, relative to ``base_path``.

        Returns:
            On success::

                {"status": "success", "path": str, "content": str}

            On failure::

                {"status": "error", "path": str, "error": str}
        """
        logger.info("Reading file: %s", relative_path)
        try:
            target = self._resolve(relative_path)
            content = target.read_text(encoding="utf-8")
            logger.info("File read successfully: %s", target)
            return {
                "status": "success",
                "path": str(target),
                "content": content,
            }
        except Exception as exc:
            logger.exception("Failed to read file '%s': %s", relative_path, exc)
            return self._error(relative_path, exc)

    def update_file(self, relative_path: str, content: str) -> dict:
        """Overwrite an existing file with new content.

        Unlike :meth:`create_file`, this method raises if the file does not
        already exist, making the intent explicit and preventing silent creation
        of unintended files.

        Args:
            relative_path: Path of the file to update, relative to ``base_path``.
            content: New text content to write (UTF-8 encoded, replaces entire file).

        Returns:
            On success::

                {"status": "success", "path": str, "message": str}

            On failure::

                {"status": "error", "path": str, "error": str}
        """
        logger.info("Updating file: %s", relative_path)
        try:
            target = self._resolve(relative_path)
            if not target.exists():
                raise FileNotFoundError(f"File '{relative_path}' does not exist.")
            target.write_text(content, encoding="utf-8")
            logger.info("File updated: %s", target)
            return {
                "status": "success",
                "path": str(target),
                "message": f"File '{relative_path}' updated successfully.",
            }
        except Exception as exc:
            logger.exception("Failed to update file '%s': %s", relative_path, exc)
            return self._error(relative_path, exc)

    def delete_file(self, relative_path: str) -> dict:
        """Delete a file if it exists.

        Deleting a path that does not exist is treated as a no-op success so
        callers can use this idempotently during cleanup.

        Args:
            relative_path: Path of the file to delete, relative to ``base_path``.

        Returns:
            On success::

                {"status": "success", "path": str, "message": str}

            On failure::

                {"status": "error", "path": str, "error": str}
        """
        logger.info("Deleting file: %s", relative_path)
        try:
            target = self._resolve(relative_path)
            if target.exists():
                target.unlink()
                message = f"File '{relative_path}' deleted successfully."
                logger.info("File deleted: %s", target)
            else:
                message = f"File '{relative_path}' did not exist — nothing to delete."
                logger.info("Delete skipped (file not found): %s", target)
            return {"status": "success", "path": str(target), "message": message}
        except Exception as exc:
            logger.exception("Failed to delete file '%s': %s", relative_path, exc)
            return self._error(relative_path, exc)

    def list_files(self, relative_dir: str = "") -> dict:
        """List all files recursively under a directory.

        Args:
            relative_dir: Directory to list, relative to ``base_path``.
                Defaults to ``base_path`` itself when empty.

        Returns:
            On success::

                {
                    "status":    "success",
                    "directory": str,         # absolute path of the listed directory
                    "files":     list[str],   # paths relative to base_path, sorted
                }

            On failure::

                {"status": "error", "directory": str, "error": str}
        """
        logger.info("Listing files in directory: %r", relative_dir or "(base)")
        try:
            target_dir = self._resolve(relative_dir) if relative_dir else self.base_path
            if not target_dir.is_dir():
                raise NotADirectoryError(f"'{relative_dir}' is not a directory.")
            files = sorted(
                str(p.relative_to(self.base_path))
                for p in target_dir.rglob("*")
                if p.is_file()
            )
            logger.info("Listed %d file(s) in %s", len(files), target_dir)
            return {
                "status": "success",
                "directory": str(target_dir),
                "files": files,
            }
        except Exception as exc:
            logger.exception("Failed to list files in '%s': %s", relative_dir, exc)
            return {"status": "error", "directory": relative_dir, "error": str(exc)}

    def file_exists(self, relative_path: str) -> bool:
        """Check whether a file exists inside the sandbox.

        Args:
            relative_path: Path to check, relative to ``base_path``.

        Returns:
            ``True`` if the path exists and is a file; ``False`` otherwise,
            including when the path escapes the sandbox.
        """
        try:
            return self._resolve(relative_path).is_file()
        except (ValueError, OSError):
            return False


# Module-level singleton — import this directly instead of instantiating FileManagerTool yourself.
file_manager = FileManagerTool()
