"""Fix-loop utility for NexusFlow.

Orchestrates a validate → fix → validate cycle on a project directory,
retrying up to a configurable maximum until all Python files pass syntax
validation or the retry budget is exhausted.

Usage::

    from src.utils.fix_loop import fix_loop

    result = await fix_loop.run("weather-app/backend")
"""

from pathlib import Path

from src.agents.file_agent import file_agent
from src.utils.code_validator import code_validator
from src.utils.error_fixer import error_fixer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FixLoop:
    """Iteratively validates and fixes a project directory.

    Calls :func:`~src.utils.code_validator.code_validator.validate_project`
    to discover broken files, then delegates repairs to
    :func:`~src.utils.error_fixer.error_fixer.fix_project_errors`, and
    repeats until every file is valid or ``max_retries`` is reached.

    Attributes:
        max_retries: Maximum validate–fix cycles before giving up.
    """

    def __init__(self, max_retries: int = 3) -> None:
        """Initialise the FixLoop.

        Args:
            max_retries: How many fix attempts to make before returning a
                partial or failed result. Defaults to 3.
        """
        self.max_retries = max_retries
        logger.info("FixLoop initialised (max_retries=%d).", self.max_retries)

    # ── Public methods ────────────────────────────────────────────────────────

    async def run(self, project_path: str) -> dict:
        """Run the validate → fix → validate loop on a project directory.

        For each attempt:

        1. Validate all ``*.py`` files under *project_path*.
        2. If every file is valid, return ``"success"`` immediately.
        3. Otherwise, read the content of each invalid file from disk.
        4. Pass them to :meth:`~src.utils.error_fixer.ErrorFixer.fix_project_errors`.
        5. Log the outcome and repeat.

        After *max_retries* attempts the loop stops and performs one final
        validation to determine the return status.

        Args:
            project_path: Root directory of the project to validate and fix.

        Returns:
            ::

                {
                    "status":               "success" | "partial" | "failed",
                    "project_path":         str,
                    "attempts":             int,
                    "final_valid_files":    list[str],
                    "final_invalid_files":  list[dict],
                    "fixes_applied":        int,
                }

            ``"success"`` — all files valid at end of loop;
            ``"partial"`` — some files still invalid after max retries;
            ``"failed"`` — no files could be fixed (or project has no Python files).
        """
        logger.info(
            "FixLoop.run: starting on %r (max_retries=%d)", project_path, self.max_retries
        )

        fixes_applied: int = 0
        attempt: int = 0

        for attempt in range(1, self.max_retries + 1):
            logger.info(
                "FixLoop attempt %d/%d — validating %r", attempt, self.max_retries, project_path
            )

            validation = code_validator.validate_project(project_path)
            invalid_files: list[dict] = validation.get("invalid_files", [])
            valid_files: list[str] = validation.get("valid_files", [])

            if not invalid_files:
                logger.info(
                    "FixLoop attempt %d: all %d file(s) valid — done.",
                    attempt, len(valid_files),
                )
                return {
                    "status": "success",
                    "project_path": project_path,
                    "attempts": attempt,
                    "final_valid_files": valid_files,
                    "final_invalid_files": [],
                    "fixes_applied": fixes_applied,
                }

            logger.info(
                "FixLoop attempt %d: %d invalid file(s) — reading source and calling error_fixer",
                attempt, len(invalid_files),
            )

            project_files = self._read_project_files(project_path)

            fix_result = await error_fixer.fix_project_errors(invalid_files, project_files)
            newly_fixed: list[str] = fix_result.get("fixed", [])
            fixes_applied += len(newly_fixed)

            logger.info(
                "FixLoop attempt %d: fixed=%d failed=%d (total fixes so far=%d)",
                attempt,
                len(newly_fixed),
                len(fix_result.get("failed", [])),
                fixes_applied,
            )

            if not newly_fixed:
                logger.warning(
                    "FixLoop attempt %d: no files were fixed — stopping early.", attempt
                )
                break

        # Final validation after loop ends or early break
        final = code_validator.validate_project(project_path)
        final_invalid: list[dict] = final.get("invalid_files", [])
        final_valid: list[str] = final.get("valid_files", [])

        if not final_invalid:
            overall = "success"
        elif final_valid:
            overall = "partial"
        else:
            overall = "failed"

        logger.info(
            "FixLoop finished — status=%s attempts=%d fixes_applied=%d valid=%d invalid=%d",
            overall, attempt, fixes_applied, len(final_valid), len(final_invalid),
        )
        return {
            "status": overall,
            "project_path": project_path,
            "attempts": attempt,
            "final_valid_files": final_valid,
            "final_invalid_files": final_invalid,
            "fixes_applied": fixes_applied,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _read_project_files(self, project_path: str) -> dict:
        """Read all Python files under *project_path* into a dict.

        Args:
            project_path: Root directory to scan.

        Returns:
            Mapping of ``{file_path_str: source_code}`` for every ``*.py``
            file that can be read. Files that cannot be decoded are skipped
            with a warning.
        """
        root = Path(project_path)
        project_files: dict = {}
        for py_file in sorted(root.rglob("*.py")):
            path_str = str(py_file)
            try:
                project_files[path_str] = py_file.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("FixLoop: could not read %r: %s", path_str, exc)
        return project_files


# Module-level singleton — import this directly instead of instantiating FixLoop yourself.
fix_loop = FixLoop()
