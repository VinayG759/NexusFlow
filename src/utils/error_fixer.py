"""Error fixer utility for NexusFlow.

Uses an LLM to automatically fix syntax and logic errors in generated project
files, then writes the corrected code back to disk.

Usage::

    from src.utils.error_fixer import error_fixer

    result = await error_fixer.fix_file("weather-app/backend/routes.py", code, errors)
    result = await error_fixer.fix_project_errors(invalid_files, project_files)
"""

from pathlib import Path

from src.agents.file_agent import file_agent
from src.tools.api_connector import api_connector
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert Python/TypeScript engineer. Fix the errors in the provided code. "
    "Return ONLY the fixed raw code with no explanation or markdown fences."
)


class ErrorFixer:
    """Fixes errors in generated project files using an LLM.

    Wraps :func:`~src.tools.api_connector.api_connector.call_groq` and
    :class:`~src.agents.file_agent.FileAgent` to send broken code to Groq,
    receive a corrected version, and write it back to disk.

    Attributes:
        agent_name: Identifier used in log messages.
    """

    def __init__(self, agent_name: str = "ErrorFixer") -> None:
        """Initialise the ErrorFixer.

        Args:
            agent_name: Label used in log output.
        """
        self.agent_name = agent_name
        logger.info("%s initialised.", self.agent_name)

    # ── Public methods ────────────────────────────────────────────────────────

    async def fix_file(
        self,
        file_path: str,
        code: str,
        errors: list[str],
        context: str = "",
    ) -> dict:
        """Send a broken file to the LLM and save the fixed version.

        Args:
            file_path: Path to the file being fixed (used for logging and saving).
            code: Current (broken) source code.
            errors: List of error messages reported by the validator.
            context: Optional extra context, e.g. content of files this file
                imports from.

        Returns:
            On success::

                {
                    "status":           "success",
                    "file_path":        str,
                    "fixed_code":       str,
                    "errors_addressed": list[str],
                }

            On failure::

                {
                    "status":  "error",
                    "error":   str,
                    "file_path": str,
                }
        """
        logger.info("%s: fixing %r (%d error(s))", self.agent_name, file_path, len(errors))

        errors_block = "\n".join(f"- {e}" for e in errors)
        user_prompt = (
            f"File: {file_path}\n\n"
            f"Current code:\n{code}\n\n"
            f"Errors to fix:\n{errors_block}"
        )
        if context:
            user_prompt += f"\n\nContext (related files):\n{context}"

        llm_result = await api_connector.call_groq(
            prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
        )

        if llm_result.get("status") != "success":
            error_msg = llm_result.get("error", "Unknown LLM error.")
            logger.error("%s: LLM call failed for %r: %s", self.agent_name, file_path, error_msg)
            return {"status": "error", "error": error_msg, "file_path": file_path}

        fixed_code = self._clean_code(llm_result["content"])

        save_result = file_agent.update_project_file(file_path, fixed_code)
        if save_result.get("status") != "success":
            error_msg = save_result.get("error", "File save failed.")
            logger.error("%s: save failed for %r: %s", self.agent_name, file_path, error_msg)
            return {"status": "error", "error": error_msg, "file_path": file_path}

        logger.info("%s: successfully fixed and saved %r", self.agent_name, file_path)
        return {
            "status": "success",
            "file_path": file_path,
            "fixed_code": fixed_code,
            "errors_addressed": errors,
        }

    async def fix_project_errors(
        self,
        invalid_files: list[dict],
        project_files: dict,
    ) -> dict:
        """Fix all invalid files reported by the code validator.

        Args:
            invalid_files: List of ``{"file_path": str, "errors": list[str]}``
                dicts as returned by
                :meth:`~src.utils.code_validator.CodeValidator.validate_project`.
            project_files: Mapping of ``{file_path: source_code}`` for every
                file in the project, used to build per-file context.

        Returns:
            ::

                {
                    "status": "success" | "partial" | "failed",
                    "fixed":  list[str],   # file paths that were fixed
                    "failed": list[str],   # file paths that could not be fixed
                }

            ``"success"`` — all files fixed; ``"partial"`` — some fixed;
            ``"failed"`` — none fixed (or no files to fix).
        """
        logger.info(
            "%s: fixing %d invalid file(s)", self.agent_name, len(invalid_files)
        )

        fixed: list[str] = []
        failed: list[str] = []

        for entry in invalid_files:
            file_path: str = entry["file_path"]
            errors: list[str] = entry.get("errors", [])

            code = project_files.get(file_path, "")
            if not code:
                try:
                    code = Path(file_path).read_text(encoding="utf-8")
                except OSError:
                    logger.warning("%s: cannot read %r — skipping", self.agent_name, file_path)
                    failed.append(file_path)
                    continue

            context = self._build_context(file_path, project_files)

            result = await self.fix_file(file_path, code, errors, context)
            if result["status"] == "success":
                fixed.append(file_path)
            else:
                failed.append(file_path)

        if not invalid_files:
            overall = "success"
        elif not failed:
            overall = "success"
        elif not fixed:
            overall = "failed"
        else:
            overall = "partial"

        logger.info(
            "%s: project fix done — status=%s fixed=%d failed=%d",
            self.agent_name, overall, len(fixed), len(failed),
        )
        return {"status": overall, "fixed": fixed, "failed": failed}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _clean_code(self, code: str) -> str:
        """Strip markdown fences from LLM output.

        Args:
            code: Raw LLM response text.

        Returns:
            Source code with leading/trailing fences removed.
        """
        code = code.strip()
        for fence in ("```python", "```typescript", "```jsx", "```tsx", "```"):
            if code.startswith(fence):
                code = code[len(fence):]
                break
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    def _build_context(self, file_path: str, project_files: dict) -> str:
        """Build a context string from files in the same directory.

        Args:
            file_path: The file being fixed.
            project_files: All available project files.

        Returns:
            A string containing the content of sibling files, or an empty
            string if none are found.
        """
        target_dir = str(Path(file_path).parent)
        parts: list[str] = []
        for path, content in project_files.items():
            if path == file_path:
                continue
            if str(Path(path).parent) == target_dir:
                parts.append(f"### {path}\n{content}")
        return "\n\n".join(parts)


# Module-level singleton — import this directly instead of instantiating ErrorFixer yourself.
error_fixer = ErrorFixer()
