import json
import os
import subprocess
import sys
import asyncio
import httpx
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DebuggingAgent:
    """
    Automatically debugs generated projects by:
    1. Running the backend and capturing errors
    2. Running the frontend build and capturing errors
    3. Sending errors to Claude API for fixes
    4. Applying fixes and retrying until success
    5. Returns URLs of running app when successful
    """

    def __init__(self):
        self.agent_name = "DebuggingAgent"
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.model = "claude-sonnet-4-20250514"
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.max_attempts = 3
        logger.info("%s initialised.", self.agent_name)

    async def debug_project(
        self,
        project_files: list[dict],
        project_name: str,
        db_session=None,
    ) -> dict:
        """
        Main debug loop:
        1. Write files to temp directory
        2. Try to run backend - capture errors
        3. Try to build frontend - capture errors
        4. If errors exist - ask Claude to fix
        5. Update files in DB with fixed versions
        6. Repeat up to max_attempts times
        Returns: {status, backend_errors, frontend_errors, fixes_applied, fixed_files}
        """
        logger.info("%s starting debug for project: %s", self.agent_name, project_name)

        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix=f"nexusflow_{project_name}_"))

        try:
            self._write_files(project_files, temp_dir)

            all_errors: list[str] = []
            fixed_files = list(project_files)
            fixes_applied = 0
            attempt = 1

            for attempt in range(1, self.max_attempts + 1):
                logger.info("%s attempt %d/%d", self.agent_name, attempt, self.max_attempts)

                self._write_files(fixed_files, temp_dir)

                backend_errors = self._check_backend_syntax(temp_dir)
                frontend_errors = self._check_frontend_build(temp_dir)

                all_errors = backend_errors + frontend_errors

                if not all_errors:
                    logger.info("%s no errors found on attempt %d", self.agent_name, attempt)
                    break

                logger.info(
                    "%s found %d error(s), asking Claude to fix...",
                    self.agent_name, len(all_errors),
                )

                fix_result = await self._fix_errors_with_claude(
                    fixed_files, all_errors, project_name
                )

                if fix_result["status"] == "success" and fix_result["files"]:
                    fixed_map = {f["path"]: f["content"] for f in fix_result["files"]}
                    fixed_files = [
                        {**f, "content": fixed_map.get(f["path"], f["content"])}
                        for f in fixed_files
                    ]
                    fixes_applied += len(fix_result["files"])
                    logger.info("%s applied %d fix(es)", self.agent_name, len(fix_result["files"]))
                else:
                    logger.warning("%s Claude fix failed on attempt %d", self.agent_name, attempt)
                    break

            return {
                "status": "success" if not all_errors else "partial",
                "fixes_applied": fixes_applied,
                "remaining_errors": all_errors,
                "fixed_files": fixed_files,
                "attempts": attempt,
            }

        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _write_files(self, files: list[dict], base_dir: Path) -> None:
        """Write project files to a directory."""
        for f in files:
            file_path = base_dir / f["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                file_path.write_text(f["content"], encoding="utf-8")
            except Exception as e:
                logger.warning("%s could not write %s: %s", self.agent_name, f["path"], e)

    def _check_backend_syntax(self, project_dir: Path) -> list[str]:
        """Check Python files for syntax errors."""
        errors: list[str] = []
        backend_dir = project_dir / "backend"
        if not backend_dir.exists():
            return errors

        for py_file in backend_dir.rglob("*.py"):
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(py_file)],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode != 0:
                    errors.append(
                        f"Syntax error in {py_file.relative_to(project_dir)}: {result.stderr}"
                    )
            except Exception as e:
                errors.append(f"Could not check {py_file.name}: {e}")

        return errors

    def _check_frontend_build(self, project_dir: Path) -> list[str]:
        """Check TypeScript files for type errors using tsc."""
        errors: list[str] = []
        frontend_dir = project_dir / "frontend"
        if not frontend_dir.exists():
            return errors

        if not (frontend_dir / "node_modules").exists():
            return []  # Can't check without installed dependencies

        try:
            result = subprocess.run(
                ["npm", "run", "build", "--", "--no-emit"],
                capture_output=True, text=True, timeout=60,
                cwd=str(frontend_dir),
            )
            if result.returncode != 0:
                error_lines = [
                    line for line in result.stderr.split("\n")
                    if "error TS" in line or "ERROR in" in line
                ][:10]
                errors.extend(error_lines)
        except Exception as e:
            logger.warning("%s frontend check failed: %s", self.agent_name, e)

        return errors

    async def _fix_errors_with_claude(
        self,
        files: list[dict],
        errors: list[str],
        project_name: str,
    ) -> dict:
        """Ask Claude to fix the errors in the project files."""
        if not self.api_key:
            return {"status": "error", "files": []}

        error_text = "\n".join(errors)

        relevant_files: list[dict] = []
        for f in files:
            for error in errors:
                if f["path"] in error or Path(f["path"]).name in error:
                    relevant_files.append(f)
                    break

        if not relevant_files:
            relevant_files = files[:5]

        files_context = "\n\n".join([
            f"### {f['path']}\n```\n{f['content'][:3000]}\n```"
            for f in relevant_files
        ])

        system_prompt = (
            "You are an expert debugger. Fix the errors in the provided code files.\n"
            "Return ONLY a valid JSON array: [{\"path\": str, \"content\": str}]\n"
            "Only include files that need to be fixed.\n"
            "Complete file content only — no truncation, no placeholders, no markdown fences in content."
        )

        user_prompt = f"""Project: {project_name}

Errors found:
{error_text}

Files to fix:
{files_context}

Fix all errors. Return JSON array of fixed files only."""

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    self.api_url,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 6000,
                        "messages": [{"role": "user", "content": user_prompt}],
                        "system": system_prompt,
                    },
                )

                if response.status_code != 200:
                    logger.error("%s API error: %s", self.agent_name, response.text)
                    return {"status": "error", "files": []}

                data = response.json()
                raw = data["content"][0]["text"].strip()

                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip().rstrip("```").strip()

                fixed_files = json.loads(raw)
                return {"status": "success", "files": fixed_files}

        except json.JSONDecodeError as e:
            logger.error("%s JSON parse error: %s", self.agent_name, e)
            return {"status": "error", "files": []}
        except Exception as e:
            logger.exception("%s fix error: %s", self.agent_name, e)
            return {"status": "error", "files": []}


debugging_agent = DebuggingAgent()
