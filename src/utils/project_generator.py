"""Project generator utility for NexusFlow.

Replaces the isolated code-generation approach with a connected, full-project
generator that plans a complete file structure in one LLM call and then
generates each file with full awareness of the project context and its
dependencies.

Usage::

    from src.utils.project_generator import project_generator

    result = await project_generator.generate_project(
        "Build a weather dashboard with a FastAPI backend and React+TypeScript frontend"
    )
"""

import json

from src.agents.file_agent import file_agent
from src.tools.api_connector import api_connector
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PLAN_SYSTEM_PROMPT = """\
You are a senior software architect designing a full-stack web application.

Your task is to produce a complete file structure plan as a single JSON object.
Rules:
1. Return ONLY a valid JSON object — no explanation, no markdown, no backticks.
2. Every file path must be relative to the project root.
3. "depends_on" must list paths of other files in the same plan that this file \
imports from or builds upon. Use an empty list if there are no dependencies.
4. Use FastAPI for the backend and React + TypeScript + TailwindCSS for the frontend.
5. Never suggest Flask, Django, or MongoDB.

The JSON must follow this exact shape:
{
  "project_name": "<slug>",
  "description": "<one sentence>",
  "backend": {
    "framework": "fastapi",
    "files": [
      {"path": "<relative path>", "description": "<what this file does>", "depends_on": []}
    ]
  },
  "frontend": {
    "framework": "react-typescript",
    "files": [
      {"path": "<relative path>", "description": "<what this file does>", "depends_on": []}
    ]
  },
  "shared_types": ["<TypeScript type names shared between front and back>"],
  "env_variables": ["<ENV_VAR_NAME>"]
}\
"""

_FILE_SYSTEM_PROMPT = """\
You are a senior software engineer generating production-ready code for a full-stack project.

Rules:
1. Return ONLY raw code — no explanation, no markdown, no triple backticks.
2. Use FastAPI + SQLAlchemy + asyncpg for backend files.
3. Use React + TypeScript + TailwindCSS for frontend files.
4. Always use async/await in FastAPI route handlers.
5. Include proper error handling, type annotations, and imports.
6. Never use Flask, Django, or MongoDB.\
"""


class ProjectGenerator:
    """Orchestrates full-project generation with context-aware file production.

    Rather than generating files in isolation, :class:`ProjectGenerator` first
    plans the complete file structure in a single LLM call (so paths,
    descriptions, and dependency edges are internally consistent) and then
    generates each file in dependency order, passing the content of dependency
    files into the prompt of every downstream file.

    Attributes:
        agent_name: Human-readable identifier used in logs and result dicts.
    """

    def __init__(self, agent_name: str = "ProjectGenerator") -> None:
        """Initialise the ProjectGenerator.

        Args:
            agent_name: Name used in log messages and result dicts.
        """
        self.agent_name = agent_name
        logger.info("ProjectGenerator '%s' initialised.", agent_name)

    # ── Public methods ────────────────────────────────────────────────────────

    async def plan_structure(self, problem_statement: str, tech_stack: dict) -> dict:
        """Plan the complete file structure for a project via a single LLM call.

        Sends the problem statement and tech-stack constraints to the LLM and
        expects a single JSON object back describing every file to be generated,
        together with inter-file dependency edges.

        Args:
            problem_statement: Plain-language description of what the project
                should do (e.g. ``"Weather dashboard with live data"``).
            tech_stack: Dict of additional tech constraints to include in the
                prompt (e.g. ``{"database": "PostgreSQL", "auth": "JWT"}``).

        Returns:
            On success — the parsed JSON structure dict::

                {
                    "project_name": str,
                    "description":  str,
                    "backend":  {"framework": str, "files": [...]},
                    "frontend": {"framework": str, "files": [...]},
                    "shared_types":   [str],
                    "env_variables":  [str],
                }

            On failure::

                {
                    "status": "error",
                    "error":  str,
                }
        """
        logger.info(
            "[%s] Planning project structure for: %r", self.agent_name, problem_statement[:80],
        )

        tech_stack_text = "\n".join(f"  - {k}: {v}" for k, v in tech_stack.items())
        user_prompt = (
            f"Problem statement: {problem_statement}\n\n"
            f"Additional tech-stack requirements:\n{tech_stack_text or '  (none)'}"
        )

        try:
            result = await api_connector.call_groq(
                prompt=user_prompt,
                system_prompt=_PLAN_SYSTEM_PROMPT,
            )
            if result["status"] != "success":
                raise RuntimeError(result.get("error", "LLM call returned non-success status."))

            raw = result["content"].strip()
            # Defensively strip any accidental markdown fences around the JSON.
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]

            structure = json.loads(raw)
            if not isinstance(structure, dict):
                raise ValueError("LLM returned a non-dict JSON value.")

            logger.info(
                "[%s] Structure planned — project=%r backend_files=%d frontend_files=%d",
                self.agent_name,
                structure.get("project_name", "?"),
                len(structure.get("backend", {}).get("files", [])),
                len(structure.get("frontend", {}).get("files", [])),
            )
            return structure

        except json.JSONDecodeError as exc:
            logger.error("[%s] Failed to parse LLM JSON response: %s", self.agent_name, exc)
            return {"status": "error", "error": f"JSON parse error: {exc}"}
        except Exception as exc:
            logger.exception("[%s] plan_structure failed: %s", self.agent_name, exc)
            return {"status": "error", "error": str(exc)}

    async def generate_file(
        self,
        file_info: dict,
        project_context: dict,
        generated_files: dict,
    ) -> dict:
        """Generate a single file with full awareness of the project context.

        Builds a prompt that includes the project name and description, the
        tech stack, the specific file's path and purpose, and the full content
        of every file it depends on — so the LLM can produce code that imports
        from and builds upon already-generated siblings correctly.

        Args:
            file_info: Dict with ``path`` (str), ``description`` (str), and
                ``depends_on`` (list[str]) keys, as produced by
                :meth:`plan_structure`.
            project_context: Dict with ``project_name``, ``description``, and
                ``tech_stack`` keys that describe the overall project.
            generated_files: Mapping of ``{relative_path: code_string}`` for
                every file already generated in this session. Only files listed
                in ``file_info["depends_on"]`` are included in the prompt.

        Returns:
            On success::

                {
                    "status": "success",
                    "path":   str,   # relative file path
                    "code":   str,   # cleaned, ready-to-save source code
                }

            On failure::

                {
                    "status": "error",
                    "path":   str,
                    "error":  str,
                }
        """
        path = file_info.get("path", "unknown")
        description = file_info.get("description", "")
        depends_on: list[str] = file_info.get("depends_on") or []

        logger.info("[%s] Generating file: %r", self.agent_name, path)

        # Build dependency context section.
        dep_sections: list[str] = []
        for dep_path in depends_on:
            dep_code = generated_files.get(dep_path)
            if dep_code:
                dep_sections.append(
                    f"--- {dep_path} ---\n{dep_code}\n--- end {dep_path} ---"
                )
            else:
                logger.warning(
                    "[%s] Dependency %r not found in generated_files for %r",
                    self.agent_name, dep_path, path,
                )

        tech_stack = project_context.get("tech_stack", {})
        tech_stack_text = "\n".join(f"  - {k}: {v}" for k, v in tech_stack.items())

        dep_context = (
            "\n\nFiles this file depends on (use their exact imports/types):\n"
            + "\n\n".join(dep_sections)
            if dep_sections
            else ""
        )

        user_prompt = (
            f"Project: {project_context.get('project_name', 'unknown')}\n"
            f"Description: {project_context.get('description', '')}\n"
            f"Tech stack:\n{tech_stack_text or '  (standard full-stack)'}\n\n"
            f"Generate the file at path: {path}\n"
            f"Purpose: {description}"
            f"{dep_context}"
        )

        try:
            result = await api_connector.call_groq(
                prompt=user_prompt,
                system_prompt=_FILE_SYSTEM_PROMPT,
            )
            if result["status"] != "success":
                raise RuntimeError(result.get("error", "LLM call returned non-success status."))

            code = self._clean_code(result["content"])
            logger.info("[%s] File generated successfully: %r (%d chars)", self.agent_name, path, len(code))
            return {"status": "success", "path": path, "code": code}

        except Exception as exc:
            logger.exception("[%s] generate_file failed for %r: %s", self.agent_name, path, exc)
            return {"status": "error", "path": path, "error": str(exc)}

    async def generate_project(self, problem_statement: str) -> dict:
        """Orchestrate full project generation from a single problem statement.

        Steps:
        1. Calls :meth:`plan_structure` to obtain the complete file plan.
        2. Merges backend and frontend file lists into a single ordered sequence.
        3. Generates files in dependency order using :meth:`generate_file`,
           accumulating produced code into a shared context dict.
        4. Saves each successfully generated file via
           :func:`~src.agents.file_agent.FileAgent.create_project_file`.

        Args:
            problem_statement: Plain-language description of what to build.

        Returns:
            ::

                {
                    "status":          "success" | "partial" | "error",
                    "project_name":    str,
                    "files_generated": list[str],   # paths saved successfully
                    "files_failed":    list[str],   # paths that failed
                    "structure":       dict,         # the raw plan from plan_structure()
                }
        """
        logger.info(
            "[%s] Starting full project generation for: %r",
            self.agent_name, problem_statement[:80],
        )

        tech_stack = {
            "backend": "FastAPI + SQLAlchemy + asyncpg + PostgreSQL",
            "frontend": "React + TypeScript + TailwindCSS",
            "auth": "JWT",
            "deployment": "Docker + docker-compose",
        }

        # Step 1 — plan the structure.
        structure = await self.plan_structure(problem_statement, tech_stack)
        if structure.get("status") == "error":
            return {
                "status": "error",
                "project_name": "",
                "files_generated": [],
                "files_failed": [],
                "structure": structure,
            }

        project_name = structure.get("project_name", "nexusflow-project")
        project_context = {
            "project_name": project_name,
            "description": structure.get("description", ""),
            "tech_stack": tech_stack,
        }

        # Step 2 — collect all files; backend first so frontend can depend on them.
        all_files: list[dict] = (
            structure.get("backend", {}).get("files", [])
            + structure.get("frontend", {}).get("files", [])
        )

        # Step 3 — generate in dependency order (multi-pass until no progress).
        generated_files: dict[str, str] = {}
        files_generated: list[str] = []
        files_failed: list[str] = []
        pending = list(all_files)

        while pending:
            advanced = False
            still_pending: list[dict] = []

            for file_info in pending:
                path = file_info.get("path", "")
                depends_on: list[str] = file_info.get("depends_on") or []
                unmet = [d for d in depends_on if d not in generated_files]

                if unmet:
                    logger.debug("[%s] %r waiting on: %s", self.agent_name, path, unmet)
                    still_pending.append(file_info)
                    continue

                result = await self.generate_file(file_info, project_context, generated_files)

                if result["status"] == "success":
                    code = result["code"]
                    generated_files[path] = code

                    # Step 4 — save to disk.
                    save_path = f"{project_name}/{path}"
                    save_result = file_agent.create_project_file(save_path, code)
                    if save_result.get("status") == "success":
                        files_generated.append(save_path)
                        logger.info("[%s] Saved %r", self.agent_name, save_path)
                    else:
                        logger.warning(
                            "[%s] Generated but failed to save %r: %s",
                            self.agent_name, save_path, save_result.get("error"),
                        )
                        files_failed.append(save_path)
                else:
                    files_failed.append(path)
                    logger.warning("[%s] Failed to generate %r", self.agent_name, path)

                advanced = True

            pending = still_pending

            if not advanced and pending:
                blocked = [f.get("path", "?") for f in pending]
                logger.error(
                    "[%s] Unresolvable dependencies — blocking: %s", self.agent_name, blocked,
                )
                files_failed.extend(blocked)
                break

        if files_failed and not files_generated:
            overall_status = "error"
        elif files_failed:
            overall_status = "partial"
        else:
            overall_status = "success"

        logger.info(
            "[%s] Project generation complete — status=%s generated=%d failed=%d",
            self.agent_name, overall_status, len(files_generated), len(files_failed),
        )
        return {
            "status": overall_status,
            "project_name": project_name,
            "files_generated": files_generated,
            "files_failed": files_failed,
            "structure": structure,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _clean_code(self, code: str) -> str:
        """Strip markdown fencing and surrounding whitespace from LLM output."""
        code = code.strip()
        for fence in ("```python", "```typescript", "```tsx", "```jsx", "```"):
            if code.startswith(fence):
                code = code[len(fence):]
                break
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()


# Module-level singleton — import this directly instead of instantiating ProjectGenerator yourself.
project_generator = ProjectGenerator()
