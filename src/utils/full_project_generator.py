"""Full project generator utility for NexusFlow.

Generates an entire project in a SINGLE LLM call — all files are returned at
once in one JSON response, guaranteeing that every import path, type, and
interface is consistent across the whole codebase before any file is written
to disk.

Usage::

    from src.utils.full_project_generator import full_project_generator

    result = await full_project_generator.generate(
        "Build a weather dashboard with a FastAPI backend and React+TypeScript frontend",
        options={"threejs": True, "gsap": False, "reactbits": False},
    )
"""

import json
from pathlib import Path

from src.agents.file_agent import file_agent
from src.config.settings import settings
from src.tools.api_connector import api_connector
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a senior full-stack software engineer.

Your task is to generate a COMPLETE, production-ready project in a single response.

Return ONLY a valid JSON object — no explanation, no markdown, no text outside the JSON.

The JSON must follow this exact structure:
{
  "project_name": "<lowercase-slug>",
  "description": "<one sentence describing the project>",
  "files": [
    {
      "path": "<relative path from project root>",
      "content": "<complete file content — no markdown fences, no placeholders>"
    }
  ],
  "setup_instructions": "<step-by-step plain text setup guide>",
  "env_variables": [
    {
      "name": "<ENV_VAR_NAME>",
      "description": "<what this variable is for>",
      "example": "<example value>"
    }
  ]
}

Rules you must follow without exception:
1. Every file must have correct imports that reference other files in THIS project by their exact paths.
2. Backend stack: FastAPI + SQLAlchemy (async) + asyncpg + PostgreSQL. Never use Flask, Django, or MongoDB.
3. Frontend stack: React 18 + TypeScript + TailwindCSS. Never use class components or JavaScript.
4. Use async/await throughout all FastAPI route handlers and SQLAlchemy queries.
5. Use SQLAlchemy 2.0 ORM style with Mapped and mapped_column.
6. Use FastAPI lifespan (asynccontextmanager) — never @app.on_event.
7. Always include these files at minimum:
   - backend/main.py
   - backend/database.py
   - backend/models.py
   - backend/routes.py
   - backend/schemas.py
   - backend/requirements.txt
   - frontend/package.json
   - frontend/tsconfig.json
   - frontend/src/App.tsx
   - frontend/src/index.tsx
   - frontend/src/styles/globals.css
   - .env.example
   - docker-compose.yml
   - README.md
8. File content must be complete — no "# TODO", no "..." ellipsis, no placeholder logic.
9. No markdown fences (no ```) anywhere inside file content strings.
10. No hardcoded credentials — use environment variables loaded via python-dotenv or process.env.
11. All TypeScript files must have proper type annotations — no implicit any.
12. The frontend must call the backend API at the URL from an environment variable (REACT_APP_API_URL).
13. Include proper CORS configuration in the FastAPI backend allowing the frontend origin.
14. docker-compose.yml must wire backend, frontend, and PostgreSQL together.
15. For PostgreSQL always use these exact credentials in .env and database.py:
     - username: postgres
     - password: vinay2004
     - host: localhost
     - port: 5432
     - database name: use the project_name slug (e.g. todo-app)
     - DATABASE_URL format: postgresql+asyncpg://postgres:vinay2004@localhost:5432/{project_name}
16. Never use placeholder credentials — always use the exact credentials above.
17. NEVER use fake, invented, or non-existent npm packages. Only use real, published packages from npmjs.com. Verified safe packages include: react, react-dom, react-router-dom, axios, typescript, tailwindcss, framer-motion, lucide-react, three, gsap, @types/react, @types/react-dom, @types/node, react-scripts, vite, @vitejs/plugin-react, zustand, react-query, date-fns, uuid, dotenv, cors, bcryptjs, jsonwebtoken.
18. Every generated frontend project must have 'skipLibCheck': true in tsconfig.json compilerOptions. This prevents TypeScript errors from third-party type definitions.
19. Every generated frontend tsconfig.json must use typescript version compatible settings:
    - 'target': 'es5' or 'ES2020'
    - 'lib': ['dom', 'dom.iterable', 'esnext']
    - 'skipLibCheck': true
    - 'esModuleInterop': true
    - 'allowSyntheticDefaultImports': true
20. For Three.js imports always add // @ts-ignore comment before the import line:
    // @ts-ignore
    import * as THREE from 'three';\
"""


class FullProjectGenerator:
    """Generates a complete full-stack project in a single LLM call.

    Unlike :class:`~src.utils.project_generator.ProjectGenerator` which makes
    one LLM call per file, :class:`FullProjectGenerator` generates every file
    in a single call. This guarantees that all imports, type definitions, API
    contracts, and environment variables are internally consistent before
    anything is written to disk.

    Attributes:
        agent_name: Human-readable identifier used in logs and result dicts.
    """

    def __init__(self, agent_name: str = "FullProjectGenerator") -> None:
        """Initialise the FullProjectGenerator.

        Args:
            agent_name: Name used in log messages and result dicts.
        """
        self.agent_name = agent_name
        logger.info("FullProjectGenerator '%s' initialised.", agent_name)

    # ── Public methods ────────────────────────────────────────────────────────

    async def generate(self, problem_statement: str, options: dict | None = None) -> dict:
        """Generate a complete project from a problem statement in one LLM call.

        Builds a user prompt from *problem_statement* and any enabled feature
        flags in *options*, then sends a single request to the Groq LLM with a
        strict system prompt that demands a complete JSON project structure.

        The returned JSON is parsed and every ``files`` entry is saved to disk
        via :func:`~src.agents.file_agent.FileAgent.create_project_file`.

        Args:
            problem_statement: Plain-language description of the project to
                build (e.g. ``"Weather dashboard with live OpenWeatherMap data"``).
            options: Optional dict of feature flags and settings:
                - ``threejs`` (bool): append a Three.js 3D visuals requirement.
                - ``gsap`` (bool): append a GSAP animation requirement.
                - ``reactbits`` (bool): append a ReactBits UI component requirement.
                - ``output_directory`` (str): base directory to save the project
                  under. Files are written to ``output_directory/project_name/path``.
                  When omitted, files are saved to ``project_name/path`` relative to
                  the NexusFlow working directory.

        Returns:
            On success or partial success::

                {
                    "status":             "success" | "partial" | "error",
                    "project_name":       str,
                    "output_path":        str,   # actual base path used for saving
                    "files_saved":        list[str],   # paths written to disk
                    "files_failed":       list[str],   # paths that failed to save
                    "setup_instructions": str,
                    "env_variables":      list[dict],  # [{name, description, example}]
                }

            On total failure (LLM error or JSON parse failure)::

                {
                    "status":       "error",
                    "project_name": "",
                    "files_saved":  [],
                    "files_failed": [],
                    "setup_instructions": "",
                    "env_variables": [],
                    "error":        str,
                }
        """
        opts = options or {}
        logger.info(
            "[%s] Starting full project generation for: %r options=%s",
            self.agent_name, problem_statement[:80], opts,
        )

        user_prompt = self._build_user_prompt(problem_statement, opts)

        # ── Single LLM call ───────────────────────────────────────────────────
        logger.info("[%s] Sending single LLM call for complete project.", self.agent_name)
        llm_result = await api_connector.call_groq(
            prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
        )

        if llm_result["status"] != "success":
            error_msg = llm_result.get("error", "LLM call returned non-success status.")
            logger.error("[%s] LLM call failed: %s", self.agent_name, error_msg)
            return {
                "status": "error",
                "project_name": "",
                "files_saved": [],
                "files_failed": [],
                "setup_instructions": "",
                "env_variables": [],
                "error": error_msg,
            }

        # ── Parse JSON response ───────────────────────────────────────────────
        try:
            clean_raw = self._clean_json(llm_result["content"])
            project = json.loads(clean_raw)
        except json.JSONDecodeError as exc:
            logger.error("[%s] Failed to parse LLM JSON response: %s", self.agent_name, exc)
            return {
                "status": "error",
                "project_name": "",
                "files_saved": [],
                "files_failed": [],
                "setup_instructions": "",
                "env_variables": [],
                "error": f"JSON parse error: {exc}",
            }

        project_name = project.get("project_name", "generated-project")
        files: list[dict] = project.get("files", [])
        setup_instructions: str = project.get("setup_instructions", "")
        env_variables: list[dict] = project.get("env_variables", [])

        # ── Resolve output base directory ─────────────────────────────────────
        output_base: str = opts.get("output_directory", "").strip() or settings.OUTPUT_DIRECTORY
        base_path = str(Path(output_base) / project_name)
        Path(base_path).mkdir(parents=True, exist_ok=True)
        logger.info(
            "[%s] LLM returned project=%r with %d file(s) — saving to: %r",
            self.agent_name, project_name, len(files), base_path,
        )

        # ── Save all files ────────────────────────────────────────────────────
        files_saved: list[str] = []
        files_failed: list[str] = []

        for file_entry in files:
            path: str = file_entry.get("path", "").strip()
            content: str = self._clean_content(file_entry.get("content", ""))

            if not path:
                logger.warning("[%s] Skipping file entry with empty path.", self.agent_name)
                continue

            save_path = f"{base_path}/{path}"
            save_result = file_agent.create_project_file(save_path, content)

            if save_result.get("status") == "success":
                files_saved.append(save_path)
                logger.info("[%s] Saved %r", self.agent_name, save_path)
            else:
                files_failed.append(save_path)
                logger.warning(
                    "[%s] Failed to save %r: %s",
                    self.agent_name, save_path, save_result.get("error"),
                )

        # ── Post-process: ensure skipLibCheck in all tsconfig.json files ────────
        import json as json_lib
        output_path = Path(base_path)
        for tsconfig in output_path.rglob("tsconfig.json"):
            if "node_modules" in str(tsconfig):
                continue
            try:
                content = json_lib.loads(tsconfig.read_text(encoding="utf-8"))
                if "compilerOptions" not in content:
                    content["compilerOptions"] = {}
                content["compilerOptions"]["skipLibCheck"] = True
                tsconfig.write_text(json_lib.dumps(content, indent=2), encoding="utf-8")
                logger.info("[%s] Post-processed tsconfig.json: %s", self.agent_name, tsconfig)
            except Exception as e:
                logger.warning("[%s] Could not patch tsconfig.json %s: %s", self.agent_name, tsconfig, e)

        # ── Determine overall status ──────────────────────────────────────────
        if files_failed and not files_saved:
            overall_status = "error"
        elif files_failed:
            overall_status = "partial"
        else:
            overall_status = "success"

        logger.info(
            "[%s] Generation complete — status=%s saved=%d failed=%d base_path=%r",
            self.agent_name, overall_status, len(files_saved), len(files_failed), base_path,
        )
        return {
            "status": overall_status,
            "project_name": project_name,
            "actual_output_path": base_path,
            "files_saved": files_saved,
            "files_failed": files_failed,
            "setup_instructions": setup_instructions,
            "env_variables": env_variables,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_user_prompt(self, problem_statement: str, options: dict) -> str:
        """Compose the user-facing prompt from the problem statement and options.

        Args:
            problem_statement: Core project description.
            options: Feature flags — ``threejs``, ``gsap``, ``reactbits``.

        Returns:
            Formatted prompt string ready to send to the LLM.
        """
        lines = [f"Project requirement: {problem_statement}"]
        extras: list[str] = []
        if options.get("threejs"):
            extras.append("Use Three.js for 3D visuals in the frontend.")
        if options.get("gsap"):
            extras.append("Use GSAP for animations in the frontend.")
        if options.get("reactbits"):
            extras.append("Use ReactBits for UI components in the frontend.")
        if extras:
            lines.append("Additional requirements:")
            lines.extend(f"  - {e}" for e in extras)
        lines.append(
            "\nGenerate the complete project now. Return only the JSON object — nothing else."
        )
        return "\n".join(lines)

    def _clean_json(self, raw: str) -> str:
        """Strip markdown fences wrapping a JSON response.

        Some LLMs wrap JSON output in ` ```json ` fences despite instructions.
        This method removes them so ``json.loads`` can parse the result cleanly.

        Args:
            raw: Raw string returned by the LLM.

        Returns:
            Clean JSON string with no surrounding markdown fences.
        """
        raw = raw.strip()
        for fence in ("```json", "```"):
            if raw.startswith(fence):
                raw = raw[len(fence):]
                break
        if raw.endswith("```"):
            raw = raw[:-3]
        return raw.strip()

    def _clean_content(self, content: str) -> str:
        """Strip markdown fences from generated file content.

        Args:
            content: File content string from the LLM JSON response.

        Returns:
            Clean file content with no surrounding markdown fences.
        """
        content = content.strip()
        for fence in ("```python", "```typescript", "```tsx", "```jsx", "```json", "```yaml", "```"):
            if content.startswith(fence):
                content = content[len(fence):]
                break
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()


# Module-level singleton — import this directly instead of instantiating FullProjectGenerator yourself.
full_project_generator = FullProjectGenerator()
