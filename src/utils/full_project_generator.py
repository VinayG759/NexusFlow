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
import os
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.debugging_agent import debugging_agent
from src.agents.file_agent import file_agent
from src.agents.ui_design_agent import ui_design_agent
from src.config.settings import settings
from src.database.models import Project, ProjectFile
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

    async def generate(self, problem_statement: str, options: dict | None = None, db: AsyncSession | None = None) -> dict:
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
        description: str = project.get("description", "")
        files: list[dict] = project.get("files", [])
        setup_instructions: str = project.get("setup_instructions", "")
        env_variables: list[dict] = project.get("env_variables", [])

        # ── Build processed file list (in-memory tsconfig patch + README) ─────
        processed: list[dict] = []

        for file_entry in files:
            path: str = file_entry.get("path", "").strip()
            content: str = self._clean_content(file_entry.get("content", ""))
            if not path:
                logger.warning("[%s] Skipping file entry with empty path.", self.agent_name)
                continue
            # Patch tsconfig.json in-memory to ensure skipLibCheck
            if path.endswith("tsconfig.json") and "node_modules" not in path:
                try:
                    tsconfig_data = json.loads(content)
                    if "compilerOptions" not in tsconfig_data:
                        tsconfig_data["compilerOptions"] = {}
                    tsconfig_data["compilerOptions"]["skipLibCheck"] = True
                    content = json.dumps(tsconfig_data, indent=2)
                except Exception:
                    pass
            processed.append({"path": path, "content": content})

        # ── UI Design Agent — enhance frontend if ANTHROPIC_API_KEY is set ──────
        frontend_files = [f for f in processed if ui_design_agent.is_frontend_file(f.get("path", ""))]
        backend_files  = [f for f in processed if not ui_design_agent.is_frontend_file(f.get("path", ""))]

        if os.getenv("ANTHROPIC_API_KEY") and frontend_files:
            logger.info("[%s] Running UIDesignAgent to enhance frontend...", self.agent_name)
            design_result = await ui_design_agent.enhance_frontend(
                project_name=project_name,
                problem_statement=problem_statement,
                existing_frontend_files=frontend_files,
                reference_context=opts.get("reference_context", ""),
            )
            if design_result["status"] == "success":
                frontend_files = design_result["files"]
                logger.info("[%s] Frontend enhanced by UIDesignAgent.", self.agent_name)
            else:
                logger.warning("[%s] UIDesignAgent failed, using Groq frontend.", self.agent_name)

        processed = backend_files + frontend_files

        # Always generate a detailed README via a separate LLM call, replacing any generic one.
        file_paths = [f["path"] for f in processed if f["path"].lower() != "readme.md"]
        readme_content = await self._generate_readme(
            project_name=project_name,
            file_list=file_paths,
            env_variables=env_variables,
        )
        processed = [f for f in processed if f["path"].lower() != "readme.md"]
        processed.append({"path": "README.md", "content": readme_content})

        logger.info(
            "[%s] LLM returned project=%r with %d file(s); LLM-generated README added.",
            self.agent_name, project_name, len(files),
        )

        # ── Debugging Agent — fix syntax / type errors ────────────────────────
        debug_fixes_applied = 0
        debug_remaining_errors: list[str] = []
        debug_attempts = 0

        logger.info("[%s] Running DebuggingAgent...", self.agent_name)
        debug_result = await debugging_agent.debug_project(
            project_files=processed,
            project_name=project_name,
            db_session=db,
        )
        if debug_result["status"] in ("success", "partial"):
            processed = debug_result["fixed_files"]
            debug_fixes_applied = debug_result.get("fixes_applied", 0)
            debug_remaining_errors = debug_result.get("remaining_errors", [])
            debug_attempts = debug_result.get("attempts", 0)
            logger.info(
                "[%s] DebuggingAgent: %d fix(es) applied, %d error(s) remaining",
                self.agent_name, debug_fixes_applied, len(debug_remaining_errors),
            )

        # ── Save to database (preferred) ──────────────────────────────────────
        if db is not None:
            db_result = await self._save_to_db(
                db=db,
                project_name=project_name,
                description=description,
                problem_statement=problem_statement,
                setup_instructions=setup_instructions,
                env_variables=env_variables,
                processed=processed,
            )
            db_result["debug_fixes_applied"] = debug_fixes_applied
            db_result["debug_remaining_errors"] = debug_remaining_errors
            db_result["debug_attempts"] = debug_attempts
            return db_result

        # ── Fallback: save to filesystem ──────────────────────────────────────
        output_base: str = opts.get("output_directory", "").strip() or settings.OUTPUT_DIRECTORY
        base_path = str(Path(output_base) / project_name)
        Path(base_path).mkdir(parents=True, exist_ok=True)

        files_saved: list[str] = []
        files_failed: list[str] = []

        for file_entry in processed:
            save_path = f"{base_path}/{file_entry['path']}"
            save_result = file_agent.create_project_file(save_path, file_entry["content"])
            if save_result.get("status") == "success":
                files_saved.append(save_path)
            else:
                files_failed.append(save_path)
                logger.warning("[%s] Failed to save %r: %s", self.agent_name, save_path, save_result.get("error"))

        overall_status = "error" if (files_failed and not files_saved) else ("partial" if files_failed else "success")
        logger.info(
            "[%s] Generation complete (filesystem) — status=%s saved=%d failed=%d",
            self.agent_name, overall_status, len(files_saved), len(files_failed),
        )
        return {
            "status": overall_status,
            "project_name": project_name,
            "actual_output_path": base_path,
            "files_saved": files_saved,
            "files_failed": files_failed,
            "setup_instructions": setup_instructions,
            "env_variables": env_variables,
            "debug_fixes_applied": debug_fixes_applied,
            "debug_remaining_errors": debug_remaining_errors,
            "debug_attempts": debug_attempts,
        }

    async def _save_to_db(
        self,
        db: AsyncSession,
        project_name: str,
        description: str,
        problem_statement: str,
        setup_instructions: str,
        env_variables: list[dict],
        processed: list[dict],
    ) -> dict:
        """Persist the generated project to the database and return the project ID."""
        try:
            project_record = Project(
                name=project_name,
                description=description,
                problem_statement=problem_statement,
                status="ready",
                setup_instructions=setup_instructions,
                tech_stack="FastAPI · React · PostgreSQL",
            )
            db.add(project_record)
            await db.flush()  # assigns project_record.id

            for file_entry in processed:
                ext = Path(file_entry["path"]).suffix.lower()
                file_type = _ext_to_type(ext)
                db.add(ProjectFile(
                    project_id=project_record.id,
                    file_path=file_entry["path"],
                    content=file_entry["content"],
                    file_type=file_type,
                ))

            await db.commit()
            await db.refresh(project_record)

            logger.info(
                "[%s] Saved project id=%d name=%r with %d file(s) to DB.",
                self.agent_name, project_record.id, project_name, len(processed),
            )
            return {
                "status": "success",
                "project_id": project_record.id,
                "project_name": project_name,
                "total_files": len(processed),
                "setup_instructions": setup_instructions,
                "env_variables": env_variables,
            }
        except Exception as exc:
            await db.rollback()
            logger.exception("[%s] DB save failed: %s", self.agent_name, exc)
            return {
                "status": "error",
                "project_name": project_name,
                "error": str(exc),
                "total_files": 0,
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

    async def _generate_readme(
        self,
        project_name: str,
        file_list: list[str],
        env_variables: list[dict],
    ) -> str:
        """Generate a detailed README via a separate LLM call.

        Falls back to :meth:`_build_readme` if the LLM call fails.
        """
        slug = project_name.lower().replace(" ", "-").replace("_", "-")
        file_tree = "\n".join(f"  {p}" for p in file_list)
        env_lines = "\n".join(
            f"  {ev.get('name', '')}: {ev.get('description', '')} (e.g. {ev.get('example', '')})"
            for ev in env_variables
        ) or "  (none specified)"

        system_prompt = (
            "You are a technical writer. Generate a detailed, accurate README.md for this project. "
            "Include EXACT commands, EXACT file paths, EXACT environment variable names. "
            "No generic placeholders. Every command must be copy-pasteable and work."
        )

        user_prompt = f"""Generate a detailed README.md for this project:
Project name: {project_name}
Tech stack: FastAPI backend, React TypeScript frontend, PostgreSQL database
Files generated:
{file_tree}
Environment variables needed:
{env_lines}

The README must include these exact sections:

# {project_name}

## Prerequisites
List exact versions needed based on requirements.txt and package.json

## Project Structure
Show the actual file tree

## Backend Setup
### 1. Create virtual environment
python -m venv venv

### 2. Activate virtual environment
Windows: venv\\Scripts\\activate
Mac/Linux: source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Create PostgreSQL database
createdb {slug}
OR using psql:
psql -U postgres -c 'CREATE DATABASE {slug};'

### 5. Configure environment variables
Create backend/.env file:
DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/{slug}
(list ALL env variables with descriptions)

### 6. Start backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Backend API available at: http://localhost:8000
API Documentation: http://localhost:8000/docs

## Frontend Setup
### 1. Install dependencies
cd frontend
npm install

### 2. Configure environment
Create frontend/.env:
REACT_APP_API_URL=http://localhost:8000

### 3. Start frontend
npm start

Frontend available at: http://localhost:3000

## Running Both Together
Open two terminals:
Terminal 1 (Backend): cd backend && uvicorn main:app --reload
Terminal 2 (Frontend): cd frontend && npm start

## API Endpoints
List all FastAPI routes found in routes.py

## Common Issues & Fixes
- Database connection error: Verify PostgreSQL is running with: pg_isready
- Port in use: Change port with --port flag
- Module not found: Make sure virtual environment is activated
- npm install fails: Try npm install --legacy-peer-deps

## Tech Stack
- Backend: FastAPI, SQLAlchemy 2.0, asyncpg, PostgreSQL
- Frontend: React 18, TypeScript, TailwindCSS
- Database: PostgreSQL 14+"""

        logger.info("[%s] Generating README via LLM for project=%r", self.agent_name, project_name)
        result = await api_connector.call_groq(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=2000,
        )

        if result["status"] == "success" and result.get("content"):
            content = result["content"].strip()
            # Strip any wrapping markdown fences some LLMs add despite instructions
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            logger.info("[%s] README generated by LLM (%d chars).", self.agent_name, len(content))
            return content.strip()

        logger.warning("[%s] README LLM call failed, using fallback template.", self.agent_name)
        return self._build_readme(project_name)

    def _build_readme(self, project_name: str) -> str:
        """Fallback README template used when the LLM call for README generation fails."""
        slug = project_name.lower().replace(" ", "-").replace("_", "-")
        return f"""\
# {project_name}

## Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 14+

## Backend Setup
```bash
cd backend
python -m venv venv
# Windows: venv\\Scripts\\activate  |  Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Create the database:
```bash
psql -U postgres -c 'CREATE DATABASE {slug};'
```

Create `backend/.env`:
```
DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/{slug}
```

Start backend:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
API docs: http://localhost:8000/docs

## Frontend Setup
```bash
cd frontend
npm install
```

Create `frontend/.env`:
```
REACT_APP_API_URL=http://localhost:8000
```

Start frontend:
```bash
npm start
```
App: http://localhost:3000

## Common Issues
- **DB connection error**: Check PostgreSQL is running and DATABASE_URL is correct
- **Port in use**: Add `--port 8001` to the uvicorn command
- **npm install fails**: Run `npm install --legacy-peer-deps`
"""

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


def _ext_to_type(ext: str) -> str:
    """Map a file extension to a simple file-type label."""
    mapping = {
        ".py": "python", ".ts": "typescript", ".tsx": "typescript",
        ".js": "javascript", ".jsx": "javascript", ".json": "json",
        ".css": "css", ".html": "html", ".md": "markdown",
        ".yml": "yaml", ".yaml": "yaml", ".sh": "bash", ".bat": "batch",
        ".env": "env", ".txt": "text", ".sql": "sql", ".toml": "toml",
    }
    return mapping.get(ext, "text")


# Module-level singleton — import this directly instead of instantiating FullProjectGenerator yourself.
full_project_generator = FullProjectGenerator()
