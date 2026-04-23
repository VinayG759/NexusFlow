"""NexusFlow — application entry point.

Starts the FastAPI server, wires all agents together, and exposes endpoints
for health checking, agent discovery, and problem orchestration.

Run directly::

    python main.py

Or via uvicorn::

    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import asyncio

import os
import shutil
import subprocess
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agents.api_agent import api_agent
from src.agents.builder_agent import builder_agent
from src.agents.deploy_agent import deploy_agent
from src.agents.file_agent import file_agent
from src.agents.orchestrator import OrchestratorAgent
from src.agents.research_agent import research_agent
from src.config.settings import settings
from src.database.connection import init_db
from src.tools.api_connector import api_connector
from src.utils.logger import get_logger
from src.utils.api_analyzer import api_analyzer
from src.utils.fix_loop import fix_loop
from src.utils.full_project_generator import full_project_generator
from src.utils.project_generator import project_generator
from src.utils.task_executor import task_executor

logger = get_logger(__name__)

# ── Orchestrator ──────────────────────────────────────────────────────────────

orchestrator = OrchestratorAgent(
    "Orchestrator",
    ["ResearchAgent", "BuilderAgent", "FileAgent", "APIAgent", "DeployAgent"],
)

# ── Request / response models ─────────────────────────────────────────────────


class RunRequest(BaseModel):
    """Request body for the POST /run endpoint."""
    problem_statement: str
    execute: bool = False


class GenerateRequest(BaseModel):
    """Request body for the POST /generate endpoint."""
    problem_statement: str
    use_threejs: bool = False
    use_gsap: bool = False
    use_reactbits: bool = False
    additional_details: str = ""


class BuildRequest(BaseModel):
    """Request body for the POST /build endpoint."""
    problem_statement: str
    use_threejs: bool = False
    use_gsap: bool = False
    use_reactbits: bool = False
    auto_fix: bool = True
    output_directory: str = ""
    api_keys: dict[str, str] = {}


class RunProjectRequest(BaseModel):
    """Request body for the POST /run-project endpoint."""
    project_path: str
    port_backend: int = 8002
    port_frontend: int = 3001


class StopProjectRequest(BaseModel):
    """Request body for the POST /stop-project endpoint."""
    project_path: str


class AnalyzeRequest(BaseModel):
    """Request body for the POST /analyze endpoint."""
    problem_statement: str


class PlanRequest(BaseModel):
    """Request body for the POST /plan endpoint."""
    problem_statement: str
    use_threejs: bool = False
    use_gsap: bool = False
    use_reactbits: bool = False


# ── Running-process store ─────────────────────────────────────────────────────
# Maps project_path → {backend: Process|None, frontend: Process|None, ...}

_running_projects: dict[str, dict] = {}


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown events for the FastAPI application."""
    logger.info("Starting %s v%s...", settings.APP_NAME, settings.APP_VERSION)
    await init_db()
    logger.info("%s is ready to accept requests.", settings.APP_NAME)

    yield

    logger.info("%s is shutting down. Goodbye.", settings.APP_NAME)
    await api_connector.close()
    logger.info("API connector closed.")


# ── App ───────────────────────────────────────────────────────────────────────


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/")
async def root() -> dict:
    """Return a welcome message."""
    return {"message": "Welcome to NexusFlow"}


@app.get("/health")
async def health_check() -> dict:
    """Return current application health status."""
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@app.get("/agents")
async def list_agents() -> dict:
    """Return all registered agents and their availability status."""
    return {
        "status": "ok",
        "agents": [
            {"name": "Orchestrator",   "type": "OrchestratorAgent", "status": "available"},
            {"name": "ResearchAgent",  "type": "ResearchAgent",     "status": "available"},
            {"name": "BuilderAgent",   "type": "BuilderAgent",      "status": "available"},
            {"name": "FileAgent",      "type": "FileAgent",         "status": "available"},
            {"name": "APIAgent",       "type": "APIAgent",          "status": "available"},
            {"name": "DeployAgent",    "type": "DeployAgent",       "status": "available"},
        ],
        "total": 6,
    }


@app.post("/run")
async def run(request: RunRequest) -> dict:
    """Orchestrate a problem statement and optionally execute the full task plan.

    Always calls the Orchestrator to decompose the problem into a structured
    task plan via the Groq LLM. When ``execute`` is ``True``, the task plan is
    forwarded to :class:`~src.utils.task_executor.TaskExecutor` which routes
    each task to the appropriate agent and collects results.

    Args:
        request: JSON body with:
            - ``problem_statement`` (str): the high-level problem to solve.
            - ``execute`` (bool, default ``False``): set to ``True`` to run
              the plan immediately after planning.

    Returns:
        Plan-only (``execute=False``)::

            {
                "status":      "success",
                "problem":     str,
                "tasks":       list[dict],
                "total_tasks": int,
            }

        With execution (``execute=True``)::

            {
                "status":            "success" | "partial" | "error",
                "problem":           str,
                "total_tasks":       int,
                "completed_tasks":   list[str],
                "failed_tasks":      list[str],
                "execution_results": dict,
            }

        On failure::

            {"status": "error", "message": str}
    """
    logger.info(
        "POST /run — problem=%r execute=%s", request.problem_statement, request.execute,
    )
    try:
        # ── Step 1: plan ──────────────────────────────────────────────────────
        plan = await orchestrator.orchestrate(request.problem_statement)
        if plan.get("status") != "success":
            return {
                "status": "error",
                "message": plan.get("error", "Orchestration failed."),
            }

        tasks = plan["tasks"]
        logger.info("POST /run — %d task(s) planned.", len(tasks))

        if not request.execute:
            return {
                "status": "success",
                "problem": plan["problem"],
                "tasks": tasks,
                "total_tasks": len(tasks),
            }

        # ── Step 2: execute ───────────────────────────────────────────────────
        logger.info("POST /run — executing %d task(s)...", len(tasks))
        execution = await task_executor.execute_plan(tasks, request.problem_statement)
        logger.info(
            "POST /run — execution finished: completed=%d failed=%d",
            len(execution["completed_tasks"]), len(execution["failed_tasks"]),
        )
        return {
            "status": execution["status"],
            "problem": execution["problem"],
            "total_tasks": len(tasks),
            "completed_tasks": execution["completed_tasks"],
            "failed_tasks": execution["failed_tasks"],
            "execution_results": execution["results"],
        }

    except Exception as exc:
        logger.exception("POST /run unhandled error: %s", exc)
        return {"status": "error", "message": str(exc)}


@app.post("/generate")
async def generate(request: GenerateRequest) -> dict:
    """Generate a complete full-stack project from a problem statement.

    Builds an enriched problem description from the request fields and delegates
    to :class:`~src.utils.project_generator.ProjectGenerator`, which plans the
    full file structure in one LLM call and then generates each file in
    dependency order before saving them to disk.

    Args:
        request: JSON body with:
            - ``problem_statement`` (str): high-level description of the project.
            - ``use_threejs`` (bool, default ``False``): append Three.js constraint.
            - ``use_gsap`` (bool, default ``False``): append GSAP constraint.
            - ``use_reactbits`` (bool, default ``False``): append ReactBits constraint.
            - ``additional_details`` (str, default ``""``): extra context appended
              to the problem statement.

    Returns:
        ::

            {
                "status":          "success" | "partial" | "error",
                "project_name":    str,
                "files_generated": list[str],
                "files_failed":    list[str],
                "total_files":     int,
                "structure":       dict,
            }

        On failure::

            {"status": "error", "message": str}
    """
    # Build the enriched problem statement.
    parts = [request.problem_statement.strip()]
    if request.additional_details.strip():
        parts.append(request.additional_details.strip())
    if request.use_threejs:
        parts.append("Use Three.js for 3D visuals.")
    if request.use_gsap:
        parts.append("Use GSAP for animations.")
    if request.use_reactbits:
        parts.append("Use ReactBits for UI components.")
    full_problem = " ".join(parts)

    logger.info(
        "POST /generate — problem=%r threejs=%s gsap=%s reactbits=%s",
        full_problem[:120], request.use_threejs, request.use_gsap, request.use_reactbits,
    )

    try:
        result = await project_generator.generate_project(full_problem)
        logger.info(
            "POST /generate — status=%s project=%r generated=%d failed=%d",
            result["status"],
            result.get("project_name", "?"),
            len(result.get("files_generated", [])),
            len(result.get("files_failed", [])),
        )
        return {
            "status": result["status"],
            "project_name": result.get("project_name", ""),
            "files_generated": result.get("files_generated", []),
            "files_failed": result.get("files_failed", []),
            "total_files": len(result.get("files_generated", [])) + len(result.get("files_failed", [])),
            "structure": result.get("structure", {}),
        }
    except Exception as exc:
        logger.exception("POST /generate unhandled error: %s", exc)
        return {"status": "error", "message": str(exc)}


@app.post("/build")
async def build(request: BuildRequest) -> dict:
    """Generate a complete full-stack project in a single LLM call.

    Delegates to :class:`~src.utils.full_project_generator.FullProjectGenerator`
    which sends one request to the LLM and receives every project file in a
    single JSON response, guaranteeing cross-file consistency before anything
    is written to disk.

    Args:
        request: JSON body with:
            - ``problem_statement`` (str): high-level description of the project.
            - ``use_threejs`` (bool, default ``False``): add Three.js requirement.
            - ``use_gsap`` (bool, default ``False``): add GSAP requirement.
            - ``use_reactbits`` (bool, default ``False``): add ReactBits requirement.
            - ``auto_fix`` (bool, default ``True``): run the validate → fix loop
              after generation.
            - ``output_directory`` (str, default ``""``): save project here instead
              of the NexusFlow root when non-empty.

    Returns:
        ::

            {
                "status":               "success" | "partial" | "error",
                "project_name":         str,
                "files_saved":          list[str],
                "files_failed":         list[str],
                "total_files":          int,
                "setup_instructions":   str,
                "env_variables":        list[dict],
                "fix_status":           str | None,
                "attempts":             int | None,
                "fixes_applied":        int | None,
                "final_valid_files":    list[str] | None,
                "final_invalid_files":  list[dict] | None,
            }

        On failure::

            {"status": "error", "message": str}
    """
    options: dict = {
        "threejs": request.use_threejs,
        "gsap": request.use_gsap,
        "reactbits": request.use_reactbits,
    }
    if request.output_directory.strip():
        options["output_directory"] = request.output_directory.strip()

    logger.info(
        "POST /build — problem=%r options=%s auto_fix=%s",
        request.problem_statement[:120], options, request.auto_fix,
    )

    try:
        result = await full_project_generator.generate(request.problem_statement, options)
        files_saved: list[str] = result.get("files_saved", [])
        files_failed: list[str] = result.get("files_failed", [])
        project_name: str = result.get("project_name", "")

        logger.info(
            "POST /build — status=%s project=%r saved=%d failed=%d",
            result["status"], project_name, len(files_saved), len(files_failed),
        )

        response: dict = {
            "status": result["status"],
            "project_name": project_name,
            "files_saved": files_saved,
            "files_failed": files_failed,
            "total_files": len(files_saved) + len(files_failed),
            "setup_instructions": result.get("setup_instructions", ""),
            "env_variables": result.get("env_variables", []),
            "actual_output_path": result.get("actual_output_path", ""),
            "fix_status": None,
            "attempts": None,
            "fixes_applied": None,
            "final_valid_files": None,
            "final_invalid_files": None,
        }

        # ── Inject API keys into .env ─────────────────────────────────────────
        if request.api_keys and files_saved:
            output_path = Path(result.get("actual_output_path") or project_name)
            env_example = output_path / ".env.example"
            if not env_example.exists():
                # Also check inside backend/ subdirectory
                env_example = output_path / "backend" / ".env.example"

            if env_example.exists():
                lines = env_example.read_text(encoding="utf-8").splitlines()
                injected: list[str] = []
                for line in lines:
                    matched = False
                    for api_name, api_key in request.api_keys.items():
                        if api_name.lower().replace(" ", "_") in line.lower():
                            # Replace everything after the first '=' with the real key
                            key_part = line.split("=", 1)[0]
                            line = f"{key_part}={api_key}"
                            matched = True
                            logger.info(
                                "POST /build — injected key for %r into %s",
                                api_name, key_part.strip(),
                            )
                            break
                    injected.append(line)

                env_file = env_example.parent / ".env"
                env_file.write_text("\n".join(injected), encoding="utf-8")
                logger.info("POST /build — .env written to %s", env_file)
            else:
                logger.warning(
                    "POST /build — api_keys provided but no .env.example found in %s", output_path,
                )

        if request.auto_fix and files_saved and project_name:
            fix_path = result.get("actual_output_path") or project_name
            logger.info("POST /build — running fix_loop on %r", fix_path)
            fix_result = await fix_loop.run(fix_path)
            response["fix_status"] = fix_result["status"]
            response["attempts"] = fix_result["attempts"]
            response["fixes_applied"] = fix_result["fixes_applied"]
            response["final_valid_files"] = fix_result["final_valid_files"]
            response["final_invalid_files"] = fix_result["final_invalid_files"]
            logger.info(
                "POST /build — fix_loop done: status=%s attempts=%d fixes=%d",
                fix_result["status"], fix_result["attempts"], fix_result["fixes_applied"],
            )

        return response

    except Exception as exc:
        logger.exception("POST /build unhandled error: %s", exc)
        return {"status": "error", "message": str(exc)}


@app.post("/analyze")
async def analyze(request: AnalyzeRequest) -> dict:
    """Identify all external APIs required to build the described project.

    Calls the LLM once to extract every third-party service needed, then
    enriches each paid-only API with web-searched free alternatives.

    Args:
        request: JSON body with ``problem_statement`` (str).

    Returns:
        ::

            {
                "status":        "success" | "error",
                "problem":       str,
                "apis":          list[dict],
                "total":         int,
                "has_paid_only": bool,
            }

        On failure::

            {"status": "error", "message": str}
    """
    logger.info("POST /analyze — problem=%r", request.problem_statement[:120])

    try:
        result = await api_analyzer.analyze(request.problem_statement)

        if result.get("status") != "success":
            return {"status": "error", "message": result.get("error", "Analysis failed.")}

        apis: list[dict] = result["apis"]

        # Enrich paid-only APIs with web-searched free alternatives.
        for api in apis:
            if api.get("is_paid_only"):
                logger.info("POST /analyze — searching alternatives for %r", api.get("name"))
                web_alts = await api_analyzer.search_alternatives(api["name"])
                api["web_alternatives"] = web_alts

        logger.info(
            "POST /analyze — done: total=%d has_paid_only=%s",
            result["total"], result["has_paid_only"],
        )
        return {
            "status":        "success",
            "problem":       request.problem_statement,
            "apis":          apis,
            "total":         result["total"],
            "has_paid_only": result["has_paid_only"],
        }

    except Exception as exc:
        logger.exception("POST /analyze unhandled error: %s", exc)
        return {"status": "error", "message": str(exc)}


@app.post("/plan")
async def plan(request: PlanRequest) -> dict:
    """Generate a visual project plan from a problem statement.

    Makes a single LLM call to produce a structured JSON plan including tech
    stack, file structure, complexity estimate, and required APIs.

    Args:
        request: JSON body with ``problem_statement`` and optional feature flags.

    Returns:
        ``{status, project_name, description, tech_stack, file_structure,
           apis_required, estimated_files, complexity, estimated_build_time}``
    """
    logger.info("POST /plan — problem=%r", request.problem_statement[:120])

    system_prompt = (
        "You are a senior software architect. Analyze the problem statement and return ONLY a valid JSON object "
        "with this exact structure — no markdown, no explanation, no text outside the JSON:\n"
        "{\n"
        '  "project_name": "lowercase-slug",\n'
        '  "description": "one sentence describing the project",\n'
        '  "tech_stack": {\n'
        '    "backend": ["FastAPI", "SQLAlchemy", "asyncpg"],\n'
        '    "frontend": ["React 18", "TypeScript", "TailwindCSS"],\n'
        '    "database": ["PostgreSQL"],\n'
        '    "devops": ["Docker", "Docker Compose"]\n'
        "  },\n"
        '  "file_structure": [\n'
        '    {"path": "backend", "description": "FastAPI backend", "type": "folder"},\n'
        '    {"path": "backend/main.py", "description": "Application entry point", "type": "file"}\n'
        "  ],\n"
        '  "apis_required": ["OpenWeatherMap API"],\n'
        '  "estimated_files": 18,\n'
        '  "complexity": "medium",\n'
        '  "estimated_build_time": "2-3 minutes"\n'
        "}\n"
        "complexity must be exactly one of: simple, medium, complex."
    )

    user_prompt = f"Project requirement: {request.problem_statement}"
    if request.use_threejs:
        user_prompt += "\nUse Three.js for 3D visuals."
    if request.use_gsap:
        user_prompt += "\nUse GSAP for animations."
    if request.use_reactbits:
        user_prompt += "\nUse ReactBits for UI components."
    user_prompt += "\n\nReturn ONLY the JSON object."

    try:
        llm_result = await api_connector.call_groq(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        if llm_result["status"] != "success":
            return {"status": "error", "message": llm_result.get("error", "LLM call failed")}

        raw = llm_result["content"].strip()
        for fence in ("```json", "```"):
            if raw.startswith(fence):
                raw = raw[len(fence):]
                break
        if raw.endswith("```"):
            raw = raw[:-3]

        plan_data = json.loads(raw.strip())
        return {"status": "success", **plan_data}

    except json.JSONDecodeError as exc:
        logger.error("POST /plan — JSON parse error: %s", exc)
        return {"status": "error", "message": f"JSON parse error: {exc}"}
    except Exception as exc:
        logger.exception("POST /plan unhandled error: %s", exc)
        return {"status": "error", "message": str(exc)}


@app.post("/run-project")
async def run_project(request: RunProjectRequest) -> dict:
    """Start a generated project's backend and/or frontend in the background.

    Detects which components exist (FastAPI backend, React frontend) and
    launches each as a subprocess.Popen process.  Processes are stored in
    ``_running_projects`` so they can be stopped later via ``POST /stop-project``.

    Uses subprocess.Popen (not asyncio.create_subprocess_exec) for Windows
    compatibility when running under uvicorn.

    Args:
        request: JSON body with ``project_path``, ``port_backend``,
            ``port_frontend``.

    Returns:
        ``{status, project_path, backend_url, frontend_url,
           backend_running, frontend_running}``
    """
    project = Path(request.project_path)
    if not project.exists():
        return {"status": "error", "message": f"Project path does not exist: {request.project_path}"}

    has_backend  = (project / "backend"  / "main.py").exists()
    has_frontend = (project / "frontend" / "package.json").exists()

    if not has_backend and not has_frontend:
        return {
            "status": "error",
            "message": "Neither backend/main.py nor frontend/package.json found in the project.",
        }

    backend_proc:  subprocess.Popen | None = None
    frontend_proc: subprocess.Popen | None = None
    backend_url:   str | None = None
    frontend_url:  str | None = None

    npm_path = shutil.which("npm") or "npm"

    # Windows: CREATE_NEW_PROCESS_GROUP lets us send Ctrl+Break to the child group
    popen_flags: dict = {}
    if sys.platform == "win32":
        popen_flags["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    # ── Backend ───────────────────────────────────────────────────────────────
    if has_backend:
        backend_dir = project / "backend"
        req_file    = backend_dir / "requirements.txt"

        if req_file.exists():
            logger.info("POST /run-project — installing backend deps in %s", backend_dir)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
                cwd=str(backend_dir),
                shell=False,
                timeout=180,
            ))

        # ── Auto-create PostgreSQL database ───────────────────────────────────
        try:
            import psycopg2
            from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

            env_file = backend_dir / ".env"
            db_name = project.name.lower().replace("-", "_").replace(" ", "_")
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if line.startswith("DATABASE_URL"):
                        url = line.split("=", 1)[-1].strip()
                        db_name = url.split("/")[-1].split("?")[0]
                        break
            conn = psycopg2.connect(
                dbname="postgres",
                user="postgres",
                password="vinay2004",
                host="localhost",
                port=5432,
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()
            cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
            if not cur.fetchone():
                cur.execute(f'CREATE DATABASE "{db_name}"')
                logger.info("POST /run-project — created database: %s", db_name)
            else:
                logger.info("POST /run-project — database already exists: %s", db_name)
            cur.close()
            conn.close()
        except Exception as db_exc:
            logger.warning("POST /run-project — could not auto-create database: %s", db_exc)

        logger.info(
            "POST /run-project — starting uvicorn on :%d for %s",
            request.port_backend, backend_dir,
        )
        backend_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app",
             "--host", "0.0.0.0", "--port", str(request.port_backend)],
            cwd=str(backend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            **popen_flags,
        )
        backend_url = f"http://localhost:{request.port_backend}"

    # ── Frontend ──────────────────────────────────────────────────────────────
    if has_frontend:
        frontend_dir = project / "frontend"
        node_modules = frontend_dir / "node_modules"
        if not node_modules.exists():
            logger.info("POST /run-project — running npm install in %s", frontend_dir)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: subprocess.run(
                [npm_path, "install", "--legacy-peer-deps"],
                cwd=str(frontend_dir),
                shell=False,
                timeout=180,
            ))
        else:
            logger.info("POST /run-project — node_modules exists, skipping npm install")

        env = {**os.environ, "PORT": str(request.port_frontend)}
        logger.info(
            "POST /run-project — starting npm start on :%d for %s",
            request.port_frontend, frontend_dir,
        )
        frontend_proc = subprocess.Popen(
            [npm_path, "start"],
            cwd=str(frontend_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            **popen_flags,
        )
        frontend_url = f"http://localhost:{request.port_frontend}"

    # ── Store processes ───────────────────────────────────────────────────────
    _running_projects[request.project_path] = {
        "backend_proc":   backend_proc,
        "frontend_proc":  frontend_proc,
        "backend_url":    backend_url,
        "frontend_url":   frontend_url,
        "port_backend":   request.port_backend  if has_backend  else None,
        "port_frontend":  request.port_frontend if has_frontend else None,
    }

    logger.info(
        "POST /run-project — launched: backend=%s frontend=%s",
        backend_url, frontend_url,
    )
    return {
        "status":           "success",
        "project_path":     request.project_path,
        "backend_url":      backend_url,
        "frontend_url":     frontend_url,
        "backend_running":  backend_proc  is not None,
        "frontend_running": frontend_proc is not None,
    }


@app.post("/stop-project")
async def stop_project(request: StopProjectRequest) -> dict:
    """Kill all running processes for a project.

    Args:
        request: JSON body with ``project_path``.

    Returns:
        ``{status, message}``
    """
    info = _running_projects.pop(request.project_path, None)
    if info is None:
        return {"status": "error", "message": f"No running project found for: {request.project_path}"}

    stopped: list[str] = []
    for key in ("backend_proc", "frontend_proc"):
        proc: subprocess.Popen | None = info.get(key)
        if proc is not None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                stopped.append(key.replace("_proc", ""))
            except Exception as exc:
                logger.warning("POST /stop-project — could not stop %s: %s", key, exc)

    logger.info("POST /stop-project — stopped %s for %s", stopped, request.project_path)
    return {
        "status":  "success",
        "message": f"Stopped {', '.join(stopped) or 'no processes'} for {request.project_path}",
    }


@app.get("/running-projects")
async def running_projects() -> dict:
    """Return all projects that have active background processes.

    Returns:
        ``{status, projects: list[{project_path, backend_url, frontend_url,
           backend_running, frontend_running}], total: int}``
    """
    alive = []
    for path, info in list(_running_projects.items()):
        bp: subprocess.Popen | None = info.get("backend_proc")
        fp: subprocess.Popen | None = info.get("frontend_proc")

        backend_alive  = bp is not None and bp.poll() is None
        frontend_alive = fp is not None and fp.poll() is None

        if not backend_alive and not frontend_alive:
            _running_projects.pop(path, None)
            continue

        alive.append({
            "project_path":     path,
            "backend_url":      info.get("backend_url"),
            "frontend_url":     info.get("frontend_url"),
            "backend_running":  backend_alive,
            "frontend_running": frontend_alive,
        })

    return {"status": "ok", "projects": alive, "total": len(alive)}


@app.get("/list-projects")
async def list_projects() -> list[dict]:
    """List all projects in the configured output directory.

    Scans ``settings.OUTPUT_DIRECTORY``, enumerates every immediate
    subdirectory, and counts the total number of files inside each one
    recursively.

    Returns:
        A JSON array of objects::

            [
                {
                    "name":       str,   // folder name
                    "path":       str,   // absolute path
                    "file_count": int    // total files (recursive)
                },
                ...
            ]

        Returns ``[]`` when the directory does not exist or is empty.
    """
    logger.info("GET /list-projects — scanning %s", settings.OUTPUT_DIRECTORY)
    output_dir = Path(settings.OUTPUT_DIRECTORY)

    if not output_dir.exists():
        logger.warning("GET /list-projects — OUTPUT_DIRECTORY does not exist: %s", output_dir)
        return []

    try:
        projects: list[dict] = []
        for entry in sorted(output_dir.iterdir()):
            if not entry.is_dir():
                continue
            try:
                file_count = sum(
                    1 for f in entry.rglob("*")
                    if f.is_file() and "node_modules" not in f.parts and "__pycache__" not in f.parts
                )
            except Exception as scan_exc:
                logger.warning("GET /list-projects — could not count files in %s: %s", entry, scan_exc)
                file_count = 0

            projects.append({
                "name":       entry.name,
                "path":       str(entry.resolve()),
                "file_count": file_count,
            })

        logger.info("GET /list-projects — found %d project(s)", len(projects))
        return projects

    except Exception as exc:
        logger.exception("GET /list-projects unhandled error: %s", exc)
        return []


# ── Entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
