"""NexusFlow — application entry point.

Starts the FastAPI server, wires all agents together, and exposes endpoints
for health checking, agent discovery, and problem orchestration.

Run directly::

    python main.py

Or via uvicorn::

    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import json
import asyncio
import httpx
import zipfile

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func as sql_func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.agents.api_agent import api_agent
from src.agents.builder_agent import builder_agent
from src.agents.deploy_agent import deploy_agent
from src.agents.file_agent import file_agent
from src.agents.orchestrator import OrchestratorAgent
from src.agents.research_agent import research_agent
from src.config.settings import settings
from src.database.connection import AsyncSessionFactory, init_db
from src.database.models import Project, ProjectFile
from src.tools.api_connector import api_connector
from src.utils.logger import get_logger

try:
    from src.tools.web_search import web_search_tool as _web_search_tool
    _web_search_available = True
except Exception:
    _web_search_tool = None  # type: ignore[assignment]
    _web_search_available = False
from src.utils.api_analyzer import api_analyzer
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
    additional_context: str = ""
    reference_context: str = ""


class ReferenceRequest(BaseModel):
    """Request body for the POST /analyze-reference endpoint."""
    url: str | None = None
    problem_statement: str
    base64_image: str | None = None


class AnalyzeRequest(BaseModel):
    """Request body for the POST /analyze endpoint."""
    problem_statement: str


class PlanRequest(BaseModel):
    """Request body for the POST /plan endpoint."""
    problem_statement: str
    use_threejs: bool = False
    use_gsap: bool = False
    use_reactbits: bool = False


class ClarifyRequest(BaseModel):
    """Request body for the POST /clarify endpoint."""
    problem_statement: str


class DesignRequest(BaseModel):
    """Request body for the POST /design endpoint."""
    component_name: str
    description: str
    style_preferences: str = ""
    framework: str = "react"
    reference_context: str = ""


class SaveComponentRequest(BaseModel):
    """Request body for the POST /save-component endpoint."""
    project_path: str
    component_name: str
    code: str
    framework: str = "react"


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
    "https://nexus-flow-ai-dashboard.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── DB dependency ─────────────────────────────────────────────────────────────


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a transactional async database session for FastAPI endpoints."""
    session: AsyncSession = AsyncSessionFactory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


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
            {"name": "UIDesignAgent",    "type": "UIDesignAgent",    "status": "available"},
            {"name": "DebuggingAgent",   "type": "DebuggingAgent",   "status": "available"},
        ],
        "total": 8,
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
async def build(request: BuildRequest, db: AsyncSession = Depends(get_db)) -> dict:
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
        "reference_context": request.reference_context,
    }

    problem = request.problem_statement
    if request.additional_context.strip():
        problem = f"{request.additional_context.strip()}\n\n{problem}"
    if len(problem) > 3000:
        problem = problem[:3000] + "..."

    logger.info("POST /build — problem=%r options=%s", problem[:120], options)

    try:
        try:
            result = await full_project_generator.generate(problem, options, db)
        except Exception as gen_exc:
            logger.exception("POST /build — generate() raised: %s", gen_exc)
            return {"status": "error", "message": str(gen_exc), "project_name": "", "project_id": None, "total_files": 0, "setup_instructions": ""}

        logger.info(
            "POST /build — status=%s project=%r id=%s total_files=%s",
            result["status"], result.get("project_name"), result.get("project_id"), result.get("total_files"),
        )
        return {
            "status": result["status"],
            "message": result.get("error", ""),
            "project_id": result.get("project_id"),
            "project_name": result.get("project_name", ""),
            "total_files": result.get("total_files", 0),
            "setup_instructions": result.get("setup_instructions", ""),
            "debug_fixes_applied": result.get("debug_fixes_applied", 0),
            "debug_remaining_errors": result.get("debug_remaining_errors", []),
        }

    except Exception as exc:
        logger.exception("POST /build unhandled error: %s", exc)
        return {"status": "error", "message": str(exc), "project_name": "", "project_id": None, "total_files": 0, "setup_instructions": ""}


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


@app.post("/clarify")
async def clarify(request: ClarifyRequest) -> dict:
    """Generate clarifying questions for a problem statement.

    Makes a single LLM call to produce 3-5 structured questions that help
    the user refine their requirements before the build pipeline runs.

    Args:
        request: JSON body with ``problem_statement``.

    Returns:
        ``{status, questions}`` where questions is a list of question objects.
    """
    logger.info("POST /clarify — problem=%r", request.problem_statement[:120])

    system_prompt = (
        "You are a senior software architect. Analyze the problem statement and generate 3-5 clarifying questions "
        "that would help build a better product. Return ONLY a valid JSON array of objects with this exact structure "
        "— no markdown, no explanation, no text outside the JSON:\n"
        "[\n"
        "  {\n"
        '    "id": "auth_type",\n'
        '    "question": "What type of authentication do you need?",\n'
        '    "type": "single_select",\n'
        '    "options": ["No auth", "Email/Password", "Google OAuth", "JWT tokens"],\n'
        '    "default": "Email/Password",\n'
        '    "importance": "required"\n'
        "  }\n"
        "]\n"
        "type must be exactly one of: single_select, multi_select, text\n"
        "importance must be exactly one of: required, optional\n"
        "options must be a list of strings for single_select and multi_select, or null for text\n"
        "Focus on: authentication needs, UI complexity, specific features, third-party integrations, deployment target.\n"
        "If the problem statement mentions a reference website or image has already been provided, do NOT ask for website links or design references again. "
        "Focus questions on functionality, features, and technical requirements only."
    )

    user_prompt = (
        f"Problem statement: {request.problem_statement}\n\n"
        "Return ONLY the JSON array of clarifying questions."
    )

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

        questions = json.loads(raw.strip())
        return {"status": "success", "questions": questions}

    except json.JSONDecodeError as exc:
        logger.error("POST /clarify — JSON parse error: %s", exc)
        return {"status": "error", "message": f"JSON parse error: {exc}"}
    except Exception as exc:
        logger.exception("POST /clarify unhandled error: %s", exc)
        return {"status": "error", "message": str(exc)}


_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_API_VERSION = "2023-06-01"
_DESIGN_MODEL = "claude-sonnet-4-20250514"


@app.post("/analyze-reference")
async def analyze_reference(request: ReferenceRequest) -> dict:
    """Analyze a reference website or image for its design system using LLM.

    Accepts either a URL (fetched via Tavily then LLM-analyzed) or a base64
    image (sent to Groq's vision model, with text-only fallback).

    Args:
        request: JSON body with ``url``, ``base64_image``, ``problem_statement``.

    Returns:
        ``{status, reference_context, analysis, summary}`` on success,
        or ``{status, message}`` on failure.
    """
    logger.info("POST /analyze-reference — url=%r has_image=%s", request.url, bool(request.base64_image))

    if not request.url and not request.base64_image:
        return {"status": "error", "message": "Provide a url or base64_image."}

    system_prompt = (
        "You are a UI/UX analyst. Analyze the provided design and extract:\n"
        "1. Color palette (primary, secondary, accent colors as hex or descriptive names)\n"
        "2. Typography style (font families, weights, sizing)\n"
        "3. Layout style (minimal, dense, card-based, sidebar, etc)\n"
        "4. Animation style (subtle, heavy, none, micro-interactions)\n"
        "5. Component patterns (navigation style, card style, button style)\n"
        "6. Overall aesthetic (dark, light, glassmorphism, neumorphism, flat, etc)\n"
        "Return ONLY a valid JSON object with these exact keys: "
        "colors, typography, layout, animations, components, aesthetic, summary\n"
        "No markdown, no explanation, no text outside the JSON."
    )

    raw_content: str | None = None

    # ── Image path ────────────────────────────────────────────────────────────
    if request.base64_image:
        raw_b64 = request.base64_image
        mime_type = "image/jpeg"
        if raw_b64.startswith("data:"):
            header, _, raw_b64 = raw_b64.partition(",")
            mime_type = header.removeprefix("data:").removesuffix(";base64") or mime_type

        if settings.GROQ_API_KEY:
            try:
                vision_messages: list[dict] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Analyze this UI design image and extract the design system. Return ONLY the JSON object."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{raw_b64}"}},
                    ]},
                ]
                async with httpx.AsyncClient(timeout=30.0) as vision_client:
                    resp = await vision_client.post(
                        _GROQ_CHAT_URL,
                        headers={
                            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={"model": _GROQ_VISION_MODEL, "messages": vision_messages, "max_tokens": 1024},
                    )
                if resp.status_code == 200:
                    raw_content = resp.json()["choices"][0]["message"]["content"]
                    logger.info("POST /analyze-reference — vision call succeeded (%d chars)", len(raw_content))
                else:
                    logger.warning("POST /analyze-reference — vision call failed (%d): %s", resp.status_code, resp.text[:200])
            except Exception as exc:
                logger.warning("POST /analyze-reference — vision call error: %s", exc)

        if raw_content is None:
            llm_result = await api_connector.call_groq(
                prompt="A UI design image was uploaded for analysis. Provide a comprehensive design system analysis based on common modern UI patterns. Return ONLY the JSON object.",
                system_prompt=system_prompt,
            )
            if llm_result["status"] == "success":
                raw_content = llm_result["content"]
            else:
                return {"status": "error", "message": llm_result.get("error", "LLM call failed")}

    # ── URL path ──────────────────────────────────────────────────────────────
    else:
        web_content = ""
        if request.url and _web_search_available and _web_search_tool is not None:
            try:
                loop = asyncio.get_event_loop()
                search_result = await loop.run_in_executor(
                    None, _web_search_tool.search_and_summarize, request.url
                )
                web_content = search_result.get("summary", "")
                logger.info("POST /analyze-reference — fetched %d chars of web content", len(web_content))
            except Exception as exc:
                logger.warning("POST /analyze-reference — web search failed: %s", exc)

        if not web_content:
            web_content = f"Website URL: {request.url}\nProject context: {request.problem_statement}"

        user_prompt = (
            f"Website URL: {request.url}\n\n"
            f"Website content:\n{web_content[:3000]}\n\n"
            "Return ONLY the JSON object."
        )
        llm_result = await api_connector.call_groq(prompt=user_prompt, system_prompt=system_prompt)
        if llm_result["status"] != "success":
            return {"status": "error", "message": llm_result.get("error", "LLM call failed")}
        raw_content = llm_result["content"]

    if not raw_content:
        return {"status": "error", "message": "LLM returned no content."}

    try:
        raw = raw_content.strip()
        for fence in ("```json", "```"):
            if raw.startswith(fence):
                raw = raw[len(fence):]
                break
        if raw.endswith("```"):
            raw = raw[:-3]

        analysis = json.loads(raw.strip())

        source_label = (
            "image upload"
            if request.base64_image and not request.url
            else f"({request.url})"
        )
        reference_context = (
            f"Reference design analysis {source_label}:\n"
            f"- Colors: {analysis.get('colors', 'N/A')}\n"
            f"- Typography: {analysis.get('typography', 'N/A')}\n"
            f"- Layout: {analysis.get('layout', 'N/A')}\n"
            f"- Animations: {analysis.get('animations', 'N/A')}\n"
            f"- Components: {analysis.get('components', 'N/A')}\n"
            f"- Aesthetic: {analysis.get('aesthetic', 'N/A')}\n"
            f"Summary: {analysis.get('summary', 'N/A')}"
        )

        logger.info("POST /analyze-reference — analysis complete: aesthetic=%r", analysis.get("aesthetic"))
        return {
            "status": "success",
            "reference_context": reference_context,
            "analysis": analysis,
            "summary": analysis.get("summary", ""),
        }

    except json.JSONDecodeError as exc:
        logger.error("POST /analyze-reference — JSON parse error: %s", exc)
        return {"status": "error", "message": f"JSON parse error: {exc}"}
    except Exception as exc:
        logger.exception("POST /analyze-reference unhandled error: %s", exc)
        return {"status": "error", "message": str(exc)}


@app.get("/projects")
async def get_projects(db: AsyncSession = Depends(get_db)) -> list[dict]:
    """List all generated projects from the database."""
    stmt = (
        select(Project, sql_func.count(ProjectFile.id).label("file_count"))
        .outerjoin(ProjectFile, ProjectFile.project_id == Project.id)
        .group_by(Project.id)
        .order_by(Project.created_at.desc())
    )
    rows = (await db.execute(stmt)).all()
    return [
        {
            "id": row[0].id,
            "name": row[0].name,
            "description": row[0].description,
            "status": row[0].status,
            "created_at": row[0].created_at.isoformat(),
            "file_count": row[1],
        }
        for row in rows
    ]


@app.get("/projects/{project_id}")
async def get_project(project_id: int, db: AsyncSession = Depends(get_db)) -> dict:
    """Return a single project with all its files."""
    stmt = select(Project).options(selectinload(Project.files)).where(Project.id == project_id)
    project = (await db.execute(stmt)).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "status": project.status,
        "setup_instructions": project.setup_instructions,
        "created_at": project.created_at.isoformat(),
        "files": [
            {"id": f.id, "path": f.file_path, "content": f.content, "file_type": f.file_type}
            for f in project.files
        ],
    }


@app.get("/projects/{project_id}/files")
async def get_project_files(project_id: int, db: AsyncSession = Depends(get_db)) -> list[dict]:
    """Return all files for a project (without full project metadata)."""
    stmt = select(Project).options(selectinload(Project.files)).where(Project.id == project_id)
    project = (await db.execute(stmt)).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return [
        {"id": f.id, "file_path": f.file_path, "content": f.content, "file_type": f.file_type}
        for f in project.files
    ]


@app.get("/projects/{project_id}/download")
async def download_project(project_id: int, db: AsyncSession = Depends(get_db)) -> StreamingResponse:
    """Stream a ZIP archive containing all project files."""
    stmt = select(Project).options(selectinload(Project.files)).where(Project.id == project_id)
    project = (await db.execute(stmt)).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in project.files:
            zf.writestr(f.file_path, f.content)
    buf.seek(0)

    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in project.name)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.zip"'},
    )


@app.delete("/projects/{project_id}")
async def delete_project(project_id: int, db: AsyncSession = Depends(get_db)) -> dict:
    """Delete a project and all its files from the database."""
    project = (await db.execute(select(Project).where(Project.id == project_id))).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    name = project.name
    await db.delete(project)
    await db.commit()
    logger.info("DELETE /projects/%d — deleted project %r", project_id, name)
    return {"status": "success", "message": f"Project '{name}' deleted"}


@app.post("/design")
async def design(request: DesignRequest) -> dict:
    """Generate a UI component using the Anthropic Claude API.

    Sends a single request to Claude with a design-focused system prompt and
    returns the full component source code.

    Args:
        request: JSON body with ``component_name``, ``description``,
            ``style_preferences``, ``framework``, and ``reference_context``.

    Returns:
        ``{status, component_name, code, framework}`` on success,
        or ``{status, message}`` on failure.
    """
    logger.info(
        "POST /design — component=%r framework=%r", request.component_name, request.framework,
    )

    if not settings.ANTHROPIC_API_KEY:
        return {"status": "error", "message": "ANTHROPIC_API_KEY is not configured."}

    system_prompt = (
        "You are an expert UI/UX designer and React developer. Generate beautiful, production-ready UI components. "
        "Always use TailwindCSS for styling. Make components visually stunning with proper animations, gradients, "
        "and modern design patterns. Return ONLY the complete component code with no explanation."
    )

    ref_part = (
        f"\nReference design context:\n{request.reference_context}"
        if request.reference_context.strip()
        else ""
    )
    user_prompt = (
        f"Create a {request.framework} component called {request.component_name}.\n"
        f"Description: {request.description}\n"
        f"Style preferences: {request.style_preferences or 'modern dark theme'}"
        f"{ref_part}"
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                _ANTHROPIC_MESSAGES_URL,
                headers={
                    "x-api-key": settings.ANTHROPIC_API_KEY,
                    "anthropic-version": _ANTHROPIC_API_VERSION,
                    "content-type": "application/json",
                },
                json={
                    "model": _DESIGN_MODEL,
                    "max_tokens": 4096,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            )

        if not resp.is_success:
            logger.error("POST /design — Claude API error %d: %s", resp.status_code, resp.text[:300])
            return {
                "status": "error",
                "message": f"Claude API error {resp.status_code}: {resp.text[:200]}",
            }

        data = resp.json()
        code = data.get("content", [{}])[0].get("text", "")
        logger.info("POST /design — generated %d chars for %r", len(code), request.component_name)
        return {
            "status": "success",
            "component_name": request.component_name,
            "code": code,
            "framework": request.framework,
        }

    except Exception as exc:
        logger.exception("POST /design unhandled error: %s", exc)
        return {"status": "error", "message": str(exc)}


@app.post("/save-component")
async def save_component(request: SaveComponentRequest) -> dict:
    """Save a generated component file into an existing project.

    For React, writes to ``project/frontend/src/components/{Name}.tsx``.
    For HTML, writes to ``project/frontend/public/{Name}.html``.

    Args:
        request: JSON body with ``project_path``, ``component_name``,
            ``code``, and ``framework``.

    Returns:
        ``{status, saved_path}`` on success, ``{status, message}`` on failure.
    """
    logger.info(
        "POST /save-component — project=%r component=%r framework=%r",
        request.project_path, request.component_name, request.framework,
    )

    project = Path(request.project_path)
    if not project.exists():
        return {"status": "error", "message": f"Project path does not exist: {request.project_path}"}

    slug = request.component_name.replace(" ", "").replace("-", "")
    if request.framework == "react":
        target_dir = project / "frontend" / "src" / "components"
        save_path = target_dir / f"{slug}.tsx"
    else:
        target_dir = project / "frontend" / "public"
        save_path = target_dir / f"{slug}.html"

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        save_path.write_text(request.code, encoding="utf-8")
        logger.info("POST /save-component — saved to %s", save_path)
        return {"status": "success", "saved_path": str(save_path)}
    except Exception as exc:
        logger.exception("POST /save-component unhandled error: %s", exc)
        return {"status": "error", "message": str(exc)}


# ── Entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
