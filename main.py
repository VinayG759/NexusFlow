"""NexusFlow — application entry point."""

import io
import json
import asyncio
import httpx
import zipfile

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func as sql_func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.config.settings import settings
from src.database.connection import AsyncSessionFactory, init_db
from src.database.models import Project, ProjectFile, ErrorPattern, BuildAttempt, TrainingExample
from src.tools.api_connector import api_connector
from src.utils.logger import get_logger

try:
    from src.tools.web_search import web_search_tool as _web_search_tool
    _web_search_available = True
except Exception:
    _web_search_tool = None  # type: ignore[assignment]
    _web_search_available = False

from src.utils.api_analyzer import api_analyzer
from src.rag.rag_retriever import rag_retriever
from src.utils.full_project_generator import full_project_generator
from src.utils.training_collector import training_collector, KNOWN_ERROR_PATTERNS
from src.utils.training_data import TRAINING_DATA

logger = get_logger(__name__)

# ── Request / response models ─────────────────────────────────────────────────


class BuildRequest(BaseModel):
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
    url: str | None = None
    problem_statement: str
    base64_image: str | None = None


class AnalyzeRequest(BaseModel):
    problem_statement: str


class PlanRequest(BaseModel):
    problem_statement: str
    use_threejs: bool = False
    use_gsap: bool = False
    use_reactbits: bool = False


class ClarifyRequest(BaseModel):
    problem_statement: str


# ── Lifespan ──────────────────────────────────────────────────────────────────


async def seed_all_error_patterns(db: AsyncSession) -> None:
    for p in KNOWN_ERROR_PATTERNS:
        result = await db.execute(
            select(ErrorPattern).where(ErrorPattern.error_type == p["error_type"])
        )
        if not result.scalar_one_or_none():
            db.add(ErrorPattern(
                error_type=p["error_type"],
                error_pattern=p["pattern"],
                fix_strategy=p["fix_strategy"],
            ))
    await db.commit()
    logger.info("Upserted error patterns (total defined: %d)", len(KNOWN_ERROR_PATTERNS))


async def seed_all_training_data(db: AsyncSession) -> None:
    from sqlalchemy import func
    result = await db.execute(select(func.count()).select_from(TrainingExample))
    count = result.scalar()
    if count < len(TRAINING_DATA):
        for item in TRAINING_DATA:
            db.add(TrainingExample(
                input_prompt=item.get("error", ""),
                error_context=item.get("error_type", ""),
                correct_output=item.get("fix", ""),
                example_type=item.get("error_type", "unknown"),
                quality_score=1.0 if item.get("instant_fix") else 0.8,
            ))
        await db.commit()
        logger.info("Seeded %d training examples", len(TRAINING_DATA))


_startup_complete = False


async def _background_startup() -> None:
    global _startup_complete
    try:
        db = AsyncSessionFactory()
        try:
            await seed_all_error_patterns(db)
            await seed_all_training_data(db)
        finally:
            await db.close()
        logger.info("Background startup complete")
    except Exception as e:
        logger.warning("Background startup failed: %s", e)
    _startup_complete = True


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting %s v%s...", settings.APP_NAME, settings.APP_VERSION)
    await init_db()
    logger.info("%s is ready to accept requests.", settings.APP_NAME)
    asyncio.create_task(_background_startup())
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── DB dependency ─────────────────────────────────────────────────────────────


async def get_db() -> AsyncGenerator[AsyncSession, None]:
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
    return {"message": "Welcome to NexusFlow"}


@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@app.get("/agents")
async def list_agents() -> dict:
    return {
        "status": "ok",
        "agents": [
            {"name": "UIDesignAgent",  "type": "UIDesignAgent",  "status": "available"},
            {"name": "DebuggingAgent", "type": "DebuggingAgent", "status": "available"},
            {"name": "FileAgent",      "type": "FileAgent",      "status": "available"},
        ],
        "total": 3,
    }


@app.post("/build")
async def build(request: BuildRequest, db: AsyncSession = Depends(get_db)) -> dict:
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
            "debug_fix_summary": result.get("debug_fix_summary", ""),
            "backend_verified": result.get("backend_verified", False),
            "frontend_verified": result.get("frontend_verified", False),
        }

    except Exception as exc:
        logger.exception("POST /build unhandled error: %s", exc)
        return {"status": "error", "message": str(exc), "project_name": "", "project_id": None, "total_files": 0, "setup_instructions": ""}


@app.post("/analyze")
async def analyze(request: AnalyzeRequest) -> dict:
    logger.info("POST /analyze — problem=%r", request.problem_statement[:120])

    try:
        result = await api_analyzer.analyze(request.problem_statement)

        if result.get("status") != "success":
            return {"status": "error", "message": result.get("error", "Analysis failed.")}

        apis: list[dict] = result["apis"]

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


@app.post("/analyze-reference")
async def analyze_reference(request: ReferenceRequest) -> dict:
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
    stmt = select(Project).options(selectinload(Project.files)).where(Project.id == project_id)
    project = (await db.execute(stmt)).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in project.files:
            zf.writestr(f.file_path, f.content)
        project_slug = project.name.replace("-", "_").replace(" ", "_")
        zf.writestr("setup.bat", _build_setup_bat(project.name, project_slug))
        zf.writestr("setup.sh", _build_setup_sh(project.name, project_slug))
    buf.seek(0)

    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in project.name)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.zip"'},
    )


@app.delete("/projects/{project_id}")
async def delete_project(project_id: int, db: AsyncSession = Depends(get_db)) -> dict:
    project = (await db.execute(select(Project).where(Project.id == project_id))).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    name = project.name
    await db.delete(project)
    await db.commit()
    logger.info("DELETE /projects/%d — deleted project %r", project_id, name)
    return {"status": "success", "message": f"Project '{name}' deleted"}


# ── Training data endpoints ───────────────────────────────────────────────────


@app.get("/training/stats")
async def get_training_stats(db: AsyncSession = Depends(get_db)) -> dict:
    return await training_collector.get_training_stats(db)


@app.get("/training/examples")
async def get_training_examples(db: AsyncSession = Depends(get_db)) -> list[dict]:
    result = await db.execute(
        select(TrainingExample).order_by(TrainingExample.id.desc()).limit(50)
    )
    examples = result.scalars().all()
    return [
        {
            "id": e.id,
            "input_prompt": e.input_prompt[:100],
            "example_type": e.example_type,
            "quality_score": e.quality_score,
            "created_at": str(e.created_at),
        }
        for e in examples
    ]


@app.post("/training/export")
async def export_training_data(db: AsyncSession = Depends(get_db)) -> StreamingResponse:
    import io
    result = await db.execute(
        select(TrainingExample).where(TrainingExample.quality_score >= 0.8)
    )
    examples = result.scalars().all()

    lines = []
    for e in examples:
        lines.append(json.dumps({
            "messages": [
                {
                    "role": "system",
                    "content": "You are NexusFlow, an expert full-stack developer that generates complete, working FastAPI + React TypeScript applications.",
                },
                {"role": "user", "content": e.input_prompt},
                {"role": "assistant", "content": e.correct_output},
            ]
        }))

    content = "\n".join(lines)
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": "attachment; filename=nexusflow_training.jsonl"},
    )


@app.post("/training/export-jsonl")
async def export_jsonl(db: AsyncSession = Depends(get_db)):
    from src.utils.finetune_pipeline import finetune_pipeline
    file_path = await finetune_pipeline.export_training_data(db)
    validation = finetune_pipeline.validate_jsonl(file_path)
    return {
        "file": str(file_path),
        "validation": validation,
        "next_step": "POST /training/submit-finetune when ready_for_training=true"
    }


@app.post("/training/submit-finetune")
async def submit_finetune(db: AsyncSession = Depends(get_db)):
    from src.utils.finetune_pipeline import finetune_pipeline
    file_path = await finetune_pipeline.export_training_data(db)
    result = await finetune_pipeline.submit_to_groq(file_path)
    return result


@app.get("/training/finetune-status/{job_id}")
async def finetune_status(job_id: str):
    from src.utils.finetune_pipeline import finetune_pipeline
    return await finetune_pipeline.check_job_status(job_id)


@app.post("/training/activate-model")
async def activate_model(model_id: str):
    from src.utils.finetune_pipeline import finetune_pipeline
    return finetune_pipeline.activate_fine_tuned_model(model_id)


# ── Setup script helpers ──────────────────────────────────────────────────────


def _build_setup_bat(project_name: str, project_slug: str) -> str:
    return f"""@echo off
echo ========================================
echo    Setting up {project_name}
echo ========================================
echo.

echo [1/5] Creating PostgreSQL database...
psql -U postgres -c "SELECT 1" -d {project_slug} >nul 2>&1 || psql -U postgres -c "CREATE DATABASE {project_slug};"

echo [2/5] Setting up Python virtual environment...
cd backend
python -m venv venv
call venv\\Scripts\\activate

echo [3/5] Installing backend dependencies...
pip install -r requirements.txt

echo [4/5] Installing frontend dependencies...
cd ..\\frontend
npm install --legacy-peer-deps

echo [5/5] Starting both servers...
cd ..
start cmd /k "cd backend && call venv\\Scripts\\activate && uvicorn main:app --host 0.0.0.0 --port 8001 --reload"
timeout /t 3 /nobreak >nul
start cmd /k "cd frontend && npm run dev"
timeout /t 5 /nobreak >nul
start http://localhost:5173

echo.
echo ========================================
echo App is running!
echo Backend:  http://localhost:8001
echo Frontend: http://localhost:5173
echo API Docs: http://localhost:8001/docs
echo ========================================
"""


def _build_setup_sh(project_name: str, project_slug: str) -> str:
    return f"""#!/bin/bash
echo "========================================"
echo "   Setting up {project_name}"
echo "========================================"

echo "[1/5] Creating PostgreSQL database..."
psql -U postgres -c "CREATE DATABASE {project_slug};" 2>/dev/null || true

echo "[2/5] Setting up Python virtual environment..."
cd backend
python3 -m venv venv
source venv/bin/activate

echo "[3/5] Installing backend dependencies..."
pip install -r requirements.txt

echo "[4/5] Installing frontend dependencies..."
cd ../frontend
npm install --legacy-peer-deps

echo "[5/5] Starting both servers..."
cd ..
osascript -e 'tell app "Terminal" to do script "cd '$(pwd)'/backend && source venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8001 --reload"' 2>/dev/null || \\
  gnome-terminal -- bash -c "cd backend && source venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8001 --reload; exec bash" 2>/dev/null || \\
  (cd backend && source venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8001 --reload &)

sleep 3
npm run dev --prefix frontend &
sleep 3
open http://localhost:5173 2>/dev/null || xdg-open http://localhost:5173 2>/dev/null

echo ""
echo "========================================"
echo "App is running!"
echo "Backend:  http://localhost:8001"
echo "Frontend: http://localhost:5173"
echo "API Docs: http://localhost:8001/docs"
echo "========================================"
"""


# ── Entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
