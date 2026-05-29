"""Integration tests for NexusFlow API routes.

All tests use the in-memory SQLite DB and patched lifespan from conftest.py.
Endpoints that invoke the LLM have api_connector.call_groq mocked so no
real API key is needed.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch

from src.database.models import Project, ProjectFile


# ── Simple / stateless endpoints ──────────────────────────────────────────────


async def test_root(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert r.json()["message"] == "Welcome to NexusFlow"


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data


async def test_agents(client):
    r = await client.get("/agents")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["total"] == 6
    names = {a["name"] for a in data["agents"]}
    assert names == {"ProjectGenerator", "DebuggingAgent", "FileAgent", "UIDesignAgent", "RAGRetriever", "DeployPipeline"}


# ── Build endpoints ───────────────────────────────────────────────────────────


async def test_build_returns_job_id(client):
    with patch("main.run_build_job", new_callable=AsyncMock):
        r = await client.post("/build", json={"problem_statement": "a todo app"})
    assert r.status_code == 200
    data = r.json()
    assert "job_id" in data
    assert data["status"] == "running"


async def test_build_status_unknown_job(client):
    r = await client.get("/build/status/nonexistent-id")
    assert r.status_code == 404


async def test_build_status_known_job(client):
    # Patch the background function so the job stays in "running" state
    # without making real LLM calls or touching PostgreSQL.
    with patch("main.run_build_job", new_callable=AsyncMock):
        post = await client.post("/build", json={"problem_statement": "a todo app"})
        job_id = post.json()["job_id"]
        r = await client.get(f"/build/status/{job_id}")
    assert r.status_code == 200
    assert r.json()["job_id"] == job_id
    assert r.json()["status"] == "running"


async def test_build_feedback_unknown_job(client):
    r = await client.post(
        "/build/nonexistent/feedback",
        json={"file_path": "backend/main.py", "feedback": "add CORS"},
    )
    assert r.status_code == 404


async def test_build_feedback_job_not_complete(client):
    """Feedback on a still-running job must return 400."""
    with patch("main.run_build_job", new_callable=AsyncMock):
        post = await client.post("/build", json={"problem_statement": "a todo app"})
        job_id = post.json()["job_id"]
        r = await client.post(
            f"/build/{job_id}/feedback",
            json={"file_path": "backend/main.py", "feedback": "add CORS"},
        )
    assert r.status_code == 400


# ── Project endpoints ─────────────────────────────────────────────────────────


async def test_get_projects_empty(client):
    r = await client.get("/projects")
    assert r.status_code == 200
    assert r.json() == []


async def test_get_project_not_found(client):
    r = await client.get("/projects/9999")
    assert r.status_code == 404


async def test_get_project_files_not_found(client):
    r = await client.get("/projects/9999/files")
    assert r.status_code == 404


async def test_delete_project_not_found(client):
    r = await client.delete("/projects/9999")
    assert r.status_code == 404


async def test_project_lifecycle(client, db_session):
    """Create a project + file in the DB, then fetch and delete via API."""
    project = Project(name="test-app", description="a test project", status="ready")
    db_session.add(project)
    await db_session.flush()
    db_session.add(ProjectFile(
        project_id=project.id,
        file_path="backend/main.py",
        content="print('hello')",
        file_type="python",
    ))
    await db_session.commit()

    # List
    r = await client.get("/projects")
    assert r.status_code == 200
    assert len(r.json()) == 1

    # Detail
    r = await client.get(f"/projects/{project.id}")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "test-app"
    assert len(data["files"]) == 1

    # Files
    r = await client.get(f"/projects/{project.id}/files")
    assert r.status_code == 200
    assert r.json()[0]["file_path"] == "backend/main.py"

    # Delete
    r = await client.delete(f"/projects/{project.id}")
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    # Gone
    r = await client.get(f"/projects/{project.id}")
    assert r.status_code == 404


async def test_download_project(client, db_session):
    project = Project(name="dl-app", description="", status="ready")
    db_session.add(project)
    await db_session.flush()
    db_session.add(ProjectFile(
        project_id=project.id,
        file_path="backend/main.py",
        content="print('hello')",
    ))
    await db_session.commit()

    r = await client.get(f"/projects/{project.id}/download")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"


# ── Preview endpoints ─────────────────────────────────────────────────────────


async def test_start_preview_returns_preview_id(client):
    r = await client.post("/projects/1/preview/start")
    assert r.status_code == 200
    data = r.json()
    assert "preview_id" in data
    assert data["status"] == "building"


async def test_preview_status_unknown(client):
    r = await client.get("/projects/1/preview/status/no-such-id")
    assert r.status_code == 404


async def test_preview_status_known(client):
    start = await client.post("/projects/1/preview/start")
    preview_id = start.json()["preview_id"]
    r = await client.get(f"/projects/1/preview/status/{preview_id}")
    assert r.status_code == 200
    assert r.json()["preview_id"] == preview_id


# ── /plan and /clarify (LLM mocked) ──────────────────────────────────────────


async def test_plan_returns_structure(client):
    plan_json = json.dumps({
        "project_name": "todo-app",
        "description": "A simple todo application",
        "tech_stack": {
            "backend": ["FastAPI"],
            "frontend": ["React 18"],
            "database": ["PostgreSQL"],
            "devops": ["Docker"],
        },
        "file_structure": [],
        "apis_required": [],
        "estimated_files": 10,
        "complexity": "simple",
        "estimated_build_time": "1-2 minutes",
    })
    with patch("main.api_connector.call_groq", new_callable=AsyncMock) as mock_groq:
        mock_groq.return_value = {"status": "success", "content": plan_json}
        r = await client.post("/plan", json={"problem_statement": "a todo app"})

    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["project_name"] == "todo-app"
    assert data["complexity"] == "simple"


async def test_plan_handles_llm_failure(client):
    with patch("main.api_connector.call_groq", new_callable=AsyncMock) as mock_groq:
        mock_groq.return_value = {"status": "error", "error": "rate limited"}
        r = await client.post("/plan", json={"problem_statement": "a todo app"})

    assert r.status_code == 200
    assert r.json()["status"] == "error"


async def test_clarify_returns_questions(client):
    questions_json = json.dumps([
        {
            "id": "auth_type",
            "question": "What type of auth?",
            "type": "single_select",
            "options": ["JWT", "OAuth"],
            "default": "JWT",
            "importance": "required",
        }
    ])
    with patch("main.api_connector.call_groq", new_callable=AsyncMock) as mock_groq:
        mock_groq.return_value = {"status": "success", "content": questions_json}
        r = await client.post("/clarify", json={"problem_statement": "a todo app"})

    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert len(data["questions"]) == 1
    assert data["questions"][0]["id"] == "auth_type"


# ── /analyze-reference validation ────────────────────────────────────────────


async def test_analyze_reference_requires_url_or_image(client):
    r = await client.post("/analyze-reference", json={"problem_statement": "build a clone"})
    assert r.status_code == 200
    assert r.json()["status"] == "error"

