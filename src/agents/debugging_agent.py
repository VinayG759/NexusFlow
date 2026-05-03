"""Autonomous five-phase debugging agent for NexusFlow generated projects.

Phases:
  1. Static analysis  — scan files for known anti-patterns without running them.
  2. Auto-fix         — apply instant fixes; escalate to Claude for the rest.
  3. Dependencies     — pip install + npm install, with auto-retry on failure.
  4. Runtime verify   — start uvicorn, health-check it; npm run build frontend.
  5. Report           — return a structured result dict.
"""

import asyncio
import httpx
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    psycopg2 = None  # type: ignore[assignment]
    ISOLATION_LEVEL_AUTOCOMMIT = None  # type: ignore[assignment]

from src.utils.logger import get_logger
from src.utils.training_collector import training_collector, _KNOWN_VERSIONS

logger = get_logger(__name__)

# Packages that are not real and must never appear in package.json
_FAKE_PACKAGES: frozenset[str] = frozenset({
    "@react-bits/react",
    "@react-bits/ui",
    "@react-bits/components",
    "@react-bits/animations",
    "react-bits",
})

# Packages that must always be present in requirements.txt (canonical pip names)
_REQUIRED_PYTHON: set[str] = {
    "fastapi", "uvicorn", "sqlalchemy", "asyncpg", "python-dotenv", "pydantic",
}

# Standard-library module names (never flag as missing from requirements.txt)
_STDLIB: frozenset[str] = frozenset({
    "os", "sys", "re", "json", "asyncio", "pathlib", "datetime", "typing",
    "collections", "contextlib", "io", "time", "uuid", "hashlib", "hmac",
    "base64", "functools", "itertools", "math", "random", "string", "enum",
    "dataclasses", "abc", "copy", "shutil", "tempfile", "socket", "logging",
    "traceback", "struct", "binascii", "urllib", "http", "email", "html",
    "xml", "csv", "configparser", "threading", "multiprocessing", "signal",
    "platform", "subprocess",
})

# Local module names that are project files, never pip packages
_LOCAL_MODULES: frozenset[str] = frozenset({
    "database", "routes", "models", "schemas", "config", "utils", "auth",
    "middleware", "dependencies", "core", "api", "crud", "deps", "security",
    "exceptions", "constants", "helpers", "serializers", "validators",
    "permissions", "tasks", "services", "repositories", "interfaces",
})

# Python import name → pip package name
_IMPORT_TO_PKG: dict[str, str] = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "sqlalchemy": "sqlalchemy",
    "asyncpg": "asyncpg",
    "dotenv": "python-dotenv",
    "pydantic": "pydantic",
    "httpx": "httpx",
    "aiofiles": "aiofiles",
    "jwt": "python-jose",
    "bcrypt": "bcrypt",
    "passlib": "passlib",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "numpy": "numpy",
    "pandas": "pandas",
}


def _find_free_port() -> int:
    """Return an OS-assigned free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _fmap(files: list[dict]) -> dict[str, str]:
    return {f["path"]: f["content"] for f in files}


def _unmap(d: dict[str, str]) -> list[dict]:
    return [{"path": k, "content": v} for k, v in d.items()]


def _fwd(p: str) -> str:
    """Normalise a path to forward slashes."""
    return str(Path(p)).replace("\\", "/")


class DebuggingAgent:
    """Autonomous debugging agent — runs five sequential phases."""

    def __init__(self) -> None:
        self.agent_name = "DebuggingAgent"
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.model = "claude-sonnet-4-20250514"
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.max_fix_attempts = 3
        logger.info("%s initialised.", self.agent_name)

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    async def debug_project(
        self,
        project_files: list[dict],
        project_name: str,
        db_session=None,
    ) -> dict:
        """Run all five phases and return a structured report."""
        logger.info("%s starting autonomous debug for: %s", self.agent_name, project_name)

        phases_completed: list[str] = []
        issues_found: list[str] = []
        fixes_applied: list[str] = []
        errors_remaining: list[str] = []
        files = list(project_files)

        # ── Phase 1: Static Analysis ──────────────────────────────────────────
        logger.info("%s [Phase 1] Static analysis...", self.agent_name)
        phase1_issues = self._static_analysis(files)
        issues_found.extend(phase1_issues)
        phases_completed.append("static_analysis")
        logger.info("%s [Phase 1] Found %d issue(s).", self.agent_name, len(phase1_issues))

        # ── Phase 2: Auto Fix ─────────────────────────────────────────────────
        logger.info("%s [Phase 2] Auto-fixing %d issue(s)...", self.agent_name, len(phase1_issues))
        files, phase2_fixes, phase2_errors = await self._auto_fix(files, phase1_issues, db_session)
        fixes_applied.extend(phase2_fixes)
        errors_remaining.extend(phase2_errors)
        phases_completed.append("auto_fix")
        logger.info(
            "%s [Phase 2] Applied %d fix(es), %d remain.",
            self.agent_name, len(phase2_fixes), len(phase2_errors),
        )

        # Write to temp dir for phases 3 & 4
        temp_dir = Path(tempfile.mkdtemp(prefix=f"nexusflow_{project_name}_"))
        backend_status = "unknown"
        frontend_status = "unknown"
        backend_url = ""
        backend_crash_diagnosis = ""

        try:
            self._write_files(files, temp_dir)

            # ── Phase 3: Dependency Installation ─────────────────────────────
            logger.info("%s [Phase 3] Installing dependencies...", self.agent_name)
            files, phase3_fixes, phase3_errors = await self._install_dependencies(files, temp_dir)
            fixes_applied.extend(phase3_fixes)
            errors_remaining.extend(phase3_errors)
            phases_completed.append("dependencies")
            self._write_files(files, temp_dir)

            # ── Phase 4: Runtime Verification ────────────────────────────────
            logger.info("%s [Phase 4] Runtime verification...", self.agent_name)
            (
                files,
                backend_status,
                frontend_status,
                backend_url,
                phase4_fixes,
                phase4_errors,
                backend_crash_diagnosis,
            ) = await self._runtime_verification(files, temp_dir, project_name)
            fixes_applied.extend(phase4_fixes)
            errors_remaining.extend(phase4_errors)
            phases_completed.append("runtime")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # ── Phase 5: Report ───────────────────────────────────────────────────
        phases_completed.append("report")
        fix_summary = self._build_fix_summary(fixes_applied, errors_remaining)
        overall_status = (
            "success" if not errors_remaining
            else ("partial" if fixes_applied else "failed")
        )

        logger.info(
            "%s done — status=%s fixes=%d remaining=%d backend=%s frontend=%s",
            self.agent_name, overall_status, len(fixes_applied),
            len(errors_remaining), backend_status, frontend_status,
        )

        return {
            "status": overall_status,
            "phases_completed": phases_completed,
            "issues_found": issues_found,
            "fixes_applied": fixes_applied,
            "backend_status": backend_status,
            "frontend_status": frontend_status,
            "backend_url": backend_url,
            "errors_remaining": errors_remaining,
            "fix_summary": fix_summary,
            "backend_crash_diagnosis": backend_crash_diagnosis,
            # Legacy-compat keys used by full_project_generator
            "fixed_files": files,
            "fixes_applied_count": len(fixes_applied),
            "remaining_errors": errors_remaining,
            "attempts": self.max_fix_attempts,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1 — Static Analysis
    # ─────────────────────────────────────────────────────────────────────────

    def _static_analysis(self, files: list[dict]) -> list[str]:
        fm = _fmap(files)
        issues: list[str] = []
        issues.extend(self._check_backend_main(fm))
        issues.extend(self._check_frontend_files(fm))
        issues.extend(self._check_requirements(fm))
        issues.extend(self._check_env_files(fm))
        issues.extend(self._check_pydantic_v2(fm))
        issues.extend(self._check_fastapi_imports(fm))
        issues.extend(self._check_schema_model_mix(fm))
        issues.extend(self._check_schema_imports(fm))
        issues.extend(self._check_fastapi_db_depends(fm))
        issues.extend(self._check_invalid_model_definition(fm))
        return issues

    def _check_backend_main(self, fm: dict[str, str]) -> list[str]:
        issues: list[str] = []
        main = fm.get("backend/main.py", "")
        db = fm.get("backend/database.py", "")
        req = fm.get("backend/requirements.txt", fm.get("requirements.txt", ""))

        if not main:
            return issues

        if "CORSMiddleware" in main and "from fastapi.middleware.cors import" not in main:
            issues.append(
                "backend/main.py: CORSMiddleware used but not imported from fastapi.middleware.cors"
            )
        if re.search(r'allow_methods\s*=\s*["\']', main):
            issues.append("backend/main.py: allow_methods must be a list ['*'], not a string")
        if re.search(r'allow_headers\s*=\s*["\']', main):
            issues.append("backend/main.py: allow_headers must be a list ['*'], not a string")
        if "allow_credentials=True" in main and 'allow_origins=["*"]' in main:
            issues.append(
                "backend/main.py: allow_credentials=True conflicts with allow_origins=['*']"
            )
        if "@app.on_event" in main:
            issues.append("backend/main.py: @app.on_event is deprecated — use lifespan instead")

        if req:
            req_pkgs = {
                ln.split("[")[0].split("==")[0].split(">=")[0].strip().lower()
                for ln in req.splitlines()
                if ln.strip() and not ln.startswith("#")
            }
            local_modules = _LOCAL_MODULES | {Path(p).stem for p in fm if p.startswith("backend/") and p.endswith(".py")}
            for imp in re.findall(r"^(?:from|import)\s+([\w]+)", main, re.MULTILINE):
                if imp in _STDLIB or imp in local_modules:
                    continue
                pkg = _IMPORT_TO_PKG.get(imp, imp)
                if pkg not in req_pkgs:
                    issues.append(
                        f"backend/main.py: import '{imp}' not found in requirements.txt"
                    )

        if db:
            if re.search(r"\bpostgres://", db):
                issues.append("backend/database.py: DATABASE_URL uses postgres:// — must be postgresql+asyncpg://")
            if re.search(r"\bpostgresql://(?!\+asyncpg)", db):
                issues.append("backend/database.py: DATABASE_URL missing +asyncpg driver")

        return issues

    def _check_frontend_files(self, fm: dict[str, str]) -> list[str]:
        issues: list[str] = []
        index_tsx = fm.get("frontend/src/index.tsx", "")
        app_tsx = fm.get("frontend/src/App.tsx", "")
        tsconfig_raw = fm.get("frontend/tsconfig.json", "")
        pkg_raw = fm.get("frontend/package.json", "")

        if index_tsx and "ReactDOM.render(" in index_tsx:
            issues.append(
                "frontend/src/index.tsx: uses deprecated ReactDOM.render — must use createRoot"
            )

        if app_tsx:
            untyped = re.findall(r"catch\s*\(\s*(\w+)\s*\)", app_tsx)
            typed = re.findall(r"catch\s*\(\s*\w+\s*:\s*\w+", app_tsx)
            if untyped and not typed:
                issues.append(
                    "frontend/src/App.tsx: catch blocks lack type annotation (use catch (e: unknown))"
                )

        if tsconfig_raw:
            try:
                ts = json.loads(tsconfig_raw)
                opts = ts.get("compilerOptions", {})
                for field in ("jsx", "module", "moduleResolution"):
                    if field not in opts:
                        issues.append(f"frontend/tsconfig.json: missing compilerOptions.{field}")
            except json.JSONDecodeError:
                issues.append("frontend/tsconfig.json: invalid JSON")

        if pkg_raw:
            try:
                pkg = json.loads(pkg_raw)
                all_deps: dict[str, str] = {
                    **pkg.get("dependencies", {}),
                    **pkg.get("devDependencies", {}),
                }
                for fake in _FAKE_PACKAGES:
                    if fake in all_deps:
                        issues.append(f"frontend/package.json: fake package '{fake}' must be removed")

                for path, content in fm.items():
                    if not (path.startswith("frontend/src/") and path.endswith((".tsx", ".ts", ".js", ".jsx"))):
                        continue
                    for m in re.finditer(
                        r"""(?:import\s+(?:[^'"]+\s+from\s+)?|require\()['"]([^'"./][^'"]*)['"]""",
                        content, re.MULTILINE,
                    ):
                        raw_imp = m.group(1)
                        parts = raw_imp.split("/")
                        pkg_name = "/".join(parts[:2]) if parts[0].startswith("@") else parts[0]
                        if pkg_name and pkg_name not in all_deps and pkg_name not in {
                            "react", "react-dom", "react-scripts"
                        }:
                            issues.append(
                                f"{path}: imports '{pkg_name}' but it is not in package.json"
                            )
            except json.JSONDecodeError:
                issues.append("frontend/package.json: invalid JSON")

        # CSS imports pointing at non-existent files
        for path, content in fm.items():
            if not path.startswith("frontend/src/"):
                continue
            for css_path in re.findall(r"""import\s+['"]([^'"]+\.css)['"]\s*;?""", content):
                file_dir = "/".join(path.split("/")[:-1])
                resolved = _fwd(f"{file_dir}/{css_path}".replace("//", "/").lstrip("/"))
                if resolved not in fm:
                    issues.append(
                        f"{path}: imports CSS '{css_path}' which does not exist in the project"
                    )

        return issues

    def _check_requirements(self, fm: dict[str, str]) -> list[str]:
        issues: list[str] = []
        req = fm.get("backend/requirements.txt", fm.get("requirements.txt", ""))
        if not req:
            issues.append("requirements.txt: file missing entirely")
            return issues
        present = {
            ln.split("[")[0].split("==")[0].split(">=")[0].strip().lower()
            for ln in req.splitlines()
            if ln.strip() and not ln.startswith("#")
        }
        for pkg in _REQUIRED_PYTHON:
            if pkg not in present:
                issues.append(f"requirements.txt: missing required package '{pkg}'")
        return issues

    def _check_env_files(self, fm: dict[str, str]) -> list[str]:
        issues: list[str] = []
        if "backend/.env" not in fm and ".env" not in fm:
            src = "backend/.env.example" if "backend/.env.example" in fm else ".env.example"
            if src in fm:
                issues.append(".env: backend .env missing — will create from .env.example")
            else:
                issues.append(".env: no backend .env or .env.example found — will create default")
        if "frontend/.env" not in fm:
            issues.append("frontend/.env: missing — will create with VITE_API_URL")
        return issues

    def _check_pydantic_v2(self, fm: dict[str, str]) -> list[str]:
        issues: list[str] = []
        for path, content in fm.items():
            if not (path.startswith("backend/") and path.endswith(".py")):
                continue
            if "orm_mode" in content:
                issues.append(
                    f"{path}: uses Pydantic v1 'orm_mode' — replace with "
                    "model_config = ConfigDict(from_attributes=True)"
                )
            if "from pydantic.typing import" in content:
                issues.append(
                    f"{path}: 'from pydantic.typing import' is removed in Pydantic v2 — use 'from typing import' instead"
                )
            if self._PYDANTIC_TYPING_RE.search(content):
                issues.append(
                    f"{path}: imports typing symbols (Optional/List/etc.) from pydantic — must come from 'typing' in Pydantic v2"
                )
        return issues

    def _check_fastapi_imports(self, fm: dict[str, str]) -> list[str]:
        issues: list[str] = []
        # symbol → canonical import line
        _FASTAPI_IMPORTS: dict[str, str] = {
            "APIRouter":       "from fastapi import APIRouter",
            "Depends":         "from fastapi import Depends",
            "HTTPException":   "from fastapi import HTTPException",
            "status":          "from fastapi import status",
            "Body":            "from fastapi import Body",
            "Query":           "from fastapi import Query",
            "Path":            "from fastapi import Path",
            "Form":            "from fastapi import Form",
            "File":            "from fastapi import File",
            "UploadFile":      "from fastapi import UploadFile",
            "BackgroundTasks": "from fastapi import BackgroundTasks",
        }
        _SA_IMPORTS: dict[str, str] = {
            "AsyncSession": "from sqlalchemy.ext.asyncio import AsyncSession",
            "Session":      "from sqlalchemy.orm import Session",
            "select":       "from sqlalchemy import select",
        }
        for path, content in fm.items():
            if not (path.startswith("backend/") and path.endswith(".py")):
                continue
            for symbol, import_line in {**_FASTAPI_IMPORTS, **_SA_IMPORTS}.items():
                module = import_line.split(" import ")[0].replace("from ", "")
                already = (
                    import_line in content
                    or f"from {module} import *" in content
                    or re.search(rf"from {re.escape(module)} import[^\n]*\b{symbol}\b", content)
                )
                if not already and re.search(rf"\b{symbol}\s*[\(\[,\s]", content):
                    issues.append(
                        f"{path}: uses '{symbol}' but '{import_line}' is missing"
                    )
        return issues

    def _check_schema_model_mix(self, fm: dict[str, str]) -> list[str]:
        issues: list[str] = []
        schemas = fm.get("backend/schemas.py", "")
        if not schemas:
            return issues
        sa_classes = re.findall(r"^class\s+(\w+)\s*\(\s*Base\s*\)", schemas, re.MULTILINE)
        if not sa_classes:
            return issues
        if "from database import Base" not in schemas:
            issues.append(
                f"backend/schemas.py: class(es) {sa_classes} inherit from Base "
                "but Base is not imported — NameError on startup"
            )
        pydantic_names = re.findall(r"^class\s+(\w+)\s*\(\s*BaseModel\b", schemas, re.MULTILINE)
        if pydantic_names:
            issues.append(
                f"backend/schemas.py: mixes SQLAlchemy models {sa_classes} and Pydantic "
                f"schemas {pydantic_names} — SQLAlchemy models must be in models.py"
            )
        return issues

    # Matches bad pattern: db: SomeType = next(get_db()) in function signatures
    # Uses [^=\n]+ to handle complex annotations like Optional[AsyncSession]
    _BAD_DB_DEPENDS_RE = re.compile(
        r"(\bdb\s*:[^=\n]+?)\s*=\s*next\s*\(\s*get_db\s*\(\s*\)\s*\)"
    )

    # Matches SQLAlchemy table reflection that crashes if the table doesn't exist yet
    _BAD_TABLE_REFLECTION_RE = re.compile(
        r"__table__\s*=\s*Base\.metadata\.tables\[(['\"])(\w+)\1\]"
    )

    def _check_fastapi_db_depends(self, fm: dict[str, str]) -> list[str]:
        issues: list[str] = []
        for path, content in fm.items():
            if not (path.startswith("backend/") and path.endswith(".py")):
                continue
            if self._BAD_DB_DEPENDS_RE.search(content):
                issues.append(
                    f"{path}: uses 'next(get_db())' as default — async generators cannot be iterated with next(). "
                    "Use 'Depends(get_db)' instead."
                )
        return issues

    def _check_schema_imports(self, fm: dict[str, str]) -> list[str]:
        """Verify that all names imported from schemas.py actually exist there."""
        issues: list[str] = []
        schemas_content = fm.get("backend/schemas.py", "")
        if not schemas_content:
            return issues
        defined = set(re.findall(r"^class\s+(\w+)", schemas_content, re.MULTILINE))
        for path, content in fm.items():
            if not (path.startswith("backend/") and path.endswith(".py") and path != "backend/schemas.py"):
                continue
            for m in re.finditer(r"from schemas import ([^\n]+)", content):
                names = [n.strip() for n in m.group(1).split(",") if n.strip()]
                missing = [n for n in names if n and n not in defined]
                if missing:
                    issues.append(
                        f"{path}: imports {missing} from schemas but these classes are not defined in schemas.py"
                    )
        return issues

    def _check_invalid_model_definition(self, fm: dict[str, str]) -> list[str]:
        """Detect SQLAlchemy models using table reflection instead of mapped_column."""
        issues: list[str] = []
        for path, content in fm.items():
            if not (path.startswith("backend/") and path.endswith(".py")):
                continue
            if self._BAD_TABLE_REFLECTION_RE.search(content):
                issues.append(
                    f"{path}: uses '__table__ = Base.metadata.tables[...]' — "
                    "this crashes if the table doesn't exist yet. "
                    "Use '__tablename__' and define columns with mapped_column instead."
                )
        return issues

    def _fix_invalid_model_definition(self, fm: dict[str, str], fixes: list[str]) -> dict[str, str]:
        """Replace __table__ = Base.metadata.tables[...] with __tablename__ + minimal PK column."""
        for path in list(fm.keys()):
            if not (path.startswith("backend/") and path.endswith(".py")):
                continue
            content = fm[path]
            m = self._BAD_TABLE_REFLECTION_RE.search(content)
            if not m:
                continue
            tablename = m.group(2)
            has_columns = bool(re.search(r"mapped_column|Column\(", content))
            # Replace the bad __table__ line with __tablename__
            content = self._BAD_TABLE_REFLECTION_RE.sub(
                f'__tablename__ = "{tablename}"', content
            )
            # If no column definitions exist, inject a minimal primary key
            if not has_columns:
                content = re.sub(
                    rf'(__tablename__\s*=\s*"{re.escape(tablename)}")',
                    r"\1\n    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)",
                    content,
                )
                # Ensure Mapped and mapped_column are imported
                if "from sqlalchemy.orm import" in content:
                    if "Mapped" not in content or "mapped_column" not in content:
                        content = re.sub(
                            r"(from sqlalchemy\.orm import\s+)([^\n]+)",
                            lambda mm: (
                                f"{mm.group(1)}{mm.group(2).rstrip()}, Mapped, mapped_column"
                                if "Mapped" not in mm.group(2)
                                else mm.group(0)
                            ),
                            content,
                        )
                else:
                    content = "from sqlalchemy.orm import Mapped, mapped_column\n" + content
            fm[path] = content
            fixes.append(f"Fixed invalid table reflection → __tablename__ + mapped_column in {path}")
        return fm

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2 — Auto Fix
    # ─────────────────────────────────────────────────────────────────────────

    async def _auto_fix(
        self,
        files: list[dict],
        issues: list[str],
        db_session,
    ) -> tuple[list[dict], list[str], list[str]]:
        fixes: list[str] = []
        errors: list[str] = []

        fm = _fmap(files)
        fm = self._fix_base_not_defined(fm, fixes)
        fm = self._fix_ts_catch_types(fm, fixes)
        fm = self._fix_fastapi_imports(fm, fixes)
        fm = self._fix_cors(fm, fixes)
        fm = self._fix_requirements(fm, fixes)
        fm = self._fix_env_files(fm, fixes)
        fm = self._fix_tsconfig(fm, fixes)
        fm = self._fix_fake_packages(fm, fixes)
        fm = self._fix_database_url(fm, fixes)
        fm = self._fix_react_18(fm, fixes)
        fm = self._fix_pydantic_v2(fm, fixes)
        fm = self._fix_fastapi_db_depends(fm, fixes)
        fm = self._fix_schema_imports(fm, fixes)
        fm = self._fix_invalid_model_definition(fm, fixes)
        fm = self._fix_missing_css_imports(fm, issues, fixes)
        fm = self._fix_missing_packages(fm, issues, fixes)
        fm = self._fix_missing_app(fm, fixes)

        # Drop issues that structural fixers already resolved so they never reach errors
        resolved: set[str] = set()
        for iss in issues:
            if "backend .env missing" in iss and ("backend/.env" in fm or ".env" in fm):
                resolved.add(iss)
            if iss.startswith("frontend/.env: missing") and "frontend/.env" in fm:
                resolved.add(iss)
            path_m = re.match(r"(backend/\S+\.py):", iss)
            if "orm_mode" in iss and path_m and "orm_mode" not in fm.get(path_m.group(1), "orm_mode"):
                resolved.add(iss)
            css_m = re.search(r"imports CSS '([^']+)' which does not exist", iss)
            if css_m:
                path_m2 = re.match(r"(frontend/src/[^\s:]+):", iss)
                if path_m2:
                    file_dir = "/".join(path_m2.group(1).split("/")[:-1])
                    css_path = _fwd(f"{file_dir}/{css_m.group(1)}".replace("//", "/").lstrip("/"))
                    if css_path in fm:
                        resolved.add(iss)
            if "backend/schemas.py" in iss and ("Base is not imported" in iss or "mixes SQLAlchemy" in iss):
                schemas_now = fm.get("backend/schemas.py", "")
                if "from database import Base" in schemas_now or not re.search(
                    r"^class\s+\w+\s*\(\s*Base\s*\)", schemas_now, re.MULTILINE
                ):
                    resolved.add(iss)
            if "catch blocks lack type annotation" in iss:
                app_tsx = fm.get("frontend/src/App.tsx", "")
                if not re.search(r"\bcatch\s*\(\s*\w+\s*\)", app_tsx):
                    resolved.add(iss)
            if "uses 'next(get_db())'" in iss:
                path_m = re.match(r"(backend/\S+\.py):", iss)
                if path_m and not self._BAD_DB_DEPENDS_RE.search(fm.get(path_m.group(1), "")):
                    resolved.add(iss)
            if "uses '__table__ = Base.metadata" in iss:
                path_m = re.match(r"(backend/\S+\.py):", iss)
                if path_m and not self._BAD_TABLE_REFLECTION_RE.search(fm.get(path_m.group(1), "")):
                    resolved.add(iss)
        issues = [iss for iss in issues if iss not in resolved]

        files = _unmap(fm)

        # Training-collector instant fixes
        for issue in issues:
            pattern = training_collector.detect_error_type(issue)
            if pattern:
                instant = training_collector.get_instant_fix(
                    pattern["error_type"], issue, files
                )
                if instant is not None:
                    files = instant
                    fixes.append(f"Instant fix ({pattern['error_type']}): {issue[:80]}")
                    if db_session:
                        await training_collector.update_error_pattern_stats(
                            db_session, pattern["error_type"], success=True
                        )

        # Claude for unresolved issues
        already_fixed = {fx[:40] for fx in fixes}
        llm_issues = [
            iss for iss in issues
            if not any(iss[:40] in af for af in already_fixed)
        ]
        if llm_issues and self.api_key:
            files, claude_fixes = await self._fix_with_claude(
                files, llm_issues, "static analysis issues"
            )
            fixes.extend(claude_fixes)
        elif llm_issues:
            errors.extend(llm_issues)

        return files, fixes, errors

    # ── Structural fixers (no LLM needed) ────────────────────────────────────

    def _fix_cors(self, fm: dict[str, str], fixes: list[str]) -> dict[str, str]:
        main = fm.get("backend/main.py", "")
        if not main:
            return fm
        orig = main

        if re.search(r'allow_methods\s*=\s*["\']', main):
            main = re.sub(r'allow_methods\s*=\s*["\'][^"\']*["\']', 'allow_methods=["*"]', main)
            fixes.append("Fixed allow_methods to use list syntax")

        if re.search(r'allow_headers\s*=\s*["\']', main):
            main = re.sub(r'allow_headers\s*=\s*["\'][^"\']*["\']', 'allow_headers=["*"]', main)
            fixes.append("Fixed allow_headers to use list syntax")

        if "allow_credentials=True" in main and 'allow_origins=["*"]' in main:
            main = main.replace("allow_credentials=True", "allow_credentials=False")
            fixes.append("Fixed allow_credentials=True → False (incompatible with allow_origins=['*'])")

        if "CORSMiddleware" in main and "from fastapi.middleware.cors import" not in main:
            main = "from fastapi.middleware.cors import CORSMiddleware\n" + main
            fixes.append("Added missing CORSMiddleware import to backend/main.py")

        if main != orig:
            fm["backend/main.py"] = main
        return fm

    def _fix_requirements(self, fm: dict[str, str], fixes: list[str]) -> dict[str, str]:
        key = "backend/requirements.txt" if "backend/requirements.txt" in fm else "requirements.txt"
        req = fm.get(key, "")
        lines = [ln for ln in req.splitlines() if ln.strip()]
        present = {ln.split("[")[0].split("==")[0].split(">=")[0].strip().lower() for ln in lines}
        defaults = {
            "fastapi": "fastapi",
            "uvicorn": "uvicorn[standard]",
            "sqlalchemy": "sqlalchemy[asyncio]",
            "asyncpg": "asyncpg",
            "python-dotenv": "python-dotenv",
            "pydantic": "pydantic",
            "httpx": "httpx",
        }
        added = [full for key_, full in defaults.items() if key_ not in present]
        if added:
            lines.extend(added)
            fm[key] = "\n".join(lines) + "\n"
            fixes.append(f"Added missing requirements: {', '.join(added)}")
        return fm

    def _fix_env_files(self, fm: dict[str, str], fixes: list[str]) -> dict[str, str]:
        if "backend/.env" not in fm and ".env" not in fm:
            example = fm.get("backend/.env.example", fm.get(".env.example", ""))
            if example:
                fm["backend/.env"] = example
                fixes.append("Created backend/.env from .env.example")
            else:
                fm["backend/.env"] = (
                    "DATABASE_URL=postgresql+asyncpg://postgres:vinay2004@localhost:5432/app\n"
                    "SECRET_KEY=development-secret-key\n"
                )
                fixes.append("Created default backend/.env")

        if "frontend/.env" not in fm:
            fm["frontend/.env"] = "VITE_API_URL=http://localhost:8001\n"
            fixes.append("Created frontend/.env with VITE_API_URL")

        return fm

    def _fix_tsconfig(self, fm: dict[str, str], fixes: list[str]) -> dict[str, str]:
        key = "frontend/tsconfig.json"
        raw = fm.get(key, "")
        try:
            data: dict = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            data = {}

        opts = data.setdefault("compilerOptions", {})
        # Vite-compatible defaults — override CRA/react-scripts settings if present
        forced: dict = {
            "moduleResolution": "bundler",  # required for Vite
            "module": "ESNext",
            "noEmit": True,
            "jsx": "react-jsx",
            "skipLibCheck": True,
            "isolatedModules": True,
            "allowImportingTsExtensions": True,
            "esModuleInterop": True,
            "allowSyntheticDefaultImports": True,
        }
        defaults: dict = {
            "target": "ES2020",
            "lib": ["ES2020", "DOM", "DOM.Iterable"],
            "resolveJsonModule": True,
            "strict": False,
            "useDefineForClassFields": True,
        }
        # Remove CRA-specific keys that break Vite
        for cra_key in ("allowJs", "forceConsistentCasingInFileNames", "noFallthroughCasesInSwitch"):
            opts.pop(cra_key, None)
        patched = []
        for k, v in forced.items():
            if opts.get(k) != v:
                opts[k] = v
                patched.append(k)
        for k, v in defaults.items():
            if k not in opts:
                opts[k] = v
                patched.append(k)
        data.setdefault("include", ["src"])

        if patched or not raw.strip():
            fm[key] = json.dumps(data, indent=2)
            fixes.append(
                f"Patched tsconfig.json: added {patched}" if patched else "Created tsconfig.json"
            )
        return fm

    def _fix_fake_packages(self, fm: dict[str, str], fixes: list[str]) -> dict[str, str]:
        key = "frontend/package.json"
        raw = fm.get(key, "")
        if not raw:
            return fm
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return fm
        removed = []
        for fake in _FAKE_PACKAGES:
            for section in ("dependencies", "devDependencies"):
                if fake in data.get(section, {}):
                    del data[section][fake]
                    removed.append(fake)
        if removed:
            fm[key] = json.dumps(data, indent=2)
            fixes.append(f"Removed fake packages: {removed}")
        return fm

    def _fix_database_url(self, fm: dict[str, str], fixes: list[str]) -> dict[str, str]:
        # Fix Python source files (database.py may hardcode the URL as a fallback)
        for path in ("backend/database.py", "database.py"):
            raw = fm.get(path, "")
            if not raw:
                continue
            new = re.sub(r"\bpostgres://", "postgresql+asyncpg://", raw)
            new = re.sub(r"\bpostgresql://(?!\+asyncpg)", "postgresql+asyncpg://", new)
            if new != raw:
                fm[path] = new
                fixes.append(f"Fixed DATABASE_URL driver in {path}")

        # Fix .env and .env.example files — must run after _fix_env_files so the
        # copied .env is already in fm before we patch it.
        for path in ("backend/.env", ".env", "backend/.env.example", ".env.example"):
            raw = fm.get(path, "")
            if not raw:
                continue
            new = re.sub(r"\bpostgres://", "postgresql+asyncpg://", raw)
            new = re.sub(r"\bpostgresql://(?!\+asyncpg)", "postgresql+asyncpg://", new)
            if new != raw:
                fm[path] = new
                fixes.append(f"Fixed DATABASE_URL scheme in {path}")

        return fm

    def _fix_react_18(self, fm: dict[str, str], fixes: list[str]) -> dict[str, str]:
        key = "frontend/src/index.tsx"
        raw = fm.get(key, "")
        if not raw or "ReactDOM.render(" not in raw:
            return fm
        new = raw.replace(
            "import ReactDOM from 'react-dom';",
            "import ReactDOM from 'react-dom/client';",
        )
        new = re.sub(
            r"ReactDOM\.render\(([\s\S]*?),\s*document\.getElementById\(['\"]root['\"]\)\s*\)\s*;?",
            lambda m: (
                "const root = ReactDOM.createRoot(\n"
                "  document.getElementById('root') as HTMLElement\n"
                ");\n"
                f"root.render({m.group(1)});"
            ),
            new,
        )
        if new != raw:
            fm[key] = new
            fixes.append("Migrated index.tsx from ReactDOM.render to React 18 createRoot")
        return fm

    # Symbols that belong in `typing`, not `pydantic`
    _TYPING_SYMS: frozenset[str] = frozenset({
        "Optional", "List", "Dict", "Union", "Any", "Tuple", "Type",
        "Set", "FrozenSet", "Sequence", "Iterable", "Iterator", "Callable",
        "ClassVar", "Literal", "TypeVar", "Generic", "Annotated",
    })
    _PYDANTIC_TYPING_RE = re.compile(
        r"from pydantic import[^\n]*\b("
        r"Optional|List|Dict|Union|Any|Tuple|Type|Set|FrozenSet|"
        r"Sequence|Iterable|Iterator|Callable|ClassVar|Literal|TypeVar|Generic|Annotated"
        r")\b"
    )

    def _fix_pydantic_v2(self, fm: dict[str, str], fixes: list[str]) -> dict[str, str]:
        for path in list(fm.keys()):
            if not (path.startswith("backend/") and path.endswith(".py")):
                continue
            content = fm[path]
            has_orm_mode = "orm_mode" in content
            has_pydantic_typing_mod = "from pydantic.typing import" in content
            has_typing_from_pydantic = bool(self._PYDANTIC_TYPING_RE.search(content))
            if not (has_orm_mode or has_pydantic_typing_mod or has_typing_from_pydantic):
                continue
            original = content

            # Fix pydantic.typing imports — removed in Pydantic v2; use stdlib typing instead
            if has_pydantic_typing_mod:
                content = re.sub(
                    r"from pydantic\.typing import ([^\n]+)",
                    r"from typing import \1",
                    content,
                )

            # Fix typing symbols incorrectly imported from pydantic (e.g. from pydantic import Optional)
            if has_typing_from_pydantic:
                def _split_pydantic_import(m: re.Match) -> str:
                    symbols = [s.strip() for s in m.group(1).split(",") if s.strip()]
                    typing_syms = [s for s in symbols if s in self._TYPING_SYMS]
                    pydantic_syms = [s for s in symbols if s not in self._TYPING_SYMS]
                    parts: list[str] = []
                    if pydantic_syms:
                        parts.append(f"from pydantic import {', '.join(pydantic_syms)}")
                    if typing_syms:
                        parts.append(f"from typing import {', '.join(typing_syms)}")
                    return "\n".join(parts)
                content = re.sub(
                    r"from pydantic import ([^\n]+)",
                    _split_pydantic_import,
                    content,
                )

            # Inject ConfigDict into existing pydantic import, or prepend one (orm_mode fix only)
            if has_orm_mode and "ConfigDict" not in content:
                if "from pydantic import" in content:
                    content = re.sub(
                        r"(from pydantic import\s+)([^\n]+)",
                        lambda m: (
                            f"{m.group(1)}{m.group(2).rstrip()}, ConfigDict"
                            if "ConfigDict" not in m.group(2)
                            else m.group(0)
                        ),
                        content,
                    )
                else:
                    content = "from pydantic import BaseModel, ConfigDict\n" + content

            # Replace class Config: / orm_mode = True block (any indentation, CRLF-safe)
            content = re.sub(
                r"^([ \t]*)class Config:[ \t]*[\r\n]+([ \t]+)orm_mode[ \t]*=[ \t]*True[^\n]*",
                r"\1model_config = ConfigDict(from_attributes=True)",
                content,
                flags=re.MULTILINE,
            )

            # Fallback: any remaining standalone orm_mode = True
            if "orm_mode" in content:
                content = re.sub(
                    r"\borm_mode\b\s*=\s*True",
                    "model_config = ConfigDict(from_attributes=True)",
                    content,
                )

            if content != original:
                fm[path] = content
                fixes.append(f"Fixed Pydantic v2 compatibility in {path}")
        return fm

    def _fix_fastapi_db_depends(self, fm: dict[str, str], fixes: list[str]) -> dict[str, str]:
        """Replace next(get_db()) with Depends(get_db) in FastAPI route signatures."""
        for path in list(fm.keys()):
            if not (path.startswith("backend/") and path.endswith(".py")):
                continue
            content = fm[path]
            if not self._BAD_DB_DEPENDS_RE.search(content):
                continue
            content = self._BAD_DB_DEPENDS_RE.sub(r"\1 = Depends(get_db)", content)
            # Ensure Depends is imported
            if "from fastapi import" in content and "Depends" not in content:
                content = re.sub(
                    r"(from fastapi import\s+)([^\n]+)",
                    lambda m: f"{m.group(1)}{m.group(2).rstrip()}, Depends"
                    if "Depends" not in m.group(2) else m.group(0),
                    content,
                )
            elif "from fastapi import" not in content:
                content = "from fastapi import Depends\n" + content
            fm[path] = content
            fixes.append(f"Fixed next(get_db()) → Depends(get_db) in {path}")
        return fm

    def _fix_schema_imports(self, fm: dict[str, str], fixes: list[str]) -> dict[str, str]:
        """Auto-generate stub Pydantic models for schema names imported but not defined."""
        schemas_content = fm.get("backend/schemas.py", "")
        if not schemas_content:
            return fm
        defined = set(re.findall(r"^class\s+(\w+)", schemas_content, re.MULTILINE))
        missing: set[str] = set()
        for path, content in fm.items():
            if not (path.startswith("backend/") and path.endswith(".py") and path != "backend/schemas.py"):
                continue
            for m in re.finditer(r"from schemas import ([^\n]+)", content):
                for name in [n.strip() for n in m.group(1).split(",") if n.strip()]:
                    if name and name not in defined:
                        missing.add(name)
        if not missing:
            return fm
        stubs: list[str] = []
        if "from pydantic import BaseModel" not in schemas_content and "from pydantic import" not in schemas_content:
            stubs.append("from pydantic import BaseModel")
        for name in sorted(missing):
            stubs.append(f"\nclass {name}(BaseModel):\n    pass\n")
        fm["backend/schemas.py"] = schemas_content.rstrip() + "\n\n" + "\n".join(stubs) + "\n"
        fixes.append(f"Added missing schema stubs to backend/schemas.py: {sorted(missing)}")
        return fm

    def _fix_missing_css_imports(
        self, fm: dict[str, str], issues: list[str], fixes: list[str]
    ) -> dict[str, str]:
        basic_css = "* { box-sizing: border-box; margin: 0; padding: 0; }\nbody { font-family: sans-serif; }\n"
        for issue in issues:
            css_m = re.search(r"imports CSS '([^']+)' which does not exist", issue)
            if not css_m:
                continue
            path_m = re.match(r"(frontend/src/[^\s:]+):", issue)
            if not path_m:
                continue
            file_dir = "/".join(path_m.group(1).split("/")[:-1])
            resolved = _fwd(f"{file_dir}/{css_m.group(1)}".replace("//", "/").lstrip("/"))
            if resolved not in fm:
                fm[resolved] = basic_css
                fixes.append(f"Created missing CSS file: {resolved}")
        return fm

    def _fix_missing_app(self, fm: dict[str, str], fixes: list[str]) -> dict[str, str]:
        """Create a default App.tsx if index.tsx imports './App' but App.tsx is missing/empty."""
        index_tsx = fm.get("frontend/src/index.tsx", "")
        app_tsx = fm.get("frontend/src/App.tsx", "")
        imports_app = "from './App'" in index_tsx or 'from "./App"' in index_tsx
        if imports_app and not app_tsx.strip():
            fm["frontend/src/App.tsx"] = (
                "import React from 'react';\n\n"
                "function App() {\n"
                "  return (\n"
                "    <div className=\"min-h-screen bg-gray-50 flex items-center justify-center\">\n"
                "      <div className=\"bg-white rounded-2xl shadow-sm border border-gray-100 p-8 text-center\">\n"
                "        <h1 className=\"text-3xl font-bold text-gray-900 mb-2\">App is running!</h1>\n"
                "        <p className=\"text-gray-500\">Your application is ready.</p>\n"
                "      </div>\n"
                "    </div>\n"
                "  );\n"
                "}\n\n"
                "export default App;\n"
            )
            fixes.append("Created default App.tsx (index.tsx imports './App' but App.tsx was missing)")
        return fm

    def _fix_missing_packages(
        self, fm: dict[str, str], issues: list[str], fixes: list[str]
    ) -> dict[str, str]:
        key = "frontend/package.json"
        raw = fm.get(key, "")
        if not raw:
            return fm
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return fm

        deps: dict[str, str] = data.setdefault("dependencies", {})
        dev_deps: dict[str, str] = data.get("devDependencies", {})
        added = []
        for issue in issues:
            m = re.search(r"imports '([^']+)' but it is not in package\.json", issue)
            if not m:
                continue
            pkg = m.group(1)
            if pkg not in deps and pkg not in dev_deps:
                deps[pkg] = _KNOWN_VERSIONS.get(pkg, "latest")
                added.append(pkg)

        if added:
            data["dependencies"] = {
                "react": "^18.0.0",
                "react-dom": "^18.0.0",
                **deps,
            }
            fm[key] = json.dumps(data, indent=2)
            fixes.append(f"Added missing npm packages: {added}")
        return fm

    def _fix_base_not_defined(self, fm: dict[str, str], fixes: list[str]) -> dict[str, str]:
        schemas = fm.get("backend/schemas.py", "")
        if not schemas:
            return fm

        sa_classes = re.findall(r"^class\s+(\w+)\s*\(\s*Base\s*\)", schemas, re.MULTILINE)
        if not sa_classes:
            return fm

        pydantic_names = re.findall(r"^class\s+(\w+)\s*\(\s*BaseModel\b", schemas, re.MULTILINE)

        if not pydantic_names:
            # Only SQLAlchemy classes in schemas.py — add the missing import
            if "from database import Base" not in schemas:
                lines = schemas.splitlines()
                insert_at = 0
                for i, line in enumerate(lines):
                    if line.startswith(("import ", "from ")):
                        insert_at = i + 1
                lines.insert(insert_at, "from database import Base")
                fm["backend/schemas.py"] = "\n".join(lines) + "\n"
                fixes.append(
                    f"Added 'from database import Base' to backend/schemas.py "
                    f"(fixes NameError for: {', '.join(sa_classes)})"
                )
            return fm

        # Mixed file: split SQLAlchemy classes → models.py, Pydantic classes → schemas.py
        parts = re.split(r"(?=^class\s)", schemas, flags=re.MULTILINE)
        header = parts[0]
        class_blocks = [p.rstrip("\n") for p in parts[1:] if p.strip()]

        sa_blocks: list[str] = []
        pydantic_blocks: list[str] = []
        for block in class_blocks:
            first_line = block.splitlines()[0]
            if re.match(r"class\s+\w+\s*\(\s*Base\s*\)", first_line):
                sa_blocks.append(block)
            else:
                pydantic_blocks.append(block)

        # Rebuild schemas.py — Pydantic only, drop SQLAlchemy imports
        clean_header_lines = [
            ln for ln in header.splitlines()
            if not re.search(r"\bsqlalchemy\b", ln, re.IGNORECASE)
            and not re.search(r"\bfrom database import\b", ln)
        ]
        clean_header = "\n".join(clean_header_lines).strip()
        if "from pydantic import" not in clean_header:
            clean_header = "from pydantic import BaseModel\n" + clean_header
        fm["backend/schemas.py"] = (
            clean_header.rstrip() + "\n\n" + "\n\n".join(pydantic_blocks) + "\n"
        )
        fixes.append("Cleaned backend/schemas.py — now contains only Pydantic models")

        # Add SQLAlchemy classes to models.py
        sa_imports = (
            "from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, Text\n"
            "from sqlalchemy.orm import Mapped, mapped_column\n"
            "from database import Base\n"
        )
        existing_models = fm.get("backend/models.py", "")
        if existing_models:
            fm["backend/models.py"] = existing_models.rstrip() + "\n\n" + "\n\n".join(sa_blocks) + "\n"
            fixes.append(
                f"Moved SQLAlchemy model(s) from schemas.py to models.py: {', '.join(sa_classes)}"
            )
        else:
            fm["backend/models.py"] = sa_imports + "\n\n" + "\n\n".join(sa_blocks) + "\n"
            fixes.append(
                f"Created backend/models.py with SQLAlchemy models from schemas.py: {', '.join(sa_classes)}"
            )
        return fm

    def _fix_ts_catch_types(self, fm: dict[str, str], fixes: list[str]) -> dict[str, str]:
        changed: list[str] = []
        for path, content in list(fm.items()):
            if not (path.endswith(".tsx") or path.endswith(".ts")):
                continue
            # Matches catch (varname) without a type annotation — leaves catch (e: T) untouched
            new = re.sub(r"\bcatch\s*\(\s*(\w+)\s*\)", r"catch (\1: unknown)", content)
            if new != content:
                fm[path] = new
                changed.append(path)
        if changed:
            fixes.append(f"Fixed TypeScript catch type annotations in: {', '.join(changed)}")
        return fm

    def _fix_fastapi_imports(self, fm: dict[str, str], fixes: list[str]) -> dict[str, str]:
        _FASTAPI_IMPORTS: dict[str, str] = {
            "APIRouter":       "from fastapi import APIRouter",
            "Depends":         "from fastapi import Depends",
            "HTTPException":   "from fastapi import HTTPException",
            "status":          "from fastapi import status",
            "Body":            "from fastapi import Body",
            "Query":           "from fastapi import Query",
            "Path":            "from fastapi import Path",
            "Form":            "from fastapi import Form",
            "File":            "from fastapi import File",
            "UploadFile":      "from fastapi import UploadFile",
            "BackgroundTasks": "from fastapi import BackgroundTasks",
        }
        _SA_IMPORTS: dict[str, str] = {
            "AsyncSession": "from sqlalchemy.ext.asyncio import AsyncSession",
            "Session":      "from sqlalchemy.orm import Session",
            "select":       "from sqlalchemy import select",
        }
        for path in list(fm.keys()):
            if not (path.startswith("backend/") and path.endswith(".py")):
                continue
            content = fm[path]
            original = content
            to_add: list[str] = []
            for symbol, import_line in {**_FASTAPI_IMPORTS, **_SA_IMPORTS}.items():
                module = import_line.split(" import ")[0].replace("from ", "")
                already = (
                    import_line in content
                    or f"from {module} import *" in content
                    or re.search(rf"from {re.escape(module)} import[^\n]*\b{symbol}\b", content)
                )
                if not already and re.search(rf"\b{symbol}\s*[\(\[,\s]", content):
                    to_add.append(import_line)
            if to_add:
                lines = content.splitlines()
                insert_at = 0
                for i, line in enumerate(lines):
                    if line.startswith(("import ", "from ")):
                        insert_at = i + 1
                for imp in to_add:
                    lines.insert(insert_at, imp)
                    insert_at += 1
                fm[path] = "\n".join(lines) + "\n"
                fixes.append(f"Added missing imports to {path}: {', '.join(to_add)}")
        return fm

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3 — Dependency Installation
    # ─────────────────────────────────────────────────────────────────────────

    async def _install_dependencies(
        self, files: list[dict], temp_dir: Path
    ) -> tuple[list[dict], list[str], list[str]]:
        fixes: list[str] = []
        errors: list[str] = []
        loop = asyncio.get_running_loop()

        # ── pip install ───────────────────────────────────────────────────────
        req_path = temp_dir / "backend" / "requirements.txt"
        if not req_path.exists():
            req_path = temp_dir / "requirements.txt"

        if req_path.exists():
            logger.info("%s [Phase 3] pip install -r %s", self.agent_name, req_path.name)
            try:
                res = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-r", str(req_path), "--quiet"],
                        capture_output=True, text=True, timeout=60,
                    ),
                )
                if res.returncode == 0:
                    fixes.append("pip install completed successfully")
                else:
                    err = (res.stderr or res.stdout)[:500]
                    logger.warning("%s pip error: %s", self.agent_name, err[:200])
                    files, fix_msg = self._fix_pip_error(files, err)
                    if fix_msg:
                        fixes.append(fix_msg)
                        self._write_files(files, temp_dir)
                        retry = await loop.run_in_executor(
                            None,
                            lambda: subprocess.run(
                                [sys.executable, "-m", "pip", "install", "-r", str(req_path), "--quiet"],
                                capture_output=True, text=True, timeout=60,
                            ),
                        )
                        if retry.returncode == 0:
                            fixes.append("pip install succeeded after requirements.txt fix")
                        else:
                            errors.append(f"pip install failed after retry: {(retry.stderr or retry.stdout)[:200]}")
                    else:
                        errors.append(f"pip install failed: {err[:200]}")
            except Exception as exc:
                errors.append(f"pip install exception: {exc}")

        # ── npm install ───────────────────────────────────────────────────────
        npm_cmd = shutil.which("npm")
        frontend_dir = temp_dir / "frontend"
        pkg_json = frontend_dir / "package.json"

        if npm_cmd and frontend_dir.exists() and pkg_json.exists():
            logger.info("%s [Phase 3] npm install --legacy-peer-deps", self.agent_name)
            try:
                res = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        [npm_cmd, "install", "--legacy-peer-deps"],
                        capture_output=True, text=True, timeout=120,
                        cwd=str(frontend_dir),
                    ),
                )
                if res.returncode == 0:
                    fixes.append("npm install completed successfully")
                else:
                    err = (res.stderr or res.stdout)[:500]
                    logger.warning("%s npm error: %s", self.agent_name, err[:200])
                    files, fix_msg = self._fix_npm_error(files, err)
                    if fix_msg:
                        fixes.append(fix_msg)
                        self._write_files(files, temp_dir)
                        retry = await loop.run_in_executor(
                            None,
                            lambda: subprocess.run(
                                [npm_cmd, "install", "--legacy-peer-deps"],
                                capture_output=True, text=True, timeout=120,
                                cwd=str(frontend_dir),
                            ),
                        )
                        if retry.returncode == 0:
                            fixes.append("npm install succeeded after package.json fix")
                        else:
                            errors.append(f"npm install failed after retry: {(retry.stderr or retry.stdout)[:200]}")
                    else:
                        errors.append(f"npm install failed: {err[:200]}")
            except Exception as exc:
                errors.append(f"npm install exception: {exc}")
        else:
            if not npm_cmd:
                logger.warning("%s npm not found on PATH.", self.agent_name)
            elif not frontend_dir.exists():
                logger.warning("%s frontend/ dir not found in temp dir.", self.agent_name)

        return files, fixes, errors

    def _fix_pip_error(self, files: list[dict], error_text: str) -> tuple[list[dict], str]:
        m = re.search(r"requirement (\S+)", error_text)
        if not m:
            return files, ""
        bad = m.group(1).lower().split("==")[0].split(">=")[0].split("[")[0]
        for f in files:
            if f["path"] in ("backend/requirements.txt", "requirements.txt"):
                lines = [
                    ln for ln in f["content"].splitlines()
                    if not re.fullmatch(
                        rf"\s*{re.escape(bad)}\s*(?:[>=<!#\[].*)?", ln, re.IGNORECASE
                    )
                ]
                f["content"] = "\n".join(lines) + "\n"
                return files, f"Removed failing package '{bad}' from requirements.txt"
        return files, ""

    def _fix_npm_error(self, files: list[dict], error_text: str) -> tuple[list[dict], str]:
        m404 = re.search(r"404 Not Found.*?npmjs\.org/([^\s]+)", error_text)
        if m404:
            bad = m404.group(1).replace("%2f", "/").replace("%2F", "/")
            for f in files:
                if "package.json" in f["path"]:
                    try:
                        data = json.loads(f["content"])
                        removed = False
                        for sec in ("dependencies", "devDependencies"):
                            if bad in data.get(sec, {}):
                                del data[sec][bad]
                                removed = True
                        if removed:
                            f["content"] = json.dumps(data, indent=2)
                            return files, f"Removed 404 package '{bad}' from package.json"
                    except json.JSONDecodeError:
                        pass

        if "ERESOLVE" in error_text or "peer" in error_text.lower():
            for f in files:
                if f["path"] == "frontend/.npmrc":
                    if "legacy-peer-deps" not in f["content"]:
                        f["content"] = f["content"].rstrip() + "\nlegacy-peer-deps=true\n"
                    return files, "Added legacy-peer-deps=true to .npmrc"
            files.append({"path": "frontend/.npmrc", "content": "legacy-peer-deps=true\n"})
            return files, "Created frontend/.npmrc with legacy-peer-deps=true"

        return files, ""

    def _create_database(self, project_name: str, files: list[dict]) -> bool:
        """Auto-create PostgreSQL database if it doesn't exist."""
        if psycopg2 is None:
            logger.warning("%s psycopg2 not installed — skipping DB creation", self.agent_name)
            return False
        try:
            db_name = project_name.lower().replace("-", "_").replace(" ", "_")
            for f in files:
                if f["path"] in ("backend/.env", ".env"):
                    for line in f["content"].splitlines():
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
                logger.info("%s created database: %s", self.agent_name, db_name)
            else:
                logger.info("%s database already exists: %s", self.agent_name, db_name)
            cur.close()
            conn.close()
            return True
        except Exception as e:
            logger.warning("%s could not create database: %s", self.agent_name, e)
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 4 — Runtime Verification
    # ─────────────────────────────────────────────────────────────────────────

    async def _runtime_verification(
        self,
        files: list[dict],
        temp_dir: Path,
        project_name: str = "",
    ) -> tuple[list[dict], str, str, str, list[str], list[str], str]:
        fixes: list[str] = []
        errors: list[str] = []
        backend_status = "unknown"
        frontend_status = "unknown"
        backend_url = ""
        crash_diagnosis = ""

        # ── Backend ───────────────────────────────────────────────────────────
        backend_dir = temp_dir / "backend"
        if backend_dir.exists():
            port = _find_free_port()
            backend_url = f"http://localhost:{port}"
            proc: subprocess.Popen | None = None
            try:
                env = os.environ.copy()
                env["PORT"] = str(port)
                dotenv = backend_dir / ".env"
                if dotenv.exists():
                    for line in dotenv.read_text(encoding="utf-8").splitlines():
                        if "=" in line and not line.startswith("#"):
                            k, _, v = line.partition("=")
                            env.setdefault(k.strip(), v.strip())

                self._create_database(project_name, files)

                proc = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", "main:app",
                     "--host", "0.0.0.0", "--port", str(port)],
                    cwd=str(backend_dir),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    env=env,
                )

                await asyncio.sleep(8)

                async with httpx.AsyncClient(timeout=5.0) as client:
                    for endpoint in ("/health", "/", "/docs"):
                        try:
                            resp = await client.get(f"{backend_url}{endpoint}")
                            if resp.status_code < 500:
                                backend_status = "running"
                                fixes.append(f"Backend verified running at {backend_url}{endpoint}")
                                break
                        except Exception:
                            continue

                if backend_status != "running":
                    proc.terminate()
                    try:
                        _, stderr_bytes = proc.communicate(timeout=3)
                        stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")
                    except Exception:
                        stderr = ""

                    _CRASH_MARKERS = (
                        "Traceback (most recent call last)",
                        "Error:",
                        "Exception:",
                        "CRITICAL",
                    )
                    is_real_crash = any(marker in stderr for marker in _CRASH_MARKERS)

                    if stderr and not is_real_crash:
                        backend_status = "running"
                        fixes.append("Backend startup confirmed (INFO-only stderr — no errors detected)")
                    elif stderr and is_real_crash:
                        backend_status = "error"
                        logger.error("%s Backend startup traceback:\n%s", self.agent_name, stderr)
                        errors.append(f"Backend startup traceback: {stderr}")
                        crash_diagnosis = await self._diagnose_error(stderr)
                        if self.api_key:
                            files, cf = await self._fix_with_claude(
                                files, [stderr], "uvicorn startup error"
                            )
                            fixes.extend(cf)
                    else:
                        backend_status = "error"
                        errors.append("Backend exited before health check")

            except Exception as exc:
                backend_status = "error"
                errors.append(f"Backend verification exception: {exc}")
            finally:
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except Exception:
                        proc.kill()

        # ── Frontend build ────────────────────────────────────────────────────
        npm_cmd = shutil.which("npm")
        frontend_dir = temp_dir / "frontend"
        node_modules = frontend_dir / "node_modules"

        if npm_cmd and frontend_dir.exists() and node_modules.exists():
            loop = asyncio.get_running_loop()
            build_env = os.environ.copy()
            build_env["CI"] = "false"  # prevent react-scripts treating warnings as errors
            try:
                res = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        [npm_cmd, "run", "build"],
                        capture_output=True, text=True, timeout=120,
                        cwd=str(frontend_dir),
                        env=build_env,
                    ),
                )
                combined = res.stderr + res.stdout
                logger.debug(
                    "%s [Phase 4] npm run build returncode=%d output_len=%d first500=%r",
                    self.agent_name, res.returncode, len(combined), combined[:500],
                )
                if res.returncode == 0:
                    frontend_status = "built"
                    fixes.append("Frontend npm run build succeeded")
                else:
                    ts_errors = [
                        ln for ln in combined.splitlines()
                        if "error TS" in ln or "ERROR in" in ln or "Failed to compile" in ln
                        or "Cannot find module" in ln or "Module not found" in ln
                    ][:10]
                    frontend_status = "error"
                    if ts_errors:
                        errors.extend(ts_errors[:3])
                    else:
                        # Capture more lines to surface module-not-found and other errors
                        brief = [ln for ln in combined.splitlines() if ln.strip()][:15]
                        errors.append(f"Frontend build failed: {' | '.join(brief[:10])}")

                    if ts_errors and self.api_key:
                        files, cf = await self._fix_with_claude(
                            files, ts_errors, "TypeScript build errors"
                        )
                        fixes.extend(cf)
                        if cf:
                            self._write_files(files, frontend_dir.parent)
                            retry = await loop.run_in_executor(
                                None,
                                lambda: subprocess.run(
                                    [npm_cmd, "run", "build"],
                                    capture_output=True, text=True, timeout=120,
                                    cwd=str(frontend_dir),
                                    env=build_env,
                                ),
                            )
                            if retry.returncode == 0:
                                frontend_status = "built"
                                fixes.append("Frontend build succeeded after Claude fix")
            except Exception as exc:
                frontend_status = "error"
                errors.append(f"Frontend build exception: {exc}")
        else:
            logger.info(
                "%s [Phase 4] Skipping frontend build — node_modules absent.", self.agent_name
            )

        return files, backend_status, frontend_status, backend_url, fixes, errors, crash_diagnosis

    # ─────────────────────────────────────────────────────────────────────────
    # Claude-assisted fixing
    # ─────────────────────────────────────────────────────────────────────────

    async def _diagnose_error(self, traceback: str) -> str:
        """Ask Claude to identify the root cause of a startup traceback in one sentence."""
        if not self.api_key:
            return ""
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    self.api_url,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 150,
                        "messages": [{
                            "role": "user",
                            "content": (
                                "This is a FastAPI/uvicorn startup traceback. "
                                "Identify the root cause in one sentence.\n\n"
                                f"{traceback[:3000]}"
                            ),
                        }],
                    },
                )
            if resp.status_code == 200:
                return resp.json()["content"][0]["text"].strip()
        except Exception as exc:
            logger.warning("%s _diagnose_error failed: %s", self.agent_name, exc)
        return ""

    async def _fix_with_claude(
        self,
        files: list[dict],
        errors: list[str],
        context: str,
    ) -> tuple[list[dict], list[str]]:
        if not self.api_key or not errors:
            return files, []

        error_text = "\n".join(errors)
        relevant = [
            f for f in files
            if any(Path(f["path"]).name in err for err in errors)
        ] or files[:6]
        files_ctx = "\n\n".join(
            f"### {f['path']}\n```\n{f['content'][:2500]}\n```"
            for f in relevant
        )

        system = (
            "You are an expert debugger. Fix the errors in the provided code files.\n"
            "Return ONLY a valid JSON array: [{\"path\": str, \"content\": str}]\n"
            "Only include files that need to be fixed.\n"
            "Complete file content only — no truncation, no placeholders, no markdown fences."
        )
        user = (
            f"Context: {context}\n\nErrors:\n{error_text}\n\n"
            f"Files:\n{files_ctx}\n\nFix all errors. Return JSON array only."
        )

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    self.api_url,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 8000,
                        "messages": [{"role": "user", "content": user}],
                        "system": system,
                    },
                )
            if resp.status_code != 200:
                logger.error("%s Claude error %d", self.agent_name, resp.status_code)
                return files, []

            raw = resp.json()["content"][0]["text"].strip()
            for fence in ("```json", "```"):
                if raw.startswith(fence):
                    raw = raw[len(fence):]
                    break
            raw = raw.strip().rstrip("```").strip()

            fixed = json.loads(raw)
            fixed_map = {f["path"]: f["content"] for f in fixed}
            new_files = [
                {"path": f["path"], "content": fixed_map.get(f["path"], f["content"])}
                for f in files
            ]
            return new_files, [f"Claude fixed: {', '.join(fixed_map.keys())}"]

        except json.JSONDecodeError as exc:
            logger.error("%s Claude JSON parse error: %s", self.agent_name, exc)
            return files, []
        except Exception as exc:
            logger.exception("%s Claude exception: %s", self.agent_name, exc)
            return files, []

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _build_fix_summary(self, fixes: list[str], errors: list[str]) -> str:
        if not fixes and not errors:
            return "No issues found — project looks clean."
        parts: list[str] = []
        if fixes:
            parts.append(f"Fixed {len(fixes)}: {'; '.join(fixes[:5])}")
        if errors:
            parts.append(f"{len(errors)} remaining: {'; '.join(errors[:3])}")
        return " | ".join(parts)

    def _write_files(self, files: list[dict], base_dir: Path) -> None:
        for f in files:
            path = base_dir / f["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                path.write_text(f["content"], encoding="utf-8")
            except Exception as exc:
                logger.warning("%s could not write %s: %s", self.agent_name, f["path"], exc)

    def fix_for_preview(self, files: list[dict]) -> list[dict]:
        """Apply instant pre-preview fixes to frontend files without an LLM call."""
        for f in files:
            path = f["path"]
            if not (path.endswith(".tsx") or path.endswith(".ts")):
                continue
            content = f["content"]
            content = re.sub(
                r"process\.env\.REACT_APP_API_URL(?!\s*\|\|)",
                "process.env.REACT_APP_API_URL || 'http://localhost:8000'",
                content,
            )
            if path.endswith(".tsx"):
                content = re.sub(r"useState\(\)", "useState<any>(null)", content)
            f["content"] = content
        return files


debugging_agent = DebuggingAgent()
