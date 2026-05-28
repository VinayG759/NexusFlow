"""Shared pytest fixtures for NexusFlow tests.

Uses an in-memory SQLite database (via aiosqlite) so tests never need a live
PostgreSQL server.  The FastAPI lifespan (which calls init_db and seeds data
from PostgreSQL) is patched out so only our test DB is used.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.connection import Base
from main import app, get_db, _build_jobs, _preview_jobs

_TEST_DB_URL = "sqlite+aiosqlite://"


@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine(
        _TEST_DB_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine):
    factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session


@pytest_asyncio.fixture
async def client(db_session):
    """HTTP test client with the test SQLite DB wired in and lifespan patched out."""

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with (
        patch("main.init_db", new_callable=AsyncMock),
        patch("main._background_startup", new_callable=AsyncMock),
        patch("main.api_connector.close", new_callable=AsyncMock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac

    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def clear_job_caches():
    """Wipe in-memory job dicts between every test so state doesn't bleed."""
    _build_jobs.clear()
    _preview_jobs.clear()
    yield
    _build_jobs.clear()
    _preview_jobs.clear()
