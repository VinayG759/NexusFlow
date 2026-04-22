"""Database connection and session management for NexusFlow.

Sets up an async SQLAlchemy engine backed by asyncpg, exposes a session
factory, a declarative Base for model definitions, and helpers for session
lifecycle management and schema initialisation.

Usage::

    # In model files — inherit from Base:
    from src.database.connection import Base

    class MyModel(Base):
        __tablename__ = "my_table"
        ...

    # In route / service code — get a transactional session:
    from src.database.connection import get_db_session

    async with get_db_session() as session:
        result = await session.execute(select(MyModel))

    # At application startup — create all tables:
    from src.database.connection import init_db

    await init_db()
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Engine ────────────────────────────────────────────────────────────────────
# asyncpg requires the scheme to be "postgresql+asyncpg://".
# settings.DATABASE_URL may use "postgresql://" or "postgresql+psycopg2://",
# so we normalise it here without mutating the original settings value.

_async_url: str = (
    settings.DATABASE_URL
    .replace("postgresql://", "postgresql+asyncpg://", 1)
    .replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
)

engine = create_async_engine(
    _async_url,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    echo=settings.DEBUG,  # logs every SQL statement when DEBUG=true
    future=True,
)

# ── Session factory ───────────────────────────────────────────────────────────

AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # keeps attributes accessible after commit
)

# ── Declarative base ──────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    """Shared declarative base — all NexusFlow ORM models must inherit from this."""


# ── Session context manager ───────────────────────────────────────────────────


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a transactional async database session.

    Automatically commits on success and rolls back on any exception.
    The session is always closed when the block exits.

    Yields:
        An :class:`AsyncSession` bound to the shared engine.

    Raises:
        Re-raises any exception thrown inside the ``async with`` block after
        rolling back the transaction.

    Example::

        async with get_db_session() as session:
            session.add(some_model_instance)
            # commit happens automatically on exit
    """
    session: AsyncSession = AsyncSessionFactory()
    try:
        yield session
        await session.commit()
        logger.debug("Database session committed successfully.")
    except Exception:
        await session.rollback()
        logger.exception("Database session rolled back due to an error.")
        raise
    finally:
        await session.close()
        logger.debug("Database session closed.")


# ── Table initialisation ──────────────────────────────────────────────────────


async def init_db() -> None:
    """Create all tables defined on Base at application startup.

    Safe to call on every startup — SQLAlchemy only creates tables that do not
    already exist (``checkfirst=True`` is the default for ``create_all``).

    Should be awaited once during the application lifespan, e.g. in the FastAPI
    ``lifespan`` startup handler or an equivalent entry point.

    Example::

        import asyncio
        from src.database.connection import init_db

        asyncio.run(init_db())
    """
    logger.info("Initialising database schema...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialised successfully.")
