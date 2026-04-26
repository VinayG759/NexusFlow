"""ORM models for NexusFlow.

Three tables form the core persistence layer:

- ``sessions``     — one row per user problem submitted to the orchestrator.
- ``tasks``        — subtasks spawned from a session, each owned by one agent.
- ``task_results`` — the output (or error) produced when a task completes.

Import Base from this module if you need to access metadata, otherwise import
individual models directly::

    from src.database.models import Session, Task, TaskResult
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.connection import Base


# ── Session ───────────────────────────────────────────────────────────────────


class Session(Base):
    """Represents a single orchestration session initiated by a user.

    A session is created when the user submits a problem statement. The
    orchestrator decomposes it into one or more :class:`Task` rows linked
    via ``session_id``.

    Attributes:
        id: Auto-generated UUID primary key.
        problem_statement: The raw problem text supplied by the user.
        status: Lifecycle state — one of ``pending``, ``running``, ``done``, ``failed``.
        created_at: UTC timestamp set automatically on insert.
        updated_at: UTC timestamp refreshed automatically on every update.
        tasks: Back-populated list of :class:`Task` objects for this session.
    """

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    problem_statement: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="pending",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    tasks: Mapped[list["Task"]] = relationship(
        "Task",
        back_populates="session",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Session id={self.id} status={self.status!r}>"


# ── Task ──────────────────────────────────────────────────────────────────────


class Task(Base):
    """A single unit of work delegated to a specialised agent.

    Tasks are created by the orchestrator during the orchestration phase and
    belong to exactly one :class:`Session`. A task may declare dependencies on
    other tasks via ``depends_on`` so the executor can respect ordering.

    Attributes:
        id: Auto-generated UUID primary key.
        session_id: Foreign key back to the owning :class:`Session`.
        description: Plain-language description of the work to be done.
        assigned_agent: Name of the agent responsible for executing this task.
        status: Lifecycle state — one of ``pending``, ``running``, ``done``, ``failed``.
        depends_on: JSON list of task UUIDs (as strings) that must complete
            before this task may start. Defaults to an empty list.
        created_at: UTC timestamp set automatically on insert.
        updated_at: UTC timestamp refreshed automatically on every update.
        session: The parent :class:`Session` instance.
        result: The associated :class:`TaskResult`, if the task has completed.
    """

    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    assigned_agent: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="pending",
    )
    depends_on: Mapped[list[Any]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    session: Mapped["Session"] = relationship("Session", back_populates="tasks")
    result: Mapped["TaskResult | None"] = relationship(
        "TaskResult",
        back_populates="task",
        cascade="all, delete-orphan",
        uselist=False,
    )

    def __repr__(self) -> str:
        return f"<Task id={self.id} agent={self.assigned_agent!r} status={self.status!r}>"


# ── TaskResult ────────────────────────────────────────────────────────────────


class TaskResult(Base):
    """The output produced by an agent when a :class:`Task` completes.

    A task has at most one result row. On success ``output`` holds the agent's
    structured response and ``error`` is ``None``; on failure ``error`` contains
    the exception message and ``output`` may be empty.

    Attributes:
        id: Auto-generated UUID primary key.
        task_id: Foreign key back to the owning :class:`Task`.
        output: JSON payload containing the agent's structured result.
        error: Human-readable error message if the task failed; ``None`` otherwise.
        created_at: UTC timestamp set automatically on insert.
        task: The parent :class:`Task` instance.
    """

    __tablename__ = "task_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    output: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    task: Mapped["Task"] = relationship("Task", back_populates="result")

    def __repr__(self) -> str:
        return f"<TaskResult id={self.id} task_id={self.task_id} error={self.error!r}>"


# ── Project ───────────────────────────────────────────────────────────────────


class Project(Base):
    """A full-stack project generated by NexusFlow and persisted to the database."""

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    problem_statement: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(50), default="ready")
    setup_instructions: Mapped[str] = mapped_column(Text, default="")
    tech_stack: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    files: Mapped[list["ProjectFile"]] = relationship(
        "ProjectFile",
        back_populates="project",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Project id={self.id} name={self.name!r} status={self.status!r}>"


# ── ProjectFile ───────────────────────────────────────────────────────────────


class ProjectFile(Base):
    """A single source file belonging to a generated project."""

    __tablename__ = "project_files"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    file_path: Mapped[str] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text)
    file_type: Mapped[str] = mapped_column(String(50), default="text")

    project: Mapped["Project"] = relationship("Project", back_populates="files")

    def __repr__(self) -> str:
        return f"<ProjectFile id={self.id} path={self.file_path!r}>"
