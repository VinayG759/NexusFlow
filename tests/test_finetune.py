"""Unit tests for FineTuningPipeline.

Focused on validate_jsonl() and auto_loop() — the two methods most likely to
have edge-case bugs and the ones that don't need a live Groq connection.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.utils.finetune_pipeline import FineTuningPipeline


@pytest.fixture
def pipeline():
    return FineTuningPipeline()


@pytest.fixture
def valid_jsonl(tmp_path) -> Path:
    """Write a JSONL file with 3 fully-valid examples."""
    p = tmp_path / "train.jsonl"
    examples = [
        {
            "messages": [
                {"role": "system", "content": "You are NexusFlow."},
                {"role": "user", "content": "Build a todo app with FastAPI and React."},
                {"role": "assistant", "content": "Sure! Here is the full project…"},
            ]
        }
    ] * 3
    p.write_text("\n".join(json.dumps(e) for e in examples), encoding="utf-8")
    return p


@pytest.fixture
def empty_jsonl(tmp_path) -> Path:
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    return p


# ── validate_jsonl ────────────────────────────────────────────────────────────


def test_validate_valid_file(pipeline, valid_jsonl):
    result = pipeline.validate_jsonl(valid_jsonl)
    assert result["total"] == 3
    assert result["valid"] == 3
    assert result["invalid"] == 0
    assert result["errors"] == []


def test_validate_empty_file(pipeline, empty_jsonl):
    result = pipeline.validate_jsonl(empty_jsonl)
    assert result["total"] == 0
    assert result["valid"] == 0
    assert result["ready_for_training"] is False


def test_validate_missing_messages_key(pipeline, tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"not_messages": []}), encoding="utf-8")
    result = pipeline.validate_jsonl(p)
    assert result["invalid"] == 1
    assert any("missing 'messages'" in e for e in result["errors"])


def test_validate_too_few_messages(pipeline, tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"messages": [{"role": "user", "content": "hi there"}]}), encoding="utf-8")
    result = pipeline.validate_jsonl(p)
    assert result["invalid"] == 1
    assert any("at least 2 messages" in e for e in result["errors"])


def test_validate_missing_user_or_assistant(pipeline, tmp_path):
    p = tmp_path / "bad.jsonl"
    example = {
        "messages": [
            {"role": "system", "content": "You are an assistant."},
            {"role": "system", "content": "You are still an assistant."},
        ]
    }
    p.write_text(json.dumps(example), encoding="utf-8")
    result = pipeline.validate_jsonl(p)
    assert result["invalid"] == 1
    assert any("user and assistant" in e for e in result["errors"])


def test_validate_short_content(pipeline, tmp_path):
    p = tmp_path / "bad.jsonl"
    example = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
    }
    p.write_text(json.dumps(example), encoding="utf-8")
    result = pipeline.validate_jsonl(p)
    assert result["invalid"] == 1


def test_validate_invalid_json_line(pipeline, tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("this is not json\n", encoding="utf-8")
    result = pipeline.validate_jsonl(p)
    assert result["invalid"] == 1
    assert any("JSON error" in e for e in result["errors"])


def test_validate_mixed_valid_invalid(pipeline, tmp_path):
    p = tmp_path / "mixed.jsonl"
    good = json.dumps({
        "messages": [
            {"role": "user", "content": "Build me a todo app with FastAPI."},
            {"role": "assistant", "content": "Here is the complete project structure…"},
        ]
    })
    bad = json.dumps({"messages": []})
    p.write_text(f"{good}\n{bad}\n", encoding="utf-8")
    result = pipeline.validate_jsonl(p)
    assert result["total"] == 2
    assert result["valid"] == 1
    assert result["invalid"] == 1


def test_validate_ready_for_training_threshold(pipeline, tmp_path):
    """ready_for_training requires >= 10 valid examples and zero errors."""
    p = tmp_path / "small.jsonl"
    good = json.dumps({
        "messages": [
            {"role": "user", "content": "Build a todo app with FastAPI and React TypeScript."},
            {"role": "assistant", "content": "Here is the complete working project…"},
        ]
    })
    p.write_text("\n".join([good] * 5), encoding="utf-8")
    result = pipeline.validate_jsonl(p)
    # 5 valid examples is below the 10-example threshold for ready_for_training
    assert result["ready_for_training"] is False
    assert "Need" in result["recommendation"]


def test_validate_ready_for_training_at_ten(pipeline, tmp_path):
    """ready_for_training flips True at >= 10 valid examples with no errors."""
    p = tmp_path / "enough.jsonl"
    good = json.dumps({
        "messages": [
            {"role": "user", "content": "Build a full-stack todo app with FastAPI and React."},
            {"role": "assistant", "content": "Here is the complete project with all files included…"},
        ]
    })
    p.write_text("\n".join([good] * 15), encoding="utf-8")
    result = pipeline.validate_jsonl(p)
    assert result["ready_for_training"] is True
    # recommendation threshold is 200 — still shows progress toward that goal
    assert "Need" in result["recommendation"]


def test_validate_recommendation_at_200(pipeline, tmp_path):
    """recommendation says 'Ready to fine-tune!' only when 200+ examples are present."""
    p = tmp_path / "lots.jsonl"
    good = json.dumps({
        "messages": [
            {"role": "user", "content": "Build a full-stack todo app with FastAPI and React."},
            {"role": "assistant", "content": "Here is the complete project with all files included…"},
        ]
    })
    p.write_text("\n".join([good] * 200), encoding="utf-8")
    result = pipeline.validate_jsonl(p)
    assert result["ready_for_training"] is True
    assert result["recommendation"] == "Ready to fine-tune!"


# ── auto_loop ─────────────────────────────────────────────────────────────────


async def test_auto_loop_not_ready(pipeline, tmp_path):
    """auto_loop returns not_ready when the exported file has too few examples."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    mock_db = AsyncMock()
    with (
        patch.object(pipeline, "export_training_data", new_callable=AsyncMock, return_value=empty),
    ):
        result = await pipeline.auto_loop(mock_db)

    assert result["status"] == "not_ready"
    assert "exported_file" in result
    assert "validation" in result


async def test_auto_loop_submits_when_ready(pipeline, tmp_path):
    """auto_loop calls submit_to_groq when validation passes."""
    good = json.dumps({
        "messages": [
            {"role": "user", "content": "Build a full todo app with FastAPI and React TypeScript."},
            {"role": "assistant", "content": "Here is the complete project with all required files…"},
        ]
    })
    enough = tmp_path / "enough.jsonl"
    enough.write_text("\n".join([good] * 15), encoding="utf-8")

    mock_db = AsyncMock()
    mock_submission = {"status": "submitted", "job_id": "ft-abc123"}

    with (
        patch.object(pipeline, "export_training_data", new_callable=AsyncMock, return_value=enough),
        patch.object(pipeline, "submit_to_groq", new_callable=AsyncMock, return_value=mock_submission),
    ):
        result = await pipeline.auto_loop(mock_db)

    assert result["status"] == "submitted"
    assert result["job_id"] == "ft-abc123"
