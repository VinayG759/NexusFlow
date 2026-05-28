"""
NexusFlow Fine-tuning Pipeline
Exports training data and submits to Groq for fine-tuning.
"""

import json
import os
import httpx
import asyncio
from datetime import datetime
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from src.database.models import TrainingExample, BuildAttempt
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FineTuningPipeline:

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.base_model = "llama-3.3-70b-versatile"
        self.output_dir = Path("fine_tuning_data")
        self.output_dir.mkdir(exist_ok=True)
        logger.info("FineTuningPipeline initialised")

    # ─── SYSTEM PROMPT ───────────────────────────────────────────────────
    SYSTEM_PROMPT = """You are NexusFlow, an expert full-stack developer.
You generate complete, production-ready FastAPI + React TypeScript applications.

Rules:
- Always use postgresql+asyncpg:// in DATABASE_URL
- Always use router = APIRouter() in routes.py
- Always use React 18 createRoot in index.tsx
- Always use Vite not Create React App
- Always use VITE_ prefix for env variables
- Always use TailwindCSS for styling
- Never mix SQLAlchemy models with Pydantic schemas
- Never add local modules (database, routes, models) to requirements.txt
- Always include all required files: main.py, database.py, models.py, routes.py, schemas.py, App.tsx, index.tsx"""

    # ─── EXPORT ──────────────────────────────────────────────────────────
    async def export_training_data(
        self,
        db: AsyncSession,
        min_quality: float = 0.8,
        include_successful_builds: bool = True,
        include_error_fixes: bool = True,
    ) -> Path:
        """Export training data as JSONL for Groq fine-tuning."""

        examples = []

        # Get high quality training examples
        result = await db.execute(
            select(TrainingExample).where(
                TrainingExample.quality_score >= min_quality
            ).order_by(TrainingExample.quality_score.desc())
        )
        training_examples = result.scalars().all()

        for ex in training_examples:
            if ex.example_type == "successful_build" and include_successful_builds:
                examples.append({
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": f"Build a complete full-stack application: {ex.input_prompt}"},
                        {"role": "assistant", "content": ex.correct_output}
                    ]
                })
            elif ex.example_type != "successful_build" and include_error_fixes:
                examples.append({
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": f"Fix this error: {ex.input_prompt}\nError type: {ex.error_context}"},
                        {"role": "assistant", "content": ex.correct_output}
                    ]
                })

        # Get successful builds from BuildAttempt
        if include_successful_builds:
            result = await db.execute(
                select(BuildAttempt).where(
                    BuildAttempt.final_status == "success"
                )
            )
            successful_builds = result.scalars().all()

            for build in successful_builds:
                if build.problem_statement and build.fixes_applied:
                    examples.append({
                        "messages": [
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": build.problem_statement},
                            {"role": "assistant", "content": f"Successfully built project. Fixes applied: {build.fixes_applied}"}
                        ]
                    })

        # Write JSONL file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"nexusflow_training_{timestamp}.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

        logger.info("Exported %d training examples to %s", len(examples), output_path)
        return output_path

    # ─── VALIDATE ────────────────────────────────────────────────────────
    def validate_jsonl(self, file_path: Path) -> dict:
        """Validate JSONL file before submitting to Groq."""
        errors = []
        valid = 0
        total = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                total += 1
                try:
                    example = json.loads(line.strip())

                    if "messages" not in example:
                        errors.append(f"Line {i+1}: missing 'messages' key")
                        continue

                    messages = example["messages"]
                    if len(messages) < 2:
                        errors.append(f"Line {i+1}: need at least 2 messages")
                        continue

                    roles = [m.get("role") for m in messages]
                    if "user" not in roles or "assistant" not in roles:
                        errors.append(f"Line {i+1}: need user and assistant messages")
                        continue

                    for msg in messages:
                        if len(msg.get("content", "")) < 10:
                            errors.append(f"Line {i+1}: message too short")
                            break
                    else:
                        valid += 1

                except json.JSONDecodeError as e:
                    errors.append(f"Line {i+1}: JSON error: {e}")

        return {
            "total": total,
            "valid": valid,
            "invalid": total - valid,
            "errors": errors[:10],
            "ready_for_training": valid >= 10 and len(errors) == 0,
            "recommendation": "Ready to fine-tune!" if valid >= 200 else f"Need {200 - valid} more examples (have {valid})"
        }

    # ─── SUBMIT TO GROQ ──────────────────────────────────────────────────
    async def submit_to_groq(self, file_path: Path) -> dict:
        """Submit training data to Groq fine-tuning API."""

        if not self.groq_api_key:
            return {"status": "error", "message": "GROQ_API_KEY not set"}

        # Validate first
        validation = self.validate_jsonl(file_path)
        if not validation["ready_for_training"]:
            return {
                "status": "not_ready",
                "message": validation["recommendation"],
                "validation": validation
            }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                with open(file_path, "rb") as f:
                    upload_response = await client.post(
                        "https://api.groq.com/openai/v1/files",
                        headers={"Authorization": f"Bearer {self.groq_api_key}"},
                        files={"file": (file_path.name, f, "application/x-ndjson")},
                        data={"purpose": "fine-tune"}
                    )

                if upload_response.status_code != 200:
                    return {"status": "error", "message": f"Upload failed: {upload_response.text}"}

                file_id = upload_response.json()["id"]
                logger.info("File uploaded to Groq: %s", file_id)

                # Create fine-tuning job
                ft_response = await client.post(
                    "https://api.groq.com/openai/v1/fine_tuning/jobs",
                    headers={
                        "Authorization": f"Bearer {self.groq_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "training_file": file_id,
                        "model": self.base_model,
                        "hyperparameters": {
                            "n_epochs": 3,
                        },
                        "suffix": "nexusflow"
                    }
                )

                if ft_response.status_code != 200:
                    return {"status": "error", "message": f"Fine-tune job failed: {ft_response.text}"}

                job = ft_response.json()
                logger.info("Fine-tuning job created: %s", job["id"])

                return {
                    "status": "submitted",
                    "job_id": job["id"],
                    "model": self.base_model,
                    "file_id": file_id,
                    "message": f"Fine-tuning job submitted! Job ID: {job['id']}"
                }

        except Exception as e:
            logger.exception("Fine-tuning submission error: %s", e)
            return {"status": "error", "message": str(e)}

    # ─── CHECK STATUS ────────────────────────────────────────────────────
    async def check_job_status(self, job_id: str) -> dict:
        """Check status of a fine-tuning job."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"https://api.groq.com/openai/v1/fine_tuning/jobs/{job_id}",
                    headers={"Authorization": f"Bearer {self.groq_api_key}"}
                )
                job = response.json()
                return {
                    "job_id": job_id,
                    "status": job.get("status"),
                    "model": job.get("fine_tuned_model"),
                    "trained_tokens": job.get("trained_tokens"),
                    "finished_at": job.get("finished_at"),
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ─── ACTIVATE MODEL ──────────────────────────────────────────────────
    def activate_fine_tuned_model(self, model_id: str):
        """Update NexusFlow to use the fine-tuned model."""
        settings_path = Path("src/config/settings.py")
        if settings_path.exists():
            content = settings_path.read_text()
            content = content.replace(
                "llama-3.3-70b-versatile",
                model_id
            )
            settings_path.write_text(content)
            logger.info("Activated fine-tuned model: %s", model_id)
            return {"status": "success", "model": model_id}
        return {"status": "error", "message": "settings.py not found"}

    # ─── AUTO LOOP ───────────────────────────────────────────────────────
    async def auto_loop(self, db: AsyncSession) -> dict:
        """Export → validate → submit in one call.

        Returns immediately with a "not_ready" status if there aren't enough
        high-quality examples yet, or with the Groq submission result if the
        data meets the minimum threshold.
        """
        logger.info("FineTuningPipeline: running auto-loop")
        file_path = await self.export_training_data(db)
        validation = self.validate_jsonl(file_path)

        result: dict = {
            "exported_file": str(file_path),
            "validation": validation,
        }

        if not validation["ready_for_training"]:
            result["status"] = "not_ready"
            result["message"] = validation["recommendation"]
            logger.info("FineTuningPipeline auto-loop: not ready — %s", validation["recommendation"])
            return result

        submission = await self.submit_to_groq(file_path)
        result.update(submission)
        logger.info("FineTuningPipeline auto-loop: submitted — status=%s", submission.get("status"))
        return result


finetune_pipeline = FineTuningPipeline()
