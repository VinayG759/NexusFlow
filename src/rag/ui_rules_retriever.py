"""UI Rules Retriever.

For each build, selects one rule per category from the UI rules knowledge base
to compose a unique design brief. Selection is driven by keyword affinity between
the project description and each rule's vibe tags, plus a random noise component
so that identical descriptions still produce different builds.
"""

import random
import re
from src.rag.ui_rules_kb import UI_RULES, CATEGORIES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _tokenise(text: str) -> set[str]:
    """Lowercase alphanumeric tokens from a string."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _score_rule(rule: dict, project_tokens: set[str], rng: random.Random) -> float:
    """Score a rule against the project description.

    Score = number of vibe-tag matches in the project description
            + small random noise so repeated builds diverge.
    """
    vibe_tokens = set(" ".join(rule["vibe"]).lower().split())
    keyword_score = len(project_tokens & vibe_tokens)
    noise = rng.uniform(0.0, 1.5)
    return keyword_score + noise


def mix_rules_for_project(
    problem_statement: str,
    project_name: str = "",
    seed: int | None = None,
) -> dict[str, dict]:
    """Return one selected rule per category, keyed by category name.

    Args:
        problem_statement: The user's problem statement / project description.
        project_name: Optional project name (adds more signal).
        seed: Optional RNG seed for reproducibility in tests.

    Returns:
        Dict mapping category → selected rule dict.
    """
    rng = random.Random(seed)
    combined_text = f"{project_name} {problem_statement}"
    tokens = _tokenise(combined_text)

    selected: dict[str, dict] = {}
    for category in CATEGORIES:
        candidates = [r for r in UI_RULES if r["category"] == category]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: _score_rule(r, tokens, rng))
        selected[category] = best
        logger.debug("UI rules: category=%s → %s", category, best["name"])

    logger.info(
        "UI rules mixed for %r — %d categories, rules: %s",
        project_name or problem_statement[:40],
        len(selected),
        ", ".join(f"{cat}:{r['name']}" for cat, r in selected.items()),
    )
    return selected


def build_design_brief(selected_rules: dict[str, dict]) -> str:
    """Convert selected rules into a structured design brief string.

    The brief is injected directly into the UIDesignAgent system prompt.
    """
    if not selected_rules:
        return ""

    lines = [
        "═══════════════════════════════════════════════════════════",
        "UNIQUE DESIGN BRIEF FOR THIS BUILD",
        "Apply ALL of the following rules precisely and consistently",
        "throughout every component and file in this project.",
        "═══════════════════════════════════════════════════════════",
        "",
    ]

    category_order = [
        "color", "typography", "spacing", "radius",
        "shadow", "card", "button", "animation", "layout", "texture",
    ]

    for cat in category_order:
        rule = selected_rules.get(cat)
        if not rule:
            continue
        label = cat.upper().replace("_", " ")
        lines.append(f"▸ {label} — {rule['name']}")
        lines.append(f"  {rule['rule']}")
        lines.append(f"  Tailwind hints: {rule['tailwind_hints']}")
        lines.append("")

    lines.append(
        "These rules are non-negotiable. Every component must express this design language."
    )
    return "\n".join(lines)
