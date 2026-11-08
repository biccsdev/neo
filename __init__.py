"""NEO — voice-first macOS mission control for Hermes Agent.

The repository root is the installable package. Hermes imports this module once
and calls ``register(ctx)``. Importing it touches no profile: NEO's durable
services are built lazily, on the first conversational tool call or hook turn.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

_PLUGIN_DIR = Path(__file__).resolve().parent
# Hermes loads a directory plugin by file location without extending
# ``sys.path``; the plugin's own modules live in this directory.
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))


def register(ctx) -> None:
    """Wire NEO's conversational tools, middleware and bundled skill."""
    from conversation import (
        pre_llm_call,
        register_capability_context,
        register_conversation_tools,
    )

    register_capability_context(ctx)
    register_conversation_tools(ctx)

    register_hook = getattr(ctx, "register_hook", None)
    if callable(register_hook):
        register_hook("pre_llm_call", pre_llm_call)

    _register_bundled_skills(ctx)


def _register_bundled_skills(ctx) -> None:
    """Register ``skills/*/SKILL.md`` when the host context supports it."""
    register_skill = getattr(ctx, "register_skill", None)
    if not callable(register_skill):
        return
    skills_dir = _PLUGIN_DIR / "skills"
    if not skills_dir.is_dir():
        return
    for child in sorted(skills_dir.iterdir()):
        skill_md = child / "SKILL.md"
        if not child.is_dir() or not skill_md.is_file():
            continue
        try:
            register_skill(child.name, skill_md)
        except Exception:
            logger.warning("neo: could not register skill %s", child.name, exc_info=True)
