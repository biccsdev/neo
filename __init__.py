"""NEO — voice-first macOS mission control for Hermes Agent.

The repository root is the installable package. Hermes imports this module
once and calls ``register(ctx)``; the registration surface grows task by task.
"""

from __future__ import annotations

__version__ = "0.1.0"


def register(ctx) -> None:
    """Wire NEO into Hermes.

    Deliberately empty for the package skeleton: tools, hooks, and skills
    arrive with later tasks, and this task promises only a clean load.
    """
    return None
