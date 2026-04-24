"""Pydantic-AI Move Agent fuer die Roboterbewegung."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic_ai import Agent

from src.agent.tools.coordinate_tool import move_to_coordinate
from src.agent.tools.motion_types import MoveAgentDeps
from src.agent.tools.turn_tool import turn
from src.agent.tools.walk_tool import stop_movement, walk

DEFAULT_MOVE_AGENT_MODEL = os.getenv("MOVE_AGENT_MODEL", "test")

MOVE_AGENT_INSTRUCTIONS = """
Du bist der Move Agent des agentenbasierten Litter-Detection-Roboters.

Deine Aufgabe:
- Führe Bewegungsanweisungen des Explore Agents in sichere Roboterbefehle aus.
- Verwende `move_to_coordinate`, wenn der Explore Agent x/y-Zielkoordinaten liefert.
- Verwende `walk`, um geradeaus vorwärts oder rückwärts zu laufen.
- Verwende `turn`, um den Roboterhund nach links oder rechts zu drehen.
- Verwende `stop_movement`, sobald eine Pause, ein Stopp oder ein unklarer Zustand vorliegt.

Sicherheitsregeln:
- Gib keine Bewegung aus, wenn ein E-Stop aktiv ist.
- Controller-Befehle haben Vorrang vor Explore- und autonomen Befehlen.
- Explore-/Autonom-Geschwindigkeiten sind auf 0.3 m/s und 30 deg/s begrenzt.
- Veraltete Explore-Befehle dürfen nicht ausgeführt werden.

Antworte nach Tool-Aufrufen kurz mit dem ausgeführten Schritt und dem Ergebnis.
"""


def create_move_agent(model: str | None = None) -> Agent[MoveAgentDeps, str]:
    """Erstellt eine Move-Agent-Instanz mit den registrierten Bewegungstools."""

    return Agent(
        model or DEFAULT_MOVE_AGENT_MODEL,
        deps_type=MoveAgentDeps,
        tools=[move_to_coordinate, walk, turn, stop_movement],
        instructions=MOVE_AGENT_INSTRUCTIONS,
    )


move_agent = create_move_agent()


def run_move_agent_sync(prompt: str, deps: MoveAgentDeps | None = None) -> str:
    """Führt den Move Agent synchron aus und gibt die finale Agentenantwort zurück."""

    result = move_agent.run_sync(prompt, deps=deps or MoveAgentDeps.from_env())
    return result.output


__all__ = [
    "MOVE_AGENT_INSTRUCTIONS",
    "create_move_agent",
    "move_agent",
    "run_move_agent_sync",
]
