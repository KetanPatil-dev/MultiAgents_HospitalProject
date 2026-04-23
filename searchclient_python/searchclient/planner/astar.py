"""
CAPA 3 — Space-time A* para planificación individual.

Interfaz pública (NO cambiar — Manager la llama):

    plan = solve(state, agent_id, goal, constraints=set())

    - state:       State actual (joint state — posiciones de todos los agentes y cajas)
    - agent_id:    índice del agente a planificar (0..n-1)
    - goal:        (box_r, box_c, goal_r, goal_c) — tarea asignada al agente
    - constraints: set de (r, c, t) — celdas prohibidas en determinado timestep
    - returns:     list[Action] con el plan del agente, o None si no hay solución

Implementación a cargo de P2.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from searchclient.action import Action, ActionType
from searchclient.state import State

if TYPE_CHECKING:
    from searchclient.heuristics import DistanceMap


# ---------------------------------------------------------------------------
# Single-agent search node for space-time A*
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _Node:
    f: int
    g: int = field(compare=False)
    r: int = field(compare=False)
    c: int = field(compare=False)
    t: int = field(compare=False)
    # Frozen snapshot of box positions relevant to this agent's task
    boxes: tuple = field(compare=False)
    parent: "_Node | None" = field(compare=False, default=None)
    action: Action = field(compare=False, default=Action.NoOp)


def solve(
    state: State,
    agent_id: int,
    goal: tuple[int, int, int, int],
    constraints: set[tuple[int, int, int]] | None = None,
    dist_map: "DistanceMap | None" = None,
) -> list[Action] | None:
    """
    Space-time A* for a single agent.

    Parameters
    ----------
    state       : current joint State (used for initial positions and box layout)
    agent_id    : which agent to plan for
    goal        : (box_r, box_c, goal_r, goal_c)
    constraints : set of (r, c, t) forbidden positions; other agents' reservations
    dist_map    : precomputed BFS distances (used as heuristic); falls back to
                  Manhattan if None

    Returns
    -------
    List of Actions for this agent only, or None if no solution found.
    """
    if constraints is None:
        constraints = set()

    # TODO(P2): implement space-time A*
    # Sketch:
    #   - Node = (agent_r, agent_c, t, boxes_snapshot)
    #   - heuristic h = dist_map.dist(goal_r, goal_c, agent_r, agent_c)
    #   - successors = all applicable single-agent actions (Move/Push/Pull/NoOp)
    #     filtered by: not wall, not constraint (r', c', t+1), not box conflict
    #   - goal test = box is at (goal_r, goal_c)  [or agent at goal for agent goals]
    #   - closed set keyed on (r, c, t, boxes_snapshot)
    raise NotImplementedError("Space-time A* not yet implemented (P2 task)")


def _h_manhattan(r: int, c: int, goal_r: int, goal_c: int) -> int:
    return abs(r - goal_r) + abs(c - goal_c)


def _h_bfs(
    dist_map: "DistanceMap",
    goal_r: int, goal_c: int,
    from_r: int, from_c: int,
) -> int:
    d = dist_map.dist(goal_r, goal_c, from_r, from_c)
    return d if d < dist_map.INF else _h_manhattan(from_r, from_c, goal_r, goal_c)


def _extract_plan(node: _Node) -> list[Action]:
    actions: list[Action] = []
    n: _Node | None = node
    while n is not None and n.parent is not None:
        actions.append(n.action)
        n = n.parent
    actions.reverse()
    return actions
