"""
CAPA 3 — Iterated Width (IW) para planificación individual.

IW(k) expande un nodo solo si es "novel": al menos una tupla de k átomos
aparece por primera vez. IW(1) y IW(2) son eficientes para dominios con
objetivos independientes (warehouse masivo, muchas cajas/agente).

Interfaz pública (NO cambiar):

    plan = iw_solve(state, agent_id, subgoals, k=1, constraints=set())

    - state:     State actual
    - agent_id:  índice del agente
    - subgoals:  list[(box_r, box_c, goal_r, goal_c)] en orden de ejecución
    - k:         anchura (1 o 2)
    - returns:   list[Action] concatenado para todos los subgoals, o None

Implementación a cargo de P3.
"""

from __future__ import annotations

from searchclient.action import Action
from searchclient.state import State


def iw_solve(
    state: State,
    agent_id: int,
    subgoals: list[tuple[int, int, int, int]],
    k: int = 1,
    constraints: set[tuple[int, int, int]] | None = None,
) -> list[Action] | None:
    """
    IW(k) over individual subgoals.

    Calls _iw_single() sequentially for each subgoal, threading the resulting
    intermediate state into the next call.

    Returns concatenated plan, or None if any subgoal fails.
    """
    if constraints is None:
        constraints = set()

    # TODO(P3): implement sequential IW over subgoals
    # full_plan: list[Action] = []
    # current = state
    # for goal in subgoals:
    #     partial = _iw_single(current, agent_id, goal, k, constraints)
    #     if partial is None:
    #         return None
    #     full_plan.extend(partial)
    #     current = _apply_plan(current, agent_id, partial)
    # return full_plan
    raise NotImplementedError("IW not yet implemented (P3 task)")


def _iw_single(
    state: State,
    agent_id: int,
    goal: tuple[int, int, int, int],
    k: int,
    constraints: set[tuple[int, int, int]],
) -> list[Action] | None:
    """
    IW(k) for a single (box → goal) subgoal.

    Atoms (for k=1): {(feature_name, value)}
      - ('ar', agent_row), ('ac', agent_col)
      - ('br', box_row), ('bc', box_col)
    Novel iff at least one atom is new.

    For k=2: pairs of atoms; catches more structure at cost of more memory.
    """
    # TODO(P3): BFS with novelty pruning
    #   novel_table: set of seen atom (or atom-pair) tuples
    #   expand only if is_novel(node, novel_table, k)
    raise NotImplementedError


def _atoms_k1(agent_r: int, agent_c: int, box_r: int, box_c: int) -> frozenset:
    return frozenset({
        ("ar", agent_r),
        ("ac", agent_c),
        ("br", box_r),
        ("bc", box_c),
    })


def _is_novel_k1(atoms: frozenset, seen: set) -> bool:
    for atom in atoms:
        if atom not in seen:
            return True
    return False


def _update_seen_k1(atoms: frozenset, seen: set) -> None:
    seen.update(atoms)
