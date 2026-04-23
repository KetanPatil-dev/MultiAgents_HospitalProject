"""
CAPA 2 — Task assignment: Hungarian algorithm + reasignación dinámica.

Interfaz pública (NO cambiar — Manager la llama):

    assignment = assign(agents, boxes, goals, dist_map)

    - agents:    list[Agent]
    - boxes:     list[(row, col, char)] — cajas planificables
    - goals:     list[(row, col, char)] — goals de caja
    - dist_map:  DistanceMap precomputado
    - returns:   dict {agent_id: [(box_r, box_c, goal_r, goal_c), ...]}

También expone:

    ordered = subgoal_order(tasks, dist_map)
    — ordena las tareas de un agente en un DAG de dependencias mínimas.

Implementación a cargo de P5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from searchclient.agent import Agent
    from searchclient.heuristics import DistanceMap


def assign(
    agents: "list[Agent]",
    boxes: list[tuple[int, int, str]],
    goals: list[tuple[int, int, str]],
    dist_map: "DistanceMap",
) -> dict[int, list[tuple[int, int, int, int]]]:
    """
    Hungarian algorithm: assign each box-goal pair to the best agent.

    Cost matrix C[i][j] = dist(agent_i_start, box_j) + dist(box_j, goal_j).
    Minimise sum of costs across all assignments (one agent per box-goal pair).

    Returns dict {agent_id: [(box_r, box_c, goal_r, goal_c), ...]}.
    Multiple boxes can be assigned to the same agent (sequential subgoals).
    """
    # TODO(P5): implement Hungarian via scipy.optimize.linear_sum_assignment
    # or hand-rolled Kuhn-Munkres.

    # Stub: round-robin assignment ignoring distances
    result: dict[int, list[tuple[int, int, int, int]]] = {a.agent_id: [] for a in agents}
    if not agents:
        return result

    # Match each goal to its box (same char), then assign round-robin
    goal_map: dict[str, list[tuple[int, int]]] = {}
    for gr, gc, gch in goals:
        goal_map.setdefault(gch, []).append((gr, gc))

    pairs: list[tuple[int, int, int, int]] = []
    for br, bc, bch in boxes:
        if bch in goal_map and goal_map[bch]:
            gr, gc = goal_map[bch].pop(0)
            pairs.append((br, bc, gr, gc))

    for idx, pair in enumerate(pairs):
        agent_id = agents[idx % len(agents)].agent_id
        result[agent_id].append(pair)

    return result


def subgoal_order(
    tasks: list[tuple[int, int, int, int]],
    dist_map: "DistanceMap",
) -> list[tuple[int, int, int, int]]:
    """
    Order tasks for a single agent to minimise total travel distance.
    Returns a permutation of tasks (greedy nearest-neighbour as stub).

    Full implementation: build dependency DAG, topological sort,
    break ties by travel cost.
    """
    # TODO(P5): implement proper DAG ordering
    # Stub: greedy nearest-goal ordering
    if not tasks:
        return []

    ordered: list[tuple[int, int, int, int]] = []
    remaining = list(tasks)
    cur_r, cur_c = tasks[0][0], tasks[0][1]

    while remaining:
        best = min(
            remaining,
            key=lambda t: dist_map.dist(t[2], t[3], cur_r, cur_c),
        )
        ordered.append(best)
        remaining.remove(best)
        cur_r, cur_c = best[2], best[3]

    return ordered
