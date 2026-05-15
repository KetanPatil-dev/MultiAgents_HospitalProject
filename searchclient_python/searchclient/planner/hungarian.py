"""
Task assignment: color-aware greedy matching + subgoal ordering.

Public interface (called by Manager._assign_tasks):

    assignment = assign(agents, boxes, goals, dist_map)

    - agents   : list[Agent]
    - boxes    : list[(row, col, char)]  — plannable boxes (from LevelProfile)
    - goals    : list[(row, col, char)]  — box goals (from LevelProfile)
    - dist_map : DistanceMap (BFS distances)
    - returns  : dict {agent_id: [(box_r, box_c, goal_r, goal_c, char), ...]}

    Also:
    agent_goals_assign(agents, agent_goals)
    — returns dict {agent_id: (goal_r, goal_c)} for agents with positional goals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from searchclient.color import Color
from searchclient.state import State

if TYPE_CHECKING:
    from searchclient.agent import Agent
    from searchclient.heuristics import DistanceMap


def assign(
    agents:   "list[Agent]",
    boxes:    list[tuple[int, int, str]],
    goals:    list[tuple[int, int, str]],
    dist_map: "DistanceMap",
) -> dict[int, list[tuple]]:
    """
    Color-aware optimal assignment.

    Box→goal pairing within each character: optimal min-cost matching via the
    Kuhn-Munkres (Hungarian) algorithm — total cost is minimised, not just per
    greedy step. Pair→agent assignment: color-eligible, load-balanced.

    Returns dict {agent_id: [(box_r, box_c, goal_r, goal_c, char), ...]}.
    """
    result: dict[int, list[tuple]] = {a.agent_id: [] for a in agents}

    if not agents or not boxes or not goals:
        return result

    box_colors  = State.box_colors   # indexed by ord(char) - ord('A')
    agent_colors = State.agent_colors  # indexed by agent_id

    # Group goals by character
    goals_by_char: dict[str, list[tuple[int, int]]] = {}
    for gr, gc, gch in goals:
        goals_by_char.setdefault(gch, []).append((gr, gc))

    # Group boxes by character
    boxes_by_char: dict[str, list[tuple[int, int]]] = {}
    for br, bc, bch in boxes:
        boxes_by_char.setdefault(bch, []).append((br, bc))

    # Optimal box→goal matching per character via Kuhn-Munkres
    pairs: list[tuple[int, int, int, int, str]] = []  # (br, bc, gr, gc, char)

    for char, goal_positions in goals_by_char.items():
        if char not in boxes_by_char:
            continue
        box_positions = boxes_by_char[char]

        # Cost matrix: cost[i][j] = BFS dist from box_i to goal_j
        cost = [
            [dist_map.dist(g[0], g[1], b[0], b[1]) for g in goal_positions]
            for b in box_positions
        ]

        # Optimal min-cost assignment (handles rectangular: m boxes, n goals)
        matching = _kuhn_munkres(cost)

        for box_idx, goal_idx in matching.items():
            br, bc = box_positions[box_idx]
            gr, gc = goal_positions[goal_idx]
            pairs.append((br, bc, gr, gc, char))

    # Assign each pair to the eligible agent with fewest current tasks
    for br, bc, gr, gc, char in pairs:
        box_color = box_colors[ord(char) - ord("A")]
        eligible = [
            a for a in agents
            if Color.compatible(agent_colors[a.agent_id], box_color)
        ]
        if not eligible:
            continue
        best_agent = min(eligible, key=lambda a: len(result[a.agent_id]))
        result[best_agent.agent_id].append((br, bc, gr, gc, char))

    return result


def _kuhn_munkres(cost: list[list[int]]) -> dict[int, int]:
    """
    Kuhn-Munkres (Hungarian) algorithm — O(n^3) optimal min-cost assignment.

    Input: cost[i][j] = cost of assigning row i to column j.
    Output: {row_idx: col_idx} mapping each row to its assigned column.

    Rectangular inputs are handled by internal padding with a large sentinel;
    the returned dict only contains real (row, col) pairs.
    """
    if not cost or not cost[0]:
        return {}
    n = len(cost)
    m = len(cost[0])
    size = max(n, m)
    BIG = 10 ** 12  # larger than any real BFS distance (which caps at INF=10^9)
    INF = float("inf")

    # Pad to square
    c = [[cost[i][j] if i < n and j < m else BIG for j in range(size)] for i in range(size)]

    # 1-indexed potentials and assignment arrays
    u = [0] * (size + 1)
    v = [0] * (size + 1)
    p = [0] * (size + 1)   # p[j] = row currently assigned to col j (0 = unassigned)
    way = [0] * (size + 1)

    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, size + 1):
                if not used[j]:
                    cur = c[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    # Extract only real (row, col) assignments — skip padded entries
    result: dict[int, int] = {}
    for j in range(1, size + 1):
        i = p[j]
        if 1 <= i <= n and j <= m:
            result[i - 1] = j - 1
    return result


def agent_goals_assign(
    agents:      "list[Agent]",
    agent_goals: list[tuple[int, int, int]],  # (row, col, agent_id)
) -> dict[int, tuple[int, int] | None]:
    """
    Returns {agent_id: (goal_r, goal_c)} for agents that have a positional goal.
    Agents without a positional goal map to None.
    """
    result: dict[int, tuple[int, int] | None] = {a.agent_id: None for a in agents}
    for gr, gc, aid in agent_goals:
        if 0 <= aid < len(agents):
            result[aid] = (gr, gc)
    return result


def subgoal_order(
    tasks:    list[tuple],
    dist_map: "DistanceMap",
    start_r:  int = 0,
    start_c:  int = 0,
) -> list[tuple]:
    """
    Order tasks for a single agent using greedy nearest-goal from current position.
    tasks are (br, bc, gr, gc, char) 5-tuples.
    """
    if not tasks:
        return []

    ordered: list[tuple] = []
    remaining = list(tasks)
    cur_r, cur_c = start_r, start_c

    while remaining:
        best = min(
            remaining,
            key=lambda t: dist_map.dist(t[2], t[3], cur_r, cur_c),
        )
        ordered.append(best)
        remaining.remove(best)
        cur_r, cur_c = best[2], best[3]  # next "start" is the goal of this task

    return ordered
