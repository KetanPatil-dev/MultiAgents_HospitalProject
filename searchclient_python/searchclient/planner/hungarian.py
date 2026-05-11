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
    Color-aware greedy assignment.

    Only agents whose color matches a box's color can be assigned that box.
    Box-goal pairs are matched greedily: cheapest (BFS dist box→goal) first.
    Agents are load-balanced: the agent with the fewest tasks gets each pair.

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

    # Build (box, goal) pairs for each character, matched greedily by BFS dist
    pairs: list[tuple[int, int, int, int, str]] = []  # (br, bc, gr, gc, char)

    for char, goal_positions in goals_by_char.items():
        if char not in boxes_by_char:
            continue
        rem_goals = list(goal_positions)
        rem_boxes = list(boxes_by_char[char])

        # Greedy min-cost matching
        while rem_boxes and rem_goals:
            best: tuple[int, int, int, int] | None = None
            best_cost = float("inf")

            for br, bc in rem_boxes:
                for gr, gc in rem_goals:
                    d = dist_map.dist(gr, gc, br, bc)
                    cost = d  # could also add agent-to-box cost, but box→goal dominates
                    if cost < best_cost:
                        best_cost = cost
                        best = (br, bc, gr, gc)

            if best is None:
                break

            br, bc, gr, gc = best
            pairs.append((br, bc, gr, gc, char))
            rem_boxes.remove((br, bc))
            rem_goals.remove((gr, gc))

    # Assign each pair to the eligible agent with fewest current tasks
    for br, bc, gr, gc, char in pairs:
        box_color = box_colors[ord(char) - ord("A")]
        eligible = [
            a for a in agents
            if Color.compatible(agent_colors[a.agent_id], box_color)
        ]
        if not eligible:
            continue
        # Load-balance: fewest tasks wins
        best_agent = min(eligible, key=lambda a: len(result[a.agent_id]))
        result[best_agent.agent_id].append((br, bc, gr, gc, char))

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
