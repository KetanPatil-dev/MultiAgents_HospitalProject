"""
Conflict-Based Search (CBS) for multi-agent path finding.

CBS finds conflict-free joint plans by:
  1. Independently planning each agent's path (low-level: A*)
  2. Detecting conflicts in the joint trajectory
  3. Branching on each conflict by adding a constraint to one of the conflicting agents
  4. Re-planning the constrained agent and repeating

Public interface:
    plans = cbs_solve(state, agent_goals, dist_map, max_nodes=300)

    - state       : joint State
    - agent_goals : dict {agent_id: goal_tuple}  where goal_tuple is what
                    astar.solve expects: (box_r, box_c, goal_r, goal_c[, char]) or
                    (None, None, goal_r, goal_c) for nav-only
    - dist_map    : DistanceMap
    - max_nodes   : maximum CBS nodes to expand before giving up
    - returns     : dict {agent_id: list[Action]} or None on failure
"""

from __future__ import annotations

import heapq
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from searchclient.action import Action, ActionType
from searchclient.planner.astar import solve
from searchclient.state import State

if TYPE_CHECKING:
    from searchclient.heuristics import DistanceMap


@dataclass(order=True)
class _CBSNode:
    cost: int
    tie: int = field(compare=False)
    constraints: dict = field(compare=False, default_factory=dict)  # agent_id -> set[(r,c,t)]
    plans: dict = field(compare=False, default_factory=dict)        # agent_id -> list[Action]


def cbs_solve(
    state: State,
    agent_goals: dict,
    dist_map: "DistanceMap | None" = None,
    max_nodes: int = 300,
) -> dict | None:
    """Run CBS over the given agents+goals. Returns dict {agent_id: plan} or None."""
    if not agent_goals:
        return {}

    # Build root: independent shortest paths.
    # Agents whose individual A* fails are excluded — they will be planned
    # reactively in the main loop instead of being part of CBS.
    root = _CBSNode(cost=0, tie=0)
    excluded: list[int] = []
    for aid, goal in agent_goals.items():
        plan = solve(state, aid, goal, set(), dist_map)
        if plan is None:
            excluded.append(aid)
            print(
                f"  CBS: agent {aid} excluded — no individual path; will plan reactively.",
                file=sys.stderr, flush=True,
            )
            continue
        root.plans[aid] = plan
        root.cost += len(plan)
        root.constraints[aid] = set()

    # Remove excluded agents from agent_goals so CBS only considers planned ones
    agent_goals = {aid: g for aid, g in agent_goals.items() if aid not in excluded}
    if not agent_goals:
        print("  CBS: no agents could be planned individually.", file=sys.stderr, flush=True)
        return {}  # Return empty dict instead of None — let manager handle reactively

    counter = 1
    open_set: list = [(root.cost, 0, root)]

    while open_set:
        if counter > max_nodes:
            print(f"  CBS: node limit reached ({max_nodes}).", file=sys.stderr, flush=True)
            return None

        _, _, node = heapq.heappop(open_set)

        # Detect first conflict in joint plan
        conflict = _find_first_conflict(state, node.plans)
        if conflict is None:
            print(
                f"  CBS: solution found, cost={node.cost}, nodes_expanded={counter}.",
                file=sys.stderr, flush=True,
            )
            return node.plans

        a1, a2, c_r, c_c, c_t = conflict

        # Branch: add constraint (c_r, c_c, c_t) to each conflicting agent
        for offending_aid in (a1, a2):
            new_constraints = {a: set(s) for a, s in node.constraints.items()}
            new_constraints.setdefault(offending_aid, set()).add((c_r, c_c, c_t))

            # Cap constraint set size for performance
            if len(new_constraints[offending_aid]) > 150:
                continue

            # Replan the constrained agent
            new_plan = solve(
                state,
                offending_aid,
                agent_goals[offending_aid],
                new_constraints[offending_aid],
                dist_map,
            )
            if new_plan is None:
                continue  # infeasible branch

            new_plans = dict(node.plans)
            new_plans[offending_aid] = new_plan
            new_cost = sum(len(p) for p in new_plans.values())

            counter += 1
            child = _CBSNode(
                cost=new_cost,
                tie=counter,
                constraints=new_constraints,
                plans=new_plans,
            )
            heapq.heappush(open_set, (new_cost, counter, child))

    print(f"  CBS: open-set exhausted after {counter} nodes.", file=sys.stderr, flush=True)
    return None


def _find_first_conflict(
    state: State, plans: dict
) -> tuple[int, int, int, int, int] | None:
    """
    Simulate joint plan execution and return the first conflict:
        (agent1_id, agent2_id, conflict_row, conflict_col, conflict_timestep)
    Returns None if conflict-free.

    Conflict types:
      - Vertex: two agents at same cell at same time
      - Edge (swap): two agents exchanging positions across one timestep
      - Box destination: two boxes/agents arriving at same cell
    """
    if not plans:
        return None

    num_agents = len(state.agent_rows)
    max_len = max(len(p) for p in plans.values()) if plans else 0

    # Track each agent's position over time
    # positions[t][agent_id] = (row, col)
    pos_history: list[dict] = [
        {aid: (state.agent_rows[aid], state.agent_cols[aid]) for aid in plans.keys()}
    ]

    current = state
    for t in range(max_len):
        # Build joint action for this timestep
        joint = [Action.NoOp] * num_agents
        for aid, plan in plans.items():
            if t < len(plan):
                joint[aid] = plan[t]

        # Pre-check vertex / edge conflicts before applying
        prev_pos = pos_history[-1]
        new_pos: dict = {}
        box_dests: dict = {}  # (r, c) -> aid pushing/pulling box there

        for aid in plans.keys():
            action = joint[aid]
            ar, ac = current.agent_rows[aid], current.agent_cols[aid]

            if action.type is ActionType.NoOp:
                new_pos[aid] = (ar, ac)
                continue

            if action.type is ActionType.Move:
                nr = ar + action.agent_row_delta
                nc = ac + action.agent_col_delta
                new_pos[aid] = (nr, nc)

            elif action.type is ActionType.Push:
                br = ar + action.agent_row_delta
                bc = ac + action.agent_col_delta
                bdr = br + action.box_row_delta
                bdc = bc + action.box_col_delta
                new_pos[aid] = (br, bc)
                # Check box destination conflict
                if (bdr, bdc) in box_dests:
                    return (aid, box_dests[(bdr, bdc)], bdr, bdc, t + 1)
                box_dests[(bdr, bdc)] = aid

            elif action.type is ActionType.Pull:
                nr = ar + action.agent_row_delta
                nc = ac + action.agent_col_delta
                new_pos[aid] = (nr, nc)

        # Vertex conflict: two agents at same cell
        seen: dict = {}
        for aid, pos in new_pos.items():
            if pos in seen:
                return (aid, seen[pos], pos[0], pos[1], t + 1)
            seen[pos] = aid

        # Edge conflict: agent A→B and agent B→A swap
        for aid1, pos1 in new_pos.items():
            for aid2, pos2 in new_pos.items():
                if aid1 >= aid2:
                    continue
                if pos1 == prev_pos.get(aid2) and pos2 == prev_pos.get(aid1):
                    return (aid1, aid2, pos1[0], pos1[1], t + 1)

        # Box-destination conflict already detected above

        # Apply the joint action to advance simulation state
        try:
            current = current.result(joint)
        except (IndexError, Exception):
            # Plan is invalid for state — treat as conflict
            for aid in plans.keys():
                if joint[aid].type is not ActionType.NoOp:
                    pos = new_pos[aid]
                    return (aid, aid, pos[0], pos[1], t + 1)
            return None

        pos_history.append(new_pos)

    return None
