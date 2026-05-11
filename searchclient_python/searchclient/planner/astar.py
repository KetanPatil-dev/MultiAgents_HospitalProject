"""
Space-time A* for single-agent planning in the hospital domain.

Public interface (called by Agent.replan):
    plan = solve(state, agent_id, goal, constraints, dist_map)

    - state       : current joint State
    - agent_id    : which agent to plan for
    - goal        : (box_r, box_c, goal_r, goal_c)  OR
                    (box_r, box_c, goal_r, goal_c, box_char)  — move that box to goal
                    (None, None, goal_r, goal_c)              — navigate agent to goal (no box)
    - constraints : set of (r, c, t) forbidden at time t for this agent
    - dist_map    : precomputed BFS distances; Manhattan fallback if None
    - returns     : list[Action] for this agent only, or None if unsolvable
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from searchclient.action import Action, ActionType
from searchclient.state import State

if TYPE_CHECKING:
    from searchclient.heuristics import DistanceMap

INF = 10 ** 9
MAX_G = 500          # max plan length before we give up
MAX_CLOSED = 300_000  # memory guard
MAX_HEAP   = 1_000_000  # hard heap cap to prevent infinite memory growth


# ---------------------------------------------------------------------------
# Search node
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _Node:
    f:      int
    g:      int    = field(compare=False)
    ar:     int    = field(compare=False)   # agent row
    ac:     int    = field(compare=False)   # agent col
    bxr:    int    = field(compare=False)   # box row  (-1 = no-box task)
    bxc:    int    = field(compare=False)   # box col  (-1 = no-box task)
    t:      int    = field(compare=False)   # timestep (for space-time)
    parent: "_Node | None" = field(compare=False, default=None)
    action: Action          = field(compare=False, default=Action.NoOp)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def solve(
    state:       State,
    agent_id:    int,
    goal:        tuple,
    constraints: set[tuple[int, int, int]] | None = None,
    dist_map:    "DistanceMap | None" = None,
) -> list[Action] | None:
    """
    A* (optionally space-time when constraints are non-empty) for one agent.

    goal may be:
      (box_r, box_c, goal_r, goal_c)         — move box to goal cell
      (box_r, box_c, goal_r, goal_c, char)   — same, char is advisory
      (None,  None,  goal_r, goal_c)          — navigate agent to goal cell (no box)
    """
    if constraints is None:
        constraints = set()

    box_r0: int | None = goal[0]
    box_c0: int | None = goal[1]
    goal_r:  int = goal[2]
    goal_c:  int = goal[3]

    nav_only = (box_r0 is None)  # pure navigation, no box to move

    # Already satisfied?
    if nav_only:
        if state.agent_rows[agent_id] == goal_r and state.agent_cols[agent_id] == goal_c:
            return []
    else:
        if box_r0 == goal_r and box_c0 == goal_c:
            return []

    agent_r0 = state.agent_rows[agent_id]
    agent_c0 = state.agent_cols[agent_id]

    walls    = State.walls
    num_rows = len(walls)
    num_cols = len(walls[0]) if num_rows > 0 else 0

    # Static obstacle grid: all boxes except the one being moved
    static = [row[:] for row in state.boxes]
    if not nav_only and 0 <= box_r0 < num_rows and 0 <= box_c0 < num_cols:
        static[box_r0][box_c0] = ""

    # Other agents' current positions (treated as static obstacles).
    # This prevents planning through cells currently occupied by teammates.
    # When space-time constraints are active, their future positions are
    # encoded in `constraints`; initial positions still serve as a baseline.
    other_agents: frozenset[tuple[int, int]] = frozenset(
        (state.agent_rows[a], state.agent_cols[a])
        for a in range(len(state.agent_rows)) if a != agent_id
    )

    use_time = len(constraints) > 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def blocked(r: int, c: int, t: int) -> bool:
        if not (0 <= r < num_rows and 0 <= c < num_cols):
            return True
        if walls[r][c]:
            return True
        if static[r][c]:
            return True
        # Other agents at t=0 are static obstacles; after that, rely on constraints
        if t == 0 and (r, c) in other_agents:
            return True
        if (r, c, t) in constraints:
            return True
        return False

    def h(ar: int, ac: int, bxr: int, bxc: int) -> int:
        """Admissible heuristic."""
        if nav_only:
            # Distance agent → goal
            if dist_map is not None:
                d = dist_map.dist(goal_r, goal_c, ar, ac)
                return d if d < INF else abs(ar - goal_r) + abs(ac - goal_c)
            return abs(ar - goal_r) + abs(ac - goal_c)

        if bxr == goal_r and bxc == goal_c:
            return 0
        # dist(box → goal)
        if dist_map is not None:
            btg = dist_map.dist(goal_r, goal_c, bxr, bxc)
            if btg >= INF:
                btg = abs(bxr - goal_r) + abs(bxc - goal_c)
        else:
            btg = abs(bxr - goal_r) + abs(bxc - goal_c)
        # dist(agent → adjacent-to-box): need to reach box first
        atb = max(0, abs(ar - bxr) + abs(ac - bxc) - 1)
        return btg + atb

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    bxr0 = box_r0 if not nav_only else -1
    bxc0 = box_c0 if not nav_only else -1

    counter = 0
    start = _Node(
        f=h(agent_r0, agent_c0, bxr0, bxc0),
        g=0, ar=agent_r0, ac=agent_c0, bxr=bxr0, bxc=bxc0,
        t=0, parent=None, action=Action.NoOp,
    )
    heap = [(start.f, counter, start)]
    closed: set[tuple] = set()

    while heap:
        if len(closed) > MAX_CLOSED or len(heap) > MAX_HEAP:
            break  # memory guard

        _, _, node = heapq.heappop(heap)
        ar, ac, bxr, bxc, t = node.ar, node.ac, node.bxr, node.bxc, node.t

        key = (ar, ac, bxr, bxc, t) if use_time else (ar, ac, bxr, bxc)
        if key in closed:
            continue
        closed.add(key)

        # Goal check
        if nav_only:
            if ar == goal_r and ac == goal_c:
                return _extract_plan(node)
        else:
            if bxr == goal_r and bxc == goal_c:
                return _extract_plan(node)

        if node.g >= MAX_G:
            continue

        tn = t + 1

        for action in Action:
            new_ar, new_ac = ar, ac
            new_bxr, new_bxc = bxr, bxc

            if action.type is ActionType.NoOp:
                if (ar, ac, tn) in constraints:
                    continue
                # position unchanged

            elif action.type is ActionType.Move:
                new_ar = ar + action.agent_row_delta
                new_ac = ac + action.agent_col_delta
                if blocked(new_ar, new_ac, tn):
                    continue
                if not nav_only and new_ar == bxr and new_ac == bxc:
                    continue  # can't walk into the target box

            elif action.type is ActionType.Push:
                if nav_only:
                    continue  # no box to push in nav-only mode
                # Box must be in agent's move direction
                if ar + action.agent_row_delta != bxr or ac + action.agent_col_delta != bxc:
                    continue
                new_bxr = bxr + action.box_row_delta
                new_bxc = bxc + action.box_col_delta
                if blocked(new_bxr, new_bxc, tn):
                    continue
                new_ar, new_ac = bxr, bxc   # agent steps into box's old cell
                if (new_ar, new_ac, tn) in constraints:
                    continue

            elif action.type is ActionType.Pull:
                if nav_only:
                    continue
                # Box must be behind the agent (opposite of move direction)
                if ar - action.box_row_delta != bxr or ac - action.box_col_delta != bxc:
                    continue
                new_ar = ar + action.agent_row_delta
                new_ac = ac + action.agent_col_delta
                if blocked(new_ar, new_ac, tn):
                    continue
                new_bxr, new_bxc = ar, ac   # box pulled to agent's old cell
                if (new_bxr, new_bxc, tn) in constraints:
                    continue

            nk = (new_ar, new_ac, new_bxr, new_bxc, tn) if use_time else (new_ar, new_ac, new_bxr, new_bxc)
            if nk in closed:
                continue

            new_g = node.g + 1
            new_h = h(new_ar, new_ac, new_bxr, new_bxc)
            counter += 1
            heapq.heappush(heap, (new_g + new_h, counter, _Node(
                f=new_g + new_h, g=new_g,
                ar=new_ar, ac=new_ac, bxr=new_bxr, bxc=new_bxc,
                t=tn, parent=node, action=action,
            )))

    return None   # no solution found within limits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_plan(node: _Node) -> list[Action]:
    actions: list[Action] = []
    n: _Node | None = node
    while n is not None and n.parent is not None:
        actions.append(n.action)
        n = n.parent
    actions.reverse()
    return actions
