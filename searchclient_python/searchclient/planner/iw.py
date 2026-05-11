"""
Iterated Width (IW) for single-agent planning.

IW(k) expands a node only if it is *novel*: at least one k-tuple of atoms
appears for the first time. IW(1) is very fast and sufficient when atoms
(agent position, box position) progress monotonically toward the goal.

Public interface:
    plan = iw_solve(state, agent_id, subgoals, k=1, constraints=set())

    - state     : current joint State
    - agent_id  : which agent to plan for
    - subgoals  : list[(box_r, box_c, goal_r, goal_c, char)] in execution order
    - k         : 1 or 2 (IW(1) is fast; IW(2) catches more structure)
    - returns   : concatenated list[Action] for all subgoals, or None
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from searchclient.action import Action, ActionType
from searchclient.state import State

if TYPE_CHECKING:
    pass


def iw_solve(
    state:       State,
    agent_id:    int,
    subgoals:    list[tuple],
    k:           int = 1,
    constraints: set[tuple[int, int, int]] | None = None,
) -> list[Action] | None:
    """
    IW(k) over a sequence of subgoals (solve each sequentially).
    Returns concatenated plan or None if any subgoal fails.
    """
    if constraints is None:
        constraints = set()

    full_plan: list[Action] = []
    current_state = state

    for subgoal in subgoals:
        box_r0, box_c0, goal_r, goal_c = subgoal[0], subgoal[1], subgoal[2], subgoal[3]
        box_char = subgoal[4] if len(subgoal) > 4 else ""

        # Find current box position
        if box_char:
            box_r, box_c = _find_box(current_state, box_char, goal_r, goal_c)
        else:
            box_r, box_c = box_r0, box_c0

        if box_r is None:
            continue  # box already at goal or not found

        if box_r == goal_r and box_c == goal_c:
            continue  # already satisfied

        partial = _iw_single(current_state, agent_id, box_r, box_c, goal_r, goal_c, k, constraints)
        if partial is None:
            return None

        full_plan.extend(partial)
        # Advance state
        current_state = _apply_plan(current_state, agent_id, partial)

    return full_plan


# ---------------------------------------------------------------------------
# IW for a single (box → goal) subgoal
# ---------------------------------------------------------------------------

def _iw_single(
    state:       State,
    agent_id:    int,
    box_r:       int,
    box_c:       int,
    goal_r:      int,
    goal_c:      int,
    k:           int,
    constraints: set[tuple[int, int, int]],
) -> list[Action] | None:
    """BFS with IW(k) novelty pruning for moving one box to one goal."""

    walls    = State.walls
    num_rows = len(walls)
    num_cols = len(walls[0]) if num_rows > 0 else 0

    agent_r0 = state.agent_rows[agent_id]
    agent_c0 = state.agent_cols[agent_id]

    # Static boxes (all others)
    static = [row[:] for row in state.boxes]
    if 0 <= box_r < num_rows and 0 <= box_c < num_cols:
        static[box_r][box_c] = ""

    # Seen atoms (for k=1: individual atoms; for k=2: pairs)
    seen_atoms: set = set()

    MAX_NODES = 200_000

    # Queue entries: (agent_r, agent_c, box_r, box_c, t, parent_idx, action)
    nodes: list[tuple] = []
    start = (agent_r0, agent_c0, box_r, box_c, 0, -1, Action.NoOp)

    if not _is_novel(k, agent_r0, agent_c0, box_r, box_c, seen_atoms):
        return None

    nodes.append(start)
    queue: deque[int] = deque([0])

    def blocked(r: int, c: int, t: int) -> bool:
        if not (0 <= r < num_rows and 0 <= c < num_cols):
            return True
        if walls[r][c]:
            return True
        if static[r][c]:
            return True
        if (r, c, t) in constraints:
            return True
        return False

    while queue:
        if len(nodes) > MAX_NODES:
            break

        idx = queue.popleft()
        ar, ac, bxr, bxc, t, par, _ = nodes[idx]

        if bxr == goal_r and bxc == goal_c:
            return _extract_iw_plan(nodes, idx)

        tn = t + 1

        for action in Action:
            if action.type is ActionType.NoOp:
                continue

            new_ar, new_ac, new_bxr, new_bxc = ar, ac, bxr, bxc

            if action.type is ActionType.Move:
                new_ar = ar + action.agent_row_delta
                new_ac = ac + action.agent_col_delta
                if blocked(new_ar, new_ac, tn):
                    continue
                if new_ar == bxr and new_ac == bxc:
                    continue

            elif action.type is ActionType.Push:
                if ar + action.agent_row_delta != bxr or ac + action.agent_col_delta != bxc:
                    continue
                new_bxr = bxr + action.box_row_delta
                new_bxc = bxc + action.box_col_delta
                if blocked(new_bxr, new_bxc, tn):
                    continue
                new_ar, new_ac = bxr, bxc

            elif action.type is ActionType.Pull:
                if ar - action.box_row_delta != bxr or ac - action.box_col_delta != bxc:
                    continue
                new_ar = ar + action.agent_row_delta
                new_ac = ac + action.agent_col_delta
                if blocked(new_ar, new_ac, tn):
                    continue
                new_bxr, new_bxc = ar, ac

            # Novelty check
            if not _is_novel(k, new_ar, new_ac, new_bxr, new_bxc, seen_atoms):
                continue

            child_idx = len(nodes)
            nodes.append((new_ar, new_ac, new_bxr, new_bxc, tn, idx, action))
            queue.append(child_idx)

    return None


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------

def _atoms_k1(ar: int, ac: int, bxr: int, bxc: int) -> frozenset:
    return frozenset({("ar", ar), ("ac", ac), ("br", bxr), ("bc", bxc)})


def _is_novel(k: int, ar: int, ac: int, bxr: int, bxc: int, seen: set) -> bool:
    """Return True iff at least one atom (or pair for k=2) is new; update seen."""
    atoms = _atoms_k1(ar, ac, bxr, bxc)
    novel = False

    if k == 1:
        for atom in atoms:
            if atom not in seen:
                seen.add(atom)
                novel = True
    else:  # k == 2
        atom_list = list(atoms)
        for i in range(len(atom_list)):
            for j in range(i + 1, len(atom_list)):
                pair = frozenset({atom_list[i], atom_list[j]})
                if pair not in seen:
                    seen.add(pair)
                    novel = True

    return novel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_iw_plan(nodes: list[tuple], idx: int) -> list[Action]:
    actions: list[Action] = []
    while nodes[idx][5] != -1:
        actions.append(nodes[idx][6])
        idx = nodes[idx][5]
    actions.reverse()
    return actions


def _apply_plan(state: State, agent_id: int, plan: list[Action]) -> State:
    """Apply a single-agent plan to the joint state (other agents stay)."""
    from searchclient.action import Action as A
    current = state
    num_agents = len(state.agent_rows)
    for action in plan:
        joint = [A.NoOp] * num_agents
        joint[agent_id] = action
        current = current.result(joint)
    return current


def _find_box(
    state:    State,
    box_char: str,
    goal_r:   int,
    goal_c:   int,
) -> tuple[int | None, int | None]:
    candidates = []
    for r, row in enumerate(state.boxes):
        for c, ch in enumerate(row):
            if ch == box_char and not (r == goal_r and c == goal_c):
                candidates.append((abs(r - goal_r) + abs(c - goal_c), r, c))
    if not candidates:
        return None, None
    candidates.sort()
    return candidates[0][1], candidates[0][2]
