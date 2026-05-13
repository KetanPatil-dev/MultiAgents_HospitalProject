"""
BDI Agent — one instance per agent in the level.

Lifecycle (called by Manager each turn):
    1. agent.sense(joint_state, t)        — update beliefs
    2. agent.plan_is_sound(joint_state, t) — check if plan is still valid
    3. agent.replan(joint_state, t)        — if not sound, replan
    4. agent.next_action()                 — consume next action from plan

Conflict recovery (called by Manager after resolution):
    agent.step_back()  — undo the advance of _plan_index if our action was overridden
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from searchclient.action import Action
from searchclient.Task import Task

if TYPE_CHECKING:
    from searchclient.heuristics import DistanceMap
    from searchclient.level_parser import LevelProfile
    from searchclient.state import State


class Agent:

    def __init__(
        self,
        agent_id: int,
        dist_map: "DistanceMap",
        profile: "LevelProfile",
    ) -> None:
        self.agent_id = agent_id
        self.dist_map = dist_map
        self.profile = profile
        self.task: Task | None = None

        # Current position of this agent in the latest observed joint state.
        # Kept in sync by the Manager so other components can inspect it.
        self.agent_row: int = -1
        self.agent_col: int = -1

        # Task queue: each task is (box_r, box_c, goal_r, goal_c, box_char)
        # box_r/box_c are INITIAL positions; replan always finds current position.
        # self.tasks: list[tuple] = []

        # Optional positional goal for the agent itself (no box)
        self.agent_goal: tuple[int, int] | None = None
        self.has_reached_its_goal: bool = False
        self.awaiting_cnt: int = 0

        # Current plan and index
        self._plan: list[Action] = []
        self._plan_index: int = 0

        # Space-time constraints: (r, c, t) cells forbidden for this agent
        self.constraints: set[tuple[int, int, int]] = set()
        # Post-escape stagger: turns to wait before replanning box task after escape
        self._escape_stagger: int = 0
        # When a nav-only agent is given a temporary escape cell, the real
        # positional goal is saved here so it can be restored afterwards.
        self._pending_agent_goal: "tuple[int, int] | None" = None

    def update_position(self, row: int, col: int) -> None:
        """Update the agent's cached position."""
        self.agent_row = row
        self.agent_col = col

    @property
    def position(self) -> tuple[int, int]:
        """Return the cached (row, col) position."""
        return self.agent_row, self.agent_col

    # ------------------------------------------------------------------
    # BDI cycle
    # ------------------------------------------------------------------

    def sense(self, joint_state: "State", timestep: int) -> None:
        """
        Update beliefs from the current joint state.
        Pops completed tasks (box already at goal).
        """
        while self.tasks:
            task = self.tasks[0]
            gr, gc, char = task[2], task[3], task[4]
            if joint_state.boxes[gr][gc] == char:
                # Box arrived at goal → task done
                self.tasks.pop(0)
                self._plan = []
                self._plan_index = 0
                print(
                    f"Agent {self.agent_id}: task done, {len(self.tasks)} remaining.",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                break

    def plan_is_sound(self, joint_state: "State", timestep: int) -> bool:
        """
        Return True iff the existing plan is non-empty and not yet exhausted.
        A more sophisticated version would also check for invalidated constraints.
        """
        if not self.tasks and self.agent_goal is None:
            return True  # nothing to do — trivially sound
        return bool(self._plan) and self._plan_index < len(self._plan)

    def replan(self, joint_state: "State", timestep: int) -> bool:
        """
        Compute a new plan for the first remaining task (or agent_goal).
        Tries IW(1) first (fast), falls back to A* if IW fails.
        Returns True on success, False if no plan found.
        """
        from searchclient.planner.astar import solve
        from searchclient.planner.iw import iw_solve

        # Block other agents' current positions for a few timesteps so replanned
        # paths don't immediately step into occupied cells.
        runtime_constraints = set(self.constraints)
        num_agents = len(joint_state.agent_rows)
        for a in range(num_agents):
            if a == self.agent_id:
                continue
            or_, oc = joint_state.agent_rows[a], joint_state.agent_cols[a]
            # Only block for 2 timesteps — the plan can route THROUGH other
            # agents' cells at t+3+ (conflict resolution will force them to yield
            # if they are "done" and blocking the path).
            runtime_constraints.add((or_, oc, timestep))
            runtime_constraints.add((or_, oc, timestep + 1))
            runtime_constraints.add((or_, oc, timestep + 2))

        # ---- box task ----
        if self.tasks:
            task = self.tasks[0]
            goal_r, goal_c, box_char = task[2], task[3], task[4]

            if joint_state.boxes[goal_r][goal_c] == box_char:
                self.tasks.pop(0)
                self._plan = []
                self._plan_index = 0
                return True

            box_r, box_c = self._find_box(joint_state, box_char, goal_r, goal_c)
            if box_r is None:
                self.tasks.pop(0)
                self._plan = []
                self._plan_index = 0
                return True

            # Try IW(1) first — fast and considers other agents via constraints
            plan = iw_solve(
                state=joint_state,
                agent_id=self.agent_id,
                subgoals=[(box_r, box_c, goal_r, goal_c, box_char)],
                k=1,
                constraints=runtime_constraints,
            )

            # A* fallback: ghost state + empty constraints = no time-indexing,
            # bounded search, guaranteed termination in large levels.
            if plan is None:
                ghost = _ghost_state_for_nav(joint_state, self.agent_id)
                plan = solve(
                    state=ghost,
                    agent_id=self.agent_id,
                    goal=(box_r, box_c, goal_r, goal_c, box_char),
                    constraints=set(),
                    dist_map=self.dist_map,
                )

            if plan is None:
                # Both IW and ghost A* failed — another box is blocking the path.
                # Find it, push it one step clear, then chain the main task plan.
                plan = self._plan_with_obstacle_clear(
                    joint_state, box_r, box_c, goal_r, goal_c, box_char
                )

            if plan is None:
                print(
                    f"Agent {self.agent_id}: replan failed for box {box_char} "
                    f"({box_r},{box_c})→({goal_r},{goal_c}).",
                    file=sys.stderr,
                    flush=True,
                )
                self._plan = []
                self._plan_index = 0
                return False

            # If agent has a positional goal, also append navigation to it.
            # This prevents the agent ending up stranded after box delivery.
            if self.agent_goal is not None and len(self.tasks) == 1:
                gr, gc = self.agent_goal
                post_state = _apply_single_agent_plan(joint_state, self.agent_id, plan)
                ar = post_state.agent_rows[self.agent_id]
                ac = post_state.agent_cols[self.agent_id]
                if not (ar == gr and ac == gc):
                    ghost_post = _ghost_state_for_nav(post_state, self.agent_id)
                    nav = solve(
                        state=ghost_post,
                        agent_id=self.agent_id,
                        goal=(None, None, gr, gc),
                        constraints=set(),
                        dist_map=self.dist_map,
                    )
                    if nav is not None:
                        plan = plan + nav

            self._plan = plan
            self._plan_index = 0
            return True

        # ---- positional goal (no box) ----
        if self.agent_goal is not None:
            gr, gc = self.agent_goal
            ar = joint_state.agent_rows[self.agent_id]
            ac = joint_state.agent_cols[self.agent_id]
            if ar == gr and ac == gc:
                self.agent_goal = None
                self._plan = []
                self._plan_index = 0
                if self._pending_agent_goal is not None:
                    # Reached escape cell — now navigate to the real positional goal
                    self.agent_goal = self._pending_agent_goal
                    self._pending_agent_goal = None
                else:
                    self._escape_stagger = self.agent_id * 3
                return True

            # Use a ghost state where other agents are invisible so A* can
            # find a topological path. Other agents yield reactively via
            # conflict resolution — they won't permanently block nav.
            ghost = _ghost_state_for_nav(joint_state, self.agent_id)
            plan = solve(
                state=ghost,
                agent_id=self.agent_id,
                goal=(None, None, gr, gc),
                constraints=set(),
                dist_map=self.dist_map,
            )
            if plan is None:
                self._plan = []
                self._plan_index = 0
                return False
            self._plan = plan
            self._plan_index = 0
            return True

        return True  # nothing to plan for

    def next_action(self) -> Action:
        """Consume and return next action from the plan."""
        if self._plan_index < len(self._plan):
            action = self._plan[self._plan_index]
            self._plan_index += 1
            return action
        return Action.NoOp

    def step_back(self) -> None:
        """
        Undo the last next_action() advance.
        Called by Manager when conflict resolution overrides our action so we
        retry the same step next turn.
        """
        if self._plan_index > 0:
            self._plan_index -= 1

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def assign_tasks(self, tasks: list[tuple]) -> None:
        """
        Called by Manager after assignment.
        tasks: list of (box_r, box_c, goal_r, goal_c, char) 5-tuples.
        """
        self.tasks = list(tasks)
        self._plan = []
        self._plan_index = 0

    def is_done(self, joint_state: "State") -> bool:
        """True iff all assigned tasks are completed and no agent goal pending."""
        if self.agent_goal is not None:
            gr, gc = self.agent_goal
            if not (
                joint_state.agent_rows[self.agent_id] == gr
                and joint_state.agent_cols[self.agent_id] == gc
            ):
                return False
        return not self.tasks

    def remaining_plan(self) -> list[Action]:
        return self._plan[self._plan_index :]

    # ------------------------------------------------------------------
    # Constraint injection (by Manager CBS)
    # ------------------------------------------------------------------

    def inject_constraint(self, r: int, c: int, t: int) -> None:
        """Add a space-time constraint and invalidate the plan if affected."""
        self.constraints.add((r, c, t))
        # Invalidate plan so we replan with new constraint next turn
        self._plan = []
        self._plan_index = 0

    def clear_old_constraints(self, current_t: int) -> None:
        """Remove constraints for past timesteps to keep the set small."""
        self.constraints = {(r, c, t) for r, c, t in self.constraints if t >= current_t}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_box(
        self,
        joint_state: "State",
        box_char: str,
        goal_r: int,
        goal_c: int,
    ) -> tuple[int | None, int | None]:
        """
        Find the current (row, col) of the box with the given char that has
        NOT yet reached (goal_r, goal_c).

        If multiple candidates exist, returns the one closest to the goal
        (most likely the one originally assigned to this agent).
        """
        candidates: list[tuple[int, int, int]] = []  # (dist_to_goal, r, c)
        for r, row in enumerate(joint_state.boxes):
            for c, ch in enumerate(row):
                if ch == box_char and not (r == goal_r and c == goal_c):
                    d = abs(r - goal_r) + abs(c - goal_c)
                    candidates.append((d, r, c))

        if not candidates:
            return None, None
        candidates.sort()
        _, r, c = candidates[0]
        return r, c

    # ------------------------------------------------------------------
    # Obstacle clearing helpers
    # ------------------------------------------------------------------

    def _find_path_obstacle(
        self,
        joint_state: "State",
        from_r: int,
        from_c: int,
        to_r: int,
        to_c: int,
    ) -> "tuple[int, int, str] | None":
        """
        BFS treating walls as obstacles and boxes as transparent.
        Returns (row, col, char) of the first non-target box on the shortest
        path from (from_r,from_c) to (to_r,to_c), or None.
        """
        from collections import deque
        from searchclient.state import State as _S

        walls = _S.walls
        num_rows = len(walls)
        num_cols = len(walls[0]) if num_rows > 0 else 0

        parent: dict = {(from_r, from_c): None}
        queue: deque = deque([(from_r, from_c)])

        while queue:
            r, c = queue.popleft()
            if r == to_r and c == to_c:
                path = []
                cur = (to_r, to_c)
                while parent.get(cur) is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                for pr, pc in path:
                    if (pr, pc) == (to_r, to_c):
                        continue
                    ch = joint_state.boxes[pr][pc]
                    if ch:
                        return (pr, pc, ch)
                return None
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < num_rows
                    and 0 <= nc < num_cols
                    and not walls[nr][nc]
                    and (nr, nc) not in parent
                ):
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))
        return None

    def _find_push_dest(
        self,
        joint_state: "State",
        blk_r: int,
        blk_c: int,
    ) -> "tuple[int, int] | None":
        """
        Find a (dest_r, dest_c) to push the box at (blk_r, blk_c) to:
        dest must be free and the push-from cell (opposite direction) must be free.
        """
        from searchclient.state import State as _S

        walls = _S.walls
        num_rows = len(walls)
        num_cols = len(walls[0]) if num_rows > 0 else 0

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            dest_r, dest_c = blk_r + dr, blk_c + dc
            frm_r, frm_c = blk_r - dr, blk_c - dc
            if not (0 <= dest_r < num_rows and 0 <= dest_c < num_cols):
                continue
            if walls[dest_r][dest_c] or joint_state.boxes[dest_r][dest_c]:
                continue
            if not (0 <= frm_r < num_rows and 0 <= frm_c < num_cols):
                continue
            if walls[frm_r][frm_c] or joint_state.boxes[frm_r][frm_c]:
                continue
            return (dest_r, dest_c)
        return None

    def _plan_with_obstacle_clear(
        self,
        joint_state: "State",
        box_r: int,
        box_c: int,
        goal_r: int,
        goal_c: int,
        box_char: str,
    ) -> "list[Action] | None":
        """
        Iteratively clears blocking boxes until a full path is found or the
        clearing limit (5 boxes) is reached.  Returns the concatenated plan
        (clears + main task), or just the clearing prefix if the main task
        still can't be planned (agent makes progress and will retry).
        Returns None only if not even the first obstacle can be cleared.
        """
        from searchclient.planner.astar import solve
        from searchclient.color import Color
        from searchclient.state import State as _S

        MAX_CLEARS = 5
        # Work on a ghost state so other agents don't block planning
        current = _ghost_state_for_nav(joint_state, self.agent_id)
        accumulated: list[Action] = []

        for iteration in range(MAX_CLEARS):
            ar = current.agent_rows[self.agent_id]
            ac = current.agent_cols[self.agent_id]

            cur_box_r, cur_box_c = self._find_box(current, box_char, goal_r, goal_c)
            if cur_box_r is None:
                break  # target box already at goal

            # Try main task first — maybe path is now clear
            main_plan = solve(
                state=current,
                agent_id=self.agent_id,
                goal=(cur_box_r, cur_box_c, goal_r, goal_c, box_char),
                constraints=set(),
                dist_map=self.dist_map,
            )
            if main_plan is not None:
                return accumulated + main_plan

            # Find the next blocking obstacle
            obstacle = self._find_path_obstacle(current, ar, ac, cur_box_r, cur_box_c)
            if obstacle is None:
                break  # no geometric obstacle found — A* failure has another cause

            o_r, o_c, o_char = obstacle
            o_color = _S.box_colors[ord(o_char) - ord("A")]
            if not Color.compatible(_S.agent_colors[self.agent_id], o_color):
                break  # can't push this box

            dest = self._find_push_dest(current, o_r, o_c)
            if dest is None:
                break  # no free cell to push into
            d_r, d_c = dest

            clear_plan = solve(
                state=current,
                agent_id=self.agent_id,
                goal=(o_r, o_c, d_r, d_c, o_char),
                constraints=set(),
                dist_map=self.dist_map,
            )
            if clear_plan is None:
                break  # can't reach the obstacle

            print(
                f"Agent {self.agent_id}: clearing [{iteration+1}] {o_char} "
                f"({o_r},{o_c})→({d_r},{d_c}) to reach {box_char}.",
                file=sys.stderr,
                flush=True,
            )
            accumulated.extend(clear_plan)
            current = _apply_single_agent_plan(current, self.agent_id, clear_plan)

        return accumulated if accumulated else None


def _apply_single_agent_plan(state: "State", agent_id: int, plan: list) -> "State":
    """Simulate applying a single-agent plan; other agents stay put."""
    from searchclient.action import Action

    current = state
    num = len(state.agent_rows)
    for action in plan:
        joint = [Action.NoOp] * num
        joint[agent_id] = action
        current = current.result(joint)
    return current


def _ghost_state_for_nav(state: "State", agent_id: int) -> "State":
    """Return a state where other agents are moved to (0,0) — a guaranteed wall.
    Used for nav-only A* so other agents don't block the topological path;
    conflict resolution makes them yield at runtime."""
    from searchclient.state import State

    fake_rows = [
        0 if a != agent_id else state.agent_rows[a]
        for a in range(len(state.agent_rows))
    ]
    fake_cols = [
        0 if a != agent_id else state.agent_cols[a]
        for a in range(len(state.agent_cols))
    ]
    return State(fake_rows, fake_cols, state.boxes)
