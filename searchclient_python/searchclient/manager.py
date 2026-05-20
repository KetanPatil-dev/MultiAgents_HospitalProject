"""
Manager — Coordinated multi-agent planning with task-based approach.

Architecture:
  - setup()           : initialize agents, build task lists, prearrange HCA*
  - get_joint_action(): per-timestep coordination (BDI cycle)
  - is_done()         : termination check
"""

from __future__ import annotations

import heapq
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import TypedDict

from searchclient.action import Action, ActionType
from searchclient.agent import Agent
from searchclient.color import Color
from searchclient.heuristics import DistanceMap
from searchclient.level_parser import LevelProfile
from searchclient.state import State
from searchclient.planner.astar import solve
from searchclient.Task import Task
from searchclient.task_swap import TaskSwap


# Joint A* state-space caps (scaled by agent count)
_JOINT_MAX_CLOSED_2  = 1_000_000   # 2-agent levels
_JOINT_MAX_CLOSED_3  = 500_000     # 3-agent levels
_JOINT_MAX_CLOSED    = 300_000     # 4+ agent levels (rarely used)
# Max future timesteps for permanent constraints in HCA* pass 2
_MAX_FUTURE = 600


@dataclass(order=True)
class _JNode:
    """Joint A* node — searches in the joint state space across all agents."""
    f:      int
    g:      int               = field(compare=False)
    state:  State             = field(compare=False)
    parent: "_JNode | None"   = field(compare=False, default=None)


class ColorTasksTypedDict(TypedDict):
    future_box_tasks: deque[Task]
    future_agent_tasks: deque[Task]
    solved_tasks: deque[Task]


def _apply_agent_plan(state: State, agent_id: int, plan: list[Action]) -> State:
    """Apply a single-agent plan to the joint state (other agents stay put)."""
    current = state
    num_agents = len(state.agent_rows)
    for action in plan:
        joint = [Action.NoOp] * num_agents
        joint[agent_id] = action
        current = current.result(joint)
    return current


class Manager:

    def __init__(self) -> None:
        self.agents: list[Agent] = []
        self.profile: LevelProfile | None = None
        self.dist_map: DistanceMap | None = None
        self.timestep: int = 0

        # Task lists by agent color: {Color: {'future': [...], 'current': [...]}}
        self.color_tasks: dict[Color | None, ColorTasksTypedDict] = {}
        self.agents_awaiting_other_agent: dict[int, int | None] = {}
        # Per-agent count of consecutive turns with no plan + an open task.
        # Used by stuck-agent recovery to inject a single-turn nudge.
        self.agents_no_plan_cnt: dict[int, int] = {}
        # Per-agent rotation counter for the recovery nudge. Incremented every
        # time _pick_recovery_move fires so consecutive nudges for the same
        # agent rotate through cardinal directions deterministically.
        self.agents_nudge_rotation: dict[int, int] = {}
        # Per-agent rotation counter for the escape-cell selection. Used by
        # Plan-B after negotiation cycle refusal so consecutive triggers
        # don't keep sending the agent to the same cell.
        self.agents_escape_rotation: dict[int, int] = {}
        # Rolling history of joint-state hashes for cycle detection.
        # When the same hash appears N+ times in the window, we're in a deadlock.
        self._state_history: deque = deque(maxlen=30)
        self._deadlock_resolved_at: int = -1  # last timestep we triggered recovery
        # Rolling history of invalid-agent sets. Used to catch oscillation
        # (alternating conflict groups) much earlier than full joint-state
        # cycling — typically within 6-10 turns instead of 40-90.
        self._invalid_agent_history: deque = deque(maxlen=20)
        self._cbs_recovered_at: int = -1  # last timestep we ran runtime CBS recovery
        # Chokepoint goals: {(goal_r, goal_c): set[agent_ids_that_need_to_pass_through]}
        # Delivering a box here is deferred until those agents reach their goals.
        self.chokepoint_goals: dict[tuple[int, int], set[int]] = {}
        # Negotiation log: every cross-agent task swap is recorded here.
        # Used for cycle detection and offline analysis / visualisation of the
        # decision chain.
        self.negotiations: list[TaskSwap] = []

    # ------------------------------------------------------------------
    # Setup (once after parsing)
    # ------------------------------------------------------------------

    def setup(self, initial_state: State, profile: LevelProfile) -> None:
        """
        Initialize manager for a new level.

        Step 1: Preplanning
        - Build task lists for each agent and agent-color group
        - Find all final positions (box goals and agent goals)
        - Assign initial current tasks to each agent
        - Perform HCA* preplanning with heuristics
        """
        self.profile = profile
        self.dist_map = DistanceMap.from_state(initial_state)

        # Create agents
        self.agents = [
            Agent(i, self.dist_map, profile) for i in range(profile.num_agents)
        ]
        self.agents_awaiting_other_agent: dict[int, int | None] = {
            agent.agent_id: None for agent in self.agents
        }
        self.agents_no_plan_cnt = {agent.agent_id: 0 for agent in self.agents}
        self.agents_nudge_rotation = {agent.agent_id: 0 for agent in self.agents}
        self.agents_escape_rotation = {agent.agent_id: 0 for agent in self.agents}
        self._sync_agent_positions(initial_state)

        # Initialize task lists for each agent color
        for agent in self.agents:
            agent_color = State.agent_colors[agent.agent_id]
            if agent_color not in self.color_tasks:
                self.color_tasks[agent_color] = ColorTasksTypedDict(
                    future_box_tasks=deque(),
                    future_agent_tasks=deque(),
                    solved_tasks=deque(),
                )
        # Also initialize for any box color that doesn't match an agent color —
        # otherwise _build_box_goal_tasks crashes with KeyError.
        for _, _, ch in profile.box_goals:
            box_color = State.box_colors[ord(ch) - ord("A")] if ch.isupper() else None
            if box_color is not None and box_color not in self.color_tasks:
                self.color_tasks[box_color] = ColorTasksTypedDict(
                    future_box_tasks=deque(),
                    future_agent_tasks=deque(),
                    solved_tasks=deque(),
                )

        # Step 1a: Find all box goals and create move_box tasks for each agent color
        print("Building box goal tasks...", file=sys.stderr, flush=True)
        self._build_box_goal_tasks()

        # Step 1b: Find all agent goals and create move_agent tasks
        print("Building agent goal tasks...", file=sys.stderr, flush=True)
        self._build_agent_goal_tasks(initial_state)

        # Size-based gating: very large levels can take minutes to preprocess.
        n_cells = profile.num_rows * profile.num_cols
        big_level = n_cells > 2000 and profile.num_agents > 8
        very_big_level = n_cells > 3000 or profile.num_agents > 10

        # Initialize joint_plan here so all early-return paths see a defined attr.
        self._joint_plan: list[list[Action]] = []

        # ──────────────────────────────────────────────────────────────────
        # CBS-first pipeline: the reactive pipeline below runs CBS (cheap,
        # solves most cases in <1s) before HCA* (expensive, often burns 10s+
        # exhausting MAX_CLOSED when its hard priority constraints make a
        # solvable problem look infeasible). Previously we attempted an
        # HCA*-first fast path here; removed because on WardRush it wasted
        # ~17s proving HCA* couldn't handle agents 1/2/3 only to then have
        # CBS solve the same problem in 1 node.
        # ──────────────────────────────────────────────────────────────────

        # Step 1c: Pre-inject corridor-clearing tasks for deco boxes that block passages
        if not very_big_level:
            self._inject_corridor_clearing_tasks(initial_state)
        else:
            print("  Skipping corridor pre-clearing (very large level).", file=sys.stderr, flush=True)

        # Step 1d: Compute box-goal chokepoints — box deliveries that would
        # block another agent's path to its agent-goal.
        if not big_level:
            self._compute_chokepoints(initial_state)
        else:
            print("  Skipping chokepoint analysis (large level).", file=sys.stderr, flush=True)

        # Step 1e: CBS preplanning — find a conflict-free joint plan for the
        # first task of each agent. Subsequent tasks are handled reactively.
        if not very_big_level:
            print("Running CBS preplanning...", file=sys.stderr, flush=True)
            self._cbs_preplan(initial_state)
        else:
            print("  Skipping CBS preplan (very large level).", file=sys.stderr, flush=True)

        # Step 1f: Hungarian assignment + HCA* preplan (ported from felix).
        # Populates agent.tasks per-agent and tries to plan full task sequences
        # cooperatively. Only used if our reactive system needs the help.
        if not very_big_level:
            try:
                self._hungarian_and_hca(initial_state)
            except Exception as e:
                import traceback
                print(f"  HCA* preplan errored ({e}); falling back to reactive.", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)

        # Step 1g: Joint A* — for small levels, try the optimal joint
        # state-space search. Two tiers:
        #   • ≤2 agents, ≤450 cells, ≤4 boxes — generous budget (5s)
        #   • 3 agents, ≤200 cells, ≤2 boxes — tight budget (3s)
        # The 3-agent tier unlocks a small set of micro-levels where joint
        # planning is tractable but the reactive system over-coordinates.
        n_boxes = len(profile.real_boxes)
        joint_eligible = False
        budget = 5.0
        if profile.num_agents <= 2 and n_cells <= 450 and n_boxes <= 4:
            joint_eligible = True
            budget = 5.0
        elif profile.num_agents == 3 and n_cells <= 200 and n_boxes <= 2:
            joint_eligible = True
            budget = 3.0

        if joint_eligible:
            print(
                f"Trying Joint A* ({profile.num_agents} agents, "
                f"{n_cells} cells, {n_boxes} boxes, budget {budget}s).",
                file=sys.stderr, flush=True,
            )
            joint_actions = self._joint_astar_plan(initial_state, time_budget_s=budget)
            if joint_actions is not None:
                self._joint_plan = joint_actions
                print(
                    f"Joint A* found plan of length {len(joint_actions)}.",
                    file=sys.stderr, flush=True,
                )

    def _hungarian_and_hca(self, initial_state: State) -> None:
        """Run Hungarian box assignment + subgoal ordering, populate agent.tasks
        for each agent, then run HCA* two-pass preplan."""
        from searchclient.planner.hungarian import assign, agent_goals_assign, subgoal_order

        if self.profile is None or self.dist_map is None:
            return

        box_assignment = assign(
            agents=self.agents,
            boxes=self.profile.real_boxes,
            goals=self.profile.box_goals,
            dist_map=self.dist_map,
        )

        agent_rows0 = initial_state.agent_rows
        agent_cols0 = initial_state.agent_cols
        any_plan_existed = any(a._plan for a in self.agents)

        for agent in self.agents:
            tasks = box_assignment.get(agent.agent_id, [])
            if len(tasks) > 1:
                tasks = subgoal_order(
                    tasks, self.dist_map,
                    start_r=agent_rows0[agent.agent_id],
                    start_c=agent_cols0[agent.agent_id],
                )
            agent.tasks = tasks

        ag_goals = agent_goals_assign(self.agents, self.profile.agent_goals)
        for agent in self.agents:
            if agent.agent_goal is None:
                agent.agent_goal = ag_goals.get(agent.agent_id)

        # Run HCA* only on agents that:
        #   (a) don't already have a CBS plan, AND
        #   (b) have NO pending tasks in their color queue (otherwise HCA* would
        #       skip our chokepoint/corridor logic by assigning a stale plan).
        def has_pending_color_tasks(aid: int) -> bool:
            ac = State.agent_colors[aid]
            bucket = self.color_tasks.get(ac)
            if bucket is None:
                return False
            return bool(bucket["future_box_tasks"]) or bool(bucket["future_agent_tasks"])

        unplanned = [
            a for a in self.agents
            if not a._plan
            and (getattr(a, "tasks", None) or a.agent_goal is not None)
            and not has_pending_color_tasks(a.agent_id)
        ]
        if not unplanned:
            return

        print(
            f"Running HCA* preplan for {len(unplanned)} agent(s)...",
            file=sys.stderr, flush=True,
        )

        # Register planned agents' trajectories as constraints for unplanned ones
        for agent in self.agents:
            if agent._plan:
                ar = initial_state.agent_rows[agent.agent_id]
                ac = initial_state.agent_cols[agent.agent_id]
                box_r, box_c = -1, -1
                if getattr(agent, "tasks", None):
                    t0 = agent.tasks[0]
                    if len(t0) >= 5:
                        br, bc = agent._find_box(initial_state, t0[4], t0[2], t0[3])
                        if br is not None:
                            box_r, box_c = br, bc
                for other in unplanned:
                    self._register_agent_path_one(ar, ac, box_r, box_c, agent._plan, other)

        for agent in unplanned:
            full_plan, success = self._plan_all_tasks(agent, initial_state, 0)
            if success and full_plan:
                agent._plan = full_plan
                agent._plan_index = 0
                print(
                    f"  HCA* recovered Agent {agent.agent_id} (plan length {len(full_plan)}).",
                    file=sys.stderr, flush=True,
                )

    def _compute_chokepoints(self, initial_state: State) -> None:
        """
        For each box goal position, determine which agents' positional goals
        become unreachable if that cell is treated as a permanent wall.
        Box deliveries to such cells will be deferred until those agents
        complete their positional goals.
        """
        from collections import deque as _dq
        assert self.profile is not None
        num_rows = len(State.walls)
        num_cols = len(State.walls[0]) if num_rows > 0 else 0

        def bfs_reachable(
            start: tuple[int, int],
            blocked_cells: set[tuple[int, int]] | tuple[int, int] = (-1, -1),
        ) -> set[tuple[int, int]]:
            # Accept either a single (r, c) or a set of cells to treat as walls.
            if isinstance(blocked_cells, set):
                blocked = blocked_cells
            else:
                blocked = {blocked_cells}
            visited: set[tuple[int, int]] = {start}
            q: _dq = _dq([start])
            while q:
                r, c = q.popleft()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in visited:
                        continue
                    if not (0 <= nr < num_rows and 0 <= nc < num_cols):
                        continue
                    if State.walls[nr][nc]:
                        continue
                    if (nr, nc) in blocked:
                        continue
                    visited.add((nr, nc))
                    q.append((nr, nc))
            return visited

        # All other agents' positional goals — they will end up sitting on
        # these cells, so during chokepoint analysis they behave as walls
        # for any agent that has NOT yet reached its own positional goal.
        all_agent_goals: dict[int, tuple[int, int]] = {
            aid: (ag_r, ag_c)
            for ag_r, ag_c, aid in self.profile.agent_goals
        }

        # For each box goal cell, simulate it as a wall and check agent reachability
        for goal_r, goal_c, _ in self.profile.box_goals:
            for ag_r, ag_c, aid in self.profile.agent_goals:
                if (ag_r, ag_c) == (goal_r, goal_c):
                    continue
                start = (initial_state.agent_rows[aid], initial_state.agent_cols[aid])
                # Treat OTHER agents' positional goals as walls too: by the time
                # the affected agent attempts its positional task, the other
                # agents may already be parked at their goals and physically
                # block the topology. Missing this causes pocket-traps like
                # TeamAgent where agent 2 stranded by agent 1 sitting at (1,2).
                other_agent_goal_blockers: set[tuple[int, int]] = {
                    pos for other_aid, pos in all_agent_goals.items()
                    if other_aid != aid and pos != (goal_r, goal_c)
                }
                reachable_without = bfs_reachable(
                    start, blocked_cells=other_agent_goal_blockers
                )
                if (ag_r, ag_c) not in reachable_without:
                    continue  # agent goal unreachable anyway, skip
                blocked_with = other_agent_goal_blockers | {(goal_r, goal_c)}
                reachable_with = bfs_reachable(start, blocked_cells=blocked_with)
                if (ag_r, ag_c) not in reachable_with:
                    self.chokepoint_goals.setdefault((goal_r, goal_c), set()).add(aid)
                    print(
                        f"  Chokepoint: box goal ({goal_r},{goal_c}) blocks Agent {aid}'s "
                        f"path to ({ag_r},{ag_c}). Will defer.",
                        file=sys.stderr, flush=True,
                    )

        # Box-blocks-box chokepoint: delivering box X to its goal cell might
        # disconnect another box Y from its goal path. We use a sentinel
        # negative agent_id (-(box_index + 1)) to track these box-deliveries
        # that must happen first before this chokepoint is "open."
        self.box_chokepoint_blockers: dict[tuple[int, int], set[tuple[int, int, str]]] = {}
        # For each box goal cell A, check: does treating A as wall make
        # any OTHER box's start→goal path unreachable?
        for goal_a_r, goal_a_c, _ in self.profile.box_goals:
            for box_b in self.profile.real_boxes:
                box_b_r, box_b_c, box_b_char = box_b
                # Find the matching goal for this box
                target_b = None
                for gbr, gbc, gbch in self.profile.box_goals:
                    if gbch == box_b_char and (gbr, gbc) != (goal_a_r, goal_a_c):
                        target_b = (gbr, gbc)
                        break
                if target_b is None:
                    continue
                # Box must be able to reach its goal in the open map
                reachable_open = bfs_reachable((box_b_r, box_b_c), (-1, -1))
                if target_b not in reachable_open:
                    continue
                # If we close goal_a, can box B still reach its goal?
                reachable_closed = bfs_reachable((box_b_r, box_b_c), (goal_a_r, goal_a_c))
                if target_b not in reachable_closed:
                    self.box_chokepoint_blockers.setdefault(
                        (goal_a_r, goal_a_c), set()
                    ).add((box_b_r, box_b_c, box_b_char))
                    print(
                        f"  Box-chokepoint: delivering to ({goal_a_r},{goal_a_c}) blocks "
                        f"box {box_b_char} delivery. Will defer.",
                        file=sys.stderr, flush=True,
                    )

    def _chokepoint_blockers_pending(self, task: Task) -> bool:
        """Return True if delivering this box-task would block either:
          (a) an agent whose positional goal is not yet satisfied, OR
          (b) another box whose delivery is not yet completed.
        """
        if task.task_type != "move_box":
            return False

        # (a) Agent-blocking chokepoint
        if task.goal_pos in self.chokepoint_goals:
            blocked_agents = self.chokepoint_goals[task.goal_pos]
            for aid in blocked_agents:
                ag = self.agents[aid]
                if not ag.has_reached_its_goal:
                    for gr, gc, gid in (self.profile.agent_goals if self.profile else []):
                        if gid == aid and (ag.agent_row, ag.agent_col) != (gr, gc):
                            return True

        # (b) Box-blocking-box chokepoint: delivering this task would close
        # a passage another box's delivery path depends on. Defer until that
        # blocker box is delivered.
        box_blockers = getattr(self, "box_chokepoint_blockers", None)
        if box_blockers and task.goal_pos in box_blockers:
            for _b_r, _b_c, b_char in box_blockers[task.goal_pos]:
                if task.box_char == b_char:
                    continue  # ignore self
                if self._box_delivery_pending(b_char):
                    return True

        return False

    def _box_delivery_pending(self, box_char: str) -> bool:
        """True if any move_box task for `box_char` is still pending (in a
        future queue or currently assigned to an agent) — i.e. not yet solved."""
        # Currently assigned to an agent?
        for agent in self.agents:
            t = agent.task
            if t is not None and t.task_type == "move_box" and t.box_char == box_char:
                return True
        # In any color's future_box_tasks?
        for bucket in self.color_tasks.values():
            for t in bucket["future_box_tasks"]:
                if t.task_type == "move_box" and t.box_char == box_char:
                    return True
        return False

    def _cbs_preplan(self, initial_state: State) -> None:
        """
        Run anytime CBS over all agents' first tasks. Pops one task per agent
        from the future queues, runs CBS with a progressively-expanding node
        cap under a wall-clock budget, and stores any solution in each agent's
        `_plan`. Falls back silently to reactive planning if CBS fails.
        """
        from searchclient.planner.cbs import anytime_cbs_solve

        agent_goals: dict[int, tuple] = {}
        assigned_tasks: dict[int, Task] = {}

        for agent in self.agents:
            task = self._pop_next_task_for_agent(agent.agent_id)
            if task is None:
                continue
            assigned_tasks[agent.agent_id] = task
            agent.task = task
            self._sync_agent_task_state(agent.agent_id, task)
            agent_goals[agent.agent_id] = self._convert_task_to_goal_tuple(task)

        if not agent_goals:
            print("  CBS: no tasks to plan.", file=sys.stderr, flush=True)
            return

        # CBS budget. Kept tight so setup never burns the whole client
        # time-window on a level CBS can't solve; reactive planning will
        # handle anything CBS misses.
        plans = anytime_cbs_solve(
            state=initial_state,
            agent_goals=agent_goals,
            dist_map=self.dist_map,
            time_budget_s=5.0,
            cap_schedule=(50, 200, 800, 3000),
        )

        if plans is None:
            print(
                "  CBS preplan failed — falling back to reactive planning.",
                file=sys.stderr, flush=True,
            )
            return

        for aid, plan in plans.items():
            self.agents[aid]._plan = plan
            self.agents[aid]._plan_index = 0
            print(
                f"  CBS plan for Agent {aid}: length {len(plan)}",
                file=sys.stderr, flush=True,
            )

    def _build_box_goal_tasks(self) -> None:
        """
        Find all box goals in the profile and create move_box tasks.
        Add to color group future tasks.
        """
        assert self.profile is not None

        assigned_boxes: set[tuple[int, int]] = set()

        for goal_r, goal_c, goal_char in self.profile.box_goals:
            task = Task(
                task_type="move_box",
                object_pos=(
                    goal_r,
                    goal_c,
                ),  # initial box position (same as goal for now)
                goal_pos=(goal_r, goal_c),
                box_char=goal_char,
                crucial=True,
            )

            # Find the actual box and update object_pos
            # NOTE: this order here might be IMPORTANT !!!
            for br, bc, box_char in self.profile.real_boxes:
                if box_char == goal_char and (br, bc) not in assigned_boxes:
                    task.object_pos = (br, bc)
                    assigned_boxes.add((br, bc))
                    break

            # Get color of the box
            box_color = (
                State.box_colors[ord(goal_char) - ord("A")]
                if goal_char.isupper()
                else None
            )

            # Add to color group future tasks
            # if box_color not in self.color_tasks:
            #     self.color_tasks[box_color] = {
            #         "future": deque(),
            #         "current": deque(),
            #     }
            self.color_tasks[box_color]["future_box_tasks"].append(task)

            print(
                f"  Box task: {goal_char} {task.object_pos} → {task.goal_pos}",
                file=sys.stderr,
                flush=True,
            )

        print(
            f" Total box tasks created: {len(assigned_boxes)}",
            file=sys.stderr,
            flush=True,
        )

    def _build_agent_goal_tasks(self, initial_state: State) -> None:
        """
        Find all agent goals in the profile and create move_agent tasks.
        Add to respective agent future tasks.
        """
        assert self.profile is not None

        for agent_goal_r, agent_goal_c, agent_id in self.profile.agent_goals:
            if 0 <= agent_id < len(self.agents):
                task = Task(
                    task_type="move_agent",
                    object_pos=(
                        initial_state.agent_rows[agent_id],
                        initial_state.agent_cols[agent_id],
                    ),
                    goal_pos=(agent_goal_r, agent_goal_c),
                    crucial=True,
                )
                # self.color_tasks[agent_id]["future_agent_tasks"].append(task)
                self.color_tasks[State.agent_colors[agent_id]][
                    "future_agent_tasks"
                ].append(task)

                print(
                    f"  Agent {agent_id} goal: {task.object_pos} → {task.goal_pos}",
                    file=sys.stderr,
                    flush=True,
                )

    def _inject_corridor_clearing_tasks(self, initial_state: State) -> None:
        """
        For each deco box blocking a corridor (only 1 free non-wall neighbor),
        inject a clearing task at the FRONT of the matching color group's queue.
        Also inject tasks for any real boxes that block the deco box's only exit.
        """
        assert self.profile is not None
        num_rows = len(State.walls)
        num_cols = len(State.walls[0]) if num_rows > 0 else 0

        for obs_r, obs_c, obs_char in self.profile.deco_boxes:
            obs_color = State.box_colors[ord(obs_char) - ord("A")]
            clear_goal = self._find_obs_clear_goal(obs_r, obs_c, initial_state)
            if clear_goal == (obs_r, obs_c):
                continue  # no free neighbor, skip

            def _inject_blocker(br: int, bc: int, label: str) -> None:
                blocker_char = initial_state.boxes[br][bc]
                if not blocker_char:
                    return
                blocker_color = State.box_colors[ord(blocker_char) - ord("A")]
                blocker_goal = self._find_obs_clear_goal(br, bc, initial_state)
                if blocker_goal == (br, bc):
                    return
                pre_task = Task(
                    task_type="move_box",
                    object_pos=(br, bc),
                    goal_pos=blocker_goal,
                    box_char=blocker_char,
                    crucial=False,
                )
                if blocker_color in self.color_tasks:
                    self.color_tasks[blocker_color]["future_box_tasks"].appendleft(pre_task)
                    print(
                        f"  Corridor {label}: {blocker_char} ({br},{bc}) → {blocker_goal}",
                        file=sys.stderr, flush=True,
                    )

            # Determine FIRST-STEP push direction (clear_goal may be multi-step away)
            if clear_goal[0] != obs_r:
                first_dr = 1 if clear_goal[0] > obs_r else -1
                first_dc = 0
            else:
                first_dr = 0
                first_dc = 1 if clear_goal[1] > obs_c else -1

            # First step destination of the push — if blocked by a box, inject pre-clear
            first_dest_r = obs_r + first_dr
            first_dest_c = obs_c + first_dc
            if (
                0 <= first_dest_r < num_rows and 0 <= first_dest_c < num_cols
                and initial_state.boxes[first_dest_r][first_dest_c] != ""
            ):
                _inject_blocker(first_dest_r, first_dest_c, "path-pre-clear")

            # Initial push-from cell — if blocked by a box, inject pre-clear
            push_from_r = obs_r - first_dr
            push_from_c = obs_c - first_dc
            if (
                0 <= push_from_r < num_rows and 0 <= push_from_c < num_cols
                and not State.walls[push_from_r][push_from_c]
                and initial_state.boxes[push_from_r][push_from_c] != ""
            ):
                _inject_blocker(push_from_r, push_from_c, "pushfrom-pre-clear")

            # Inject the deco box clearing task at front of queue
            clear_task = Task(
                task_type="move_box",
                object_pos=(obs_r, obs_c),
                goal_pos=clear_goal,
                box_char=obs_char,
                crucial=False,
            )
            if obs_color in self.color_tasks:
                self.color_tasks[obs_color]["future_box_tasks"].appendleft(clear_task)
                print(
                    f"  Corridor clear injected: {obs_char} ({obs_r},{obs_c}) → {clear_goal}",
                    file=sys.stderr, flush=True,
                )

    def _sync_agent_task_state(self, agent_id: int, task: Task) -> None:
        """
        Keep the Agent's internal task/goal fields aligned with the manager
        task — `_preplan_for_agent` reads these fields when computing routes.
        """
        agent = self.agents[agent_id]

        if task.task_type == "move_box":
            assert task.box_char is not None
            agent.tasks = [
                (
                    task.object_pos[0],
                    task.object_pos[1],
                    task.goal_pos[0],
                    task.goal_pos[1],
                    task.box_char,
                )
            ]
            agent.agent_goal = None
        else:
            agent.tasks = []
            agent.agent_goal = task.goal_pos

    def _build_runtime_constraints(
        self, agent_id: int, timestep: int, horizon: int = 5
    ) -> set[tuple[int, int, int]]:
        """
        Build temporary space-time constraints for replanning.

        Agents currently waiting on another agent are treated as stationary
        obstacles for a short horizon so other agents do not immediately plan
        through their cells.
        """
        runtime_constraints = set(self.agents[agent_id].constraints)

        # Translate the agent's absolute-time constraints into A*-search-time.
        # An abs_constraint at global T becomes a local constraint at search
        # step (T - self.timestep). Drop entries that are already in the past.
        for (r, c, T) in self.agents[agent_id].abs_constraints:
            local_t = T - self.timestep
            if local_t >= 0:
                runtime_constraints.add((r, c, local_t))

        for other_agent_id, waiting_for in self.agents_awaiting_other_agent.items():
            if other_agent_id == agent_id or waiting_for is None:
                continue

            other_agent = self.agents[other_agent_id]
            for dt in range(horizon):
                runtime_constraints.add(
                    (
                        other_agent.agent_row,
                        other_agent.agent_col,
                        timestep + dt,
                    )
                )

        return runtime_constraints

    def _task_is_solved(self, joint_state: State, agent_id: int, task: Task) -> bool:
        """Return True if the task has already been completed in the joint state."""
        if task.task_type == "move_box":
            if task.box_char is None:
                return False
            gr, gc = task.goal_pos
            return joint_state.boxes[gr][gc] == task.box_char

        ar = joint_state.agent_rows[agent_id]
        ac = joint_state.agent_cols[agent_id]
        return (ar, ac) == task.goal_pos

    def _pop_next_task_for_agent(
        self, agent_id: int, joint_state: "State | None" = None
    ) -> Task | None:
        """
        Pull the next task for an agent, preferring the agent's color-group
        box tasks first and the agent's positional task last.

        If `joint_state` is provided, stale tasks are dropped: a box task is
        stale if its goal cell is already satisfied AND the source box is
        already at one of its proper goals. This prevents the queue's stale
        obstacle-clear tasks (pushed by the swap mechanism) from undoing
        legitimately-completed deliveries after all real boxes are delivered.
        """
        agent_color = State.agent_colors[agent_id]
        color_bucket = self.color_tasks.get(agent_color)

        def is_stale_box_task(t: Task) -> bool:
            if joint_state is None or t.task_type != "move_box" or t.box_char is None:
                return False
            ch = t.box_char
            # Stale if the goal cell isn't even a goal for this char anymore
            # (e.g. swap-fabricated bogus goal_pos).
            gr, gc = t.goal_pos
            if State.goals[gr][gc] != ch:
                return True
            # Stale if there's no undelivered same-char box left in the world.
            for r in range(len(joint_state.boxes)):
                for c in range(len(joint_state.boxes[r])):
                    if joint_state.boxes[r][c] == ch and State.goals[r][c] != ch:
                        return False  # at least one undelivered box of this char
            return True  # all same-char boxes are at goals — task is stale

        def is_unreachable_by_agent(t: Task) -> bool:
            """True if the agent physically can't reach the box's current
            position (walls separate them). Catches Hungarian/CBS assignments
            of boxes in isolated rooms to agents that can't get there."""
            if joint_state is None or t.task_type != "move_box":
                return False
            ar = joint_state.agent_rows[agent_id]
            ac = joint_state.agent_cols[agent_id]
            br, bc = t.object_pos
            # BFS through grid cells (walls block, boxes are transparent for
            # this reachability check — we'd push them anyway).
            walls = State.walls
            num_rows = len(walls)
            num_cols = len(walls[0]) if walls else 0
            visited = {(ar, ac)}
            q = deque([(ar, ac)])
            while q:
                r, c = q.popleft()
                if (r, c) == (br, bc):
                    return False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in visited:
                        continue
                    if not (0 <= nr < num_rows and 0 <= nc < num_cols):
                        continue
                    if walls[nr][nc]:
                        continue
                    visited.add((nr, nc))
                    q.append((nr, nc))
            return True  # box not reachable

        if color_bucket is not None:
            if color_bucket["future_box_tasks"]:
                # Find the first non-deferred non-stale REACHABLE task.
                tasks_queue = color_bucket["future_box_tasks"]
                while tasks_queue and is_stale_box_task(tasks_queue[0]):
                    dropped = tasks_queue.popleft()
                    print(
                        f"  Dropped stale box task: {dropped.box_char} "
                        f"{dropped.object_pos} → {dropped.goal_pos}.",
                        file=sys.stderr, flush=True,
                    )
                idx_to_pop = None
                for i, t in enumerate(tasks_queue):
                    if is_stale_box_task(t):
                        continue
                    if is_unreachable_by_agent(t):
                        continue  # leave in queue for a reachable agent
                    if not self._chokepoint_blockers_pending(t):
                        idx_to_pop = i
                        break
                if idx_to_pop is not None:
                    if idx_to_pop == 0:
                        task = tasks_queue.popleft()
                    else:
                        task = tasks_queue[idx_to_pop]
                        del tasks_queue[idx_to_pop]
                    return task
                else:
                    # All box tasks are chokepoint-deferred OR unreachable —
                    # skip to agent goals
                    pass

            if color_bucket["future_agent_tasks"]:
                for idx, task in enumerate(
                    self.color_tasks[agent_color]["solved_tasks"]
                ):
                    if task.task_type == "move_agent" and task.goal_pos == (
                        self.agents[agent_id].agent_row,
                        self.agents[agent_id].agent_col,
                    ):
                        # this agent has already reached its goal, move on to next task without assigning this one
                        print(
                            f"  Agent {agent_id} has already reached its agent goal at {task.goal_pos}, skipping assignment of next agent goal task.",
                            file=sys.stderr,
                            flush=True,
                        )
                        return None

                task = color_bucket["future_agent_tasks"].popleft()
                task.object_pos = (
                    self.agents[agent_id].agent_row,
                    self.agents[agent_id].agent_col,
                )
                return task
        return None

    def _maybe_advance_completed_task_or_preplan(
        self, joint_state: State, agent_id: int
    ) -> bool:
        """
        If the agent's current task is solved, move it to solved and assign a new task.
        Returns True when a new task was assigned and replanned.
        """

        current_task = self.agents[agent_id].task

        if current_task is None:
            return False

        if not self._task_is_solved(joint_state, agent_id, current_task):
            if len(self.agents[agent_id]._plan) == 0:
                # preplan here
                return self._preplan_for_agent(agent_id, joint_state, self.timestep)
            return False

        self.color_tasks[State.agent_colors[agent_id]]["solved_tasks"].append(
            current_task
        )
        self.agents[agent_id].task = None
        self.agents[agent_id]._plan = []
        self.agents[agent_id]._plan_index = 0

        # If this agent has a positional goal that's no longer satisfied (the
        # agent moved away from it after marking it solved), re-queue the
        # move_agent task so it goes back to its goal.
        if self.profile is not None:
            ag_color = State.agent_colors[agent_id]
            ag = self.agents[agent_id]
            for gr, gc, gid in self.profile.agent_goals:
                if gid != agent_id:
                    continue
                if (ag.agent_row, ag.agent_col) == (gr, gc):
                    continue  # still at goal — nothing to do
                # Find a solved move_agent task for this goal and move it back
                solved_list = self.color_tasks[ag_color]["solved_tasks"]
                for idx in range(len(solved_list) - 1, -1, -1):
                    st = solved_list[idx]
                    if (
                        st.task_type == "move_agent"
                        and st.goal_pos == (gr, gc)
                    ):
                        ag.has_reached_its_goal = False
                        self.color_tasks[ag_color]["future_agent_tasks"].append(st)
                        del solved_list[idx]
                        print(
                            f"  Agent {agent_id}: moved away from goal ({gr},{gc}), "
                            f"re-queuing move_agent task.",
                            file=sys.stderr, flush=True,
                        )
                        break

        print(
            f"  Agent {agent_id}: current task solved, moved to solved list.",
            file=sys.stderr,
            flush=True,
        )

        # Unblock any agents waiting for this agent to clear an obstacle
        for other_id, waiting_for in self.agents_awaiting_other_agent.items():
            if waiting_for == agent_id:
                self.agents_awaiting_other_agent[other_id] = None
                self.agents[other_id].awaiting_cnt = 0
                self.agents[other_id]._plan = []
                self.agents[other_id]._plan_index = 0
                print(
                    f"  Agent {other_id}: unblocked (Agent {agent_id} completed task).",
                    file=sys.stderr,
                    flush=True,
                )

        next_task = self._pop_next_task_for_agent(agent_id, joint_state)
        if next_task is None:
            self.agents[agent_id]._plan = []
            self.agents[agent_id]._plan_index = 0
            self.agents[agent_id].tasks = []
            self.agents[agent_id].agent_goal = None
            print(
                f"  Agent {agent_id}: no more tasks available.",
                file=sys.stderr,
                flush=True,
            )
            return False

        # current_tasks.append(next_task)
        self.agents[agent_id].task = next_task
        self._sync_agent_task_state(agent_id, next_task)

        print(
            f"  Agent {agent_id}: assigned new task {next_task.task_type}, replanning.",
            file=sys.stderr,
            flush=True,
        )

        return self._preplan_for_agent(agent_id, joint_state, self.timestep)

    def _preplan_for_agent(
        self, agent_id: int, joint_state: State, timestep: int
    ) -> bool:
        if len(self.agents[agent_id]._plan) == 0:
            print(
                f"  Agent {agent_id}: no plan found for current task, attempting preplan.",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"  Agent {agent_id}: replanning for agent.",
                file=sys.stderr,
                flush=True,
            )

        agent = self.agents[agent_id]
        current_task = agent.task

        # NOTE: an agent should try to solve its task and if it keeps on failing then it should just await
        # this is because another agent can unblock it by solving the obstacle task that is blocking it

        # tho for now i will just skip this check as the agents are stuck awaiting each other despite their plans have changed?
        if self.agents_awaiting_other_agent[agent_id] is not None:
            print(
                f"  Agent {agent_id}: currently awaiting task from Agent {self.agents_awaiting_other_agent[agent_id]}, but will attempt to preplan anyway in case the blocking obstacle has been resolved.",
                file=sys.stderr,
                flush=True,
            )
            awaiting_threshold = (
                10
                + agent_id  # number of timesteps to wait before attempting to preplan anyway
            )
            if self.agents[agent_id].awaiting_cnt > awaiting_threshold:
                print(
                    f"  Agent {agent_id}: has been awaiting for {self.agents[agent_id].awaiting_cnt} timesteps, will attempt to preplan regardless of awaiting status.",
                    file=sys.stderr,
                    flush=True,
                )
                self.agents[agent_id].awaiting_cnt = 0
            else:
                self.agents[agent_id].awaiting_cnt += 1
                return False
        # if self.agents_awaiting_other_agent[agent_id] is not None:
        #     print(
        #         f"  Agent {agent_id}: currently awaiting task from Agent {self.agents_awaiting_other_agent[agent_id]}, skipping preplan until that task is received.",
        #         file=sys.stderr,
        #         flush=True,
        #     )
        #     return False

        if current_task is None:
            return False

        if len(agent._plan) == 0:
            obstacles = self._find_obstacles(joint_state, agent_id, current_task)
            if obstacles is None:
                box_obstacles = None
            else:
                box_obstacles = [
                    (r, c, color, after_box, obj_type)
                    for r, c, color, after_box, obj_type in obstacles
                    if obj_type != "agent"
                ]

            if (
                box_obstacles is not None
                and len(box_obstacles) > 0
                and box_obstacles[0][2] == State.agent_colors[agent_id]
                and box_obstacles[0][4] == "box"
            ):
                # NOTE: same colored box obstacle case
                managed_to_swap = self._swap_task_with_obstacle(
                    agent.agent_id,
                    box_obstacles[0],
                    state=joint_state,
                )
                if managed_to_swap and agent.task is not None:
                    current_task = agent.task
                else:
                    print(
                        f"  Agent {agent_id}: failed to swap with obstacle task, will attempt to plan with original task.",
                        file=sys.stderr,
                        flush=True,
                    )
            elif (
                box_obstacles is not None
                and len(box_obstacles) > 0
                and box_obstacles[0][2] != State.agent_colors[agent_id]
                and box_obstacles[0][4] == "box"
            ):
                print(
                    f"  Agent {agent_id}: different colored box obstacle detected at ({box_obstacles[0][0]}, {box_obstacles[0][1]}).",
                    file=sys.stderr,
                    flush=True,
                )

                foreign_agent = next(
                    (
                        agent
                        for agent in self.agents
                        if agent.agent_id != agent_id
                        and State.agent_colors[agent.agent_id] == box_obstacles[0][2]
                    ),
                    None,
                )
                if foreign_agent is None:
                    print(
                        f"  Agent {agent_id}: no foreign agent of color {box_obstacles[0][2]} exists, skipping swap.",
                        file=sys.stderr, flush=True,
                    )
                elif self.agents_awaiting_other_agent[agent_id] is None:

                    managed_foreign_to_swap = self._swap_foreign_task_with_obstacle(
                        foreign_agent.agent_id,
                        box_obstacles[0],
                        joint_state,
                        requester_id=agent_id,
                    )

                    if managed_foreign_to_swap and foreign_agent.task is not None:
                        print(
                            f"  Agent {agent_id}: successfully swapped obstacle task with Agent {foreign_agent.agent_id}. Attempting to replan with original task.",
                            file=sys.stderr,
                            flush=True,
                        )

                        foreign_agent._plan = []
                        foreign_agent._plan_index = 0
                        self.agents_awaiting_other_agent[agent_id] = (
                            foreign_agent.agent_id
                        )
                        self.agents_awaiting_other_agent[foreign_agent.agent_id] = None
                else:
                    print(
                        f"  Agent {agent_id}: already awaiting task from Agent {self.agents_awaiting_other_agent[agent_id]}.",
                        file=sys.stderr,
                        flush=True,
                    )
            elif (
                obstacles is not None
                and len(obstacles) > 0
                and obstacles[0][4] == "agent"
            ):
                print(
                    f"  Agent {agent_id}: agent obstacle detected at ({obstacles[0][0]}, {obstacles[0][1]}). Will attempt to re-plan with original task and hope to find a way around the agent obstacle.",
                    file=sys.stderr,
                    flush=True,
                )
                obstacle_agent_id = next(
                    agent.agent_id
                    for agent in self.agents
                    if agent.agent_id != agent_id
                    and State.agent_colors[agent.agent_id] == obstacles[0][2]
                    and agent.agent_row == obstacles[0][0]
                    and agent.agent_col == obstacles[0][1]
                )
                self.agents_awaiting_other_agent[agent_id] = obstacle_agent_id

                if (
                    self.agents[obstacle_agent_id].task is not None
                    and len(self.agents[obstacle_agent_id]._plan) == 0
                ):
                    # Give the obstacle agent a single deterministic cardinal
                    # nudge to move out of the way. Direction rotates per
                    # consecutive call so we don't pick the same one twice.
                    obstacle_agent = self.agents[obstacle_agent_id]
                    chosen = self._pick_recovery_move(joint_state, obstacle_agent.agent_id)
                    if chosen is not None:
                        obstacle_agent._plan = [chosen]
                        obstacle_agent._plan_index = 0
                        print(
                            f"  Obstacle Agent {obstacle_agent_id}: unblock nudge {chosen.name_}",
                            file=sys.stderr,
                            flush=True,
                        )

            # NOTE: other edge cases here i guess

            # Refresh the box's actual position from joint_state — it may have moved
            # since the task was created. Stale object_pos was causing infinite
            # invalid-action loops.
            if (
                current_task.task_type == "move_box"
                and current_task.box_char is not None
            ):
                gr, gc = current_task.goal_pos
                actual_pos = self._find_box_current_pos(
                    joint_state, current_task.box_char, gr, gc
                )
                if actual_pos is not None and actual_pos != current_task.object_pos:
                    print(
                        f"  Agent {agent_id}: box {current_task.box_char} moved from "
                        f"{current_task.object_pos} to {actual_pos}, refreshing task.",
                        file=sys.stderr, flush=True,
                    )
                    current_task.object_pos = actual_pos

            goal_tuple = self._convert_task_to_goal_tuple(current_task)

            # A* runs in solver-local time starting at t=0; awaiting-agent
            # constraints are encoded at search-time t=0..9 (10-turn horizon).
            runtime_constraints = self._build_runtime_constraints(
                agent_id, 0, horizon=10
            )
            plan = solve(
                state=joint_state,
                agent_id=agent.agent_id,
                goal=goal_tuple,
                constraints=runtime_constraints,
                dist_map=self.dist_map,
            )

            # Ghost-state fallback: if regular A* fails, retry with other
            # agents removed (placed at the wall cell (0,0) so they vanish).
            # A* then finds a topological path through the map; runtime
            # conflict resolution handles collisions when they materialise.
            # Applies to both nav and box tasks — works when an agent is
            # blocked by another agent sitting in the only viable corridor.
            if plan is None:
                ghost_rows = list(joint_state.agent_rows)
                ghost_cols = list(joint_state.agent_cols)
                for a in range(len(ghost_rows)):
                    if a != agent_id:
                        ghost_rows[a] = 0  # wall cell — agent vanishes
                        ghost_cols[a] = 0
                ghost = State(ghost_rows, ghost_cols, joint_state.boxes)
                plan = solve(
                    state=ghost,
                    agent_id=agent.agent_id,
                    goal=goal_tuple,
                    constraints=set(),
                    dist_map=self.dist_map,
                )
                if plan:
                    print(
                        f"  Agent {agent_id}: {current_task.task_type} plan "
                        f"via ghost-state (other agents removed), length "
                        f"{len(plan)}.",
                        file=sys.stderr, flush=True,
                    )

            if plan:
                agent._plan = plan
                agent._plan_index = 0
                self.agents_awaiting_other_agent[agent_id] = None
                print(
                    f"  Agent {agent_id}: preplan found (length {len(plan)})",
                    file=sys.stderr,
                    flush=True,
                )
                return True

            # A* failed. Check if goal cell is blocked by another box and
            # wait for the responsible agent of that color to clear it.
            if current_task.task_type == "move_box" and self.agents_awaiting_other_agent[agent_id] is None:
                gr, gc = current_task.goal_pos
                blocker_char = joint_state.boxes[gr][gc]
                if blocker_char:
                    blocker_color = State.box_colors[ord(blocker_char) - ord("A")]
                    for other in self.agents:
                        if other.agent_id != agent_id and Color.compatible(
                            State.agent_colors[other.agent_id], blocker_color
                        ):
                            self.agents_awaiting_other_agent[agent_id] = other.agent_id
                            self.agents[agent_id].awaiting_cnt = 0
                            print(
                                f"  Agent {agent_id}: goal cell ({gr},{gc}) has {blocker_char}, waiting for Agent {other.agent_id}.",
                                file=sys.stderr,
                                flush=True,
                            )
                            break

            print(
                f"  Agent {agent_id}: preplan FAILED for task {current_task}",
                file=sys.stderr,
                flush=True,
            )

            # Give-up counter: if the SAME task fails N times in a row, push
            # it back to the queue and clear agent.task so the next pop tries
            # a different task (final-sweep will re-enqueue if needed). Stops
            # infinite preplan-failure loops.
            sig = (
                current_task.task_type, current_task.box_char,
                current_task.object_pos, current_task.goal_pos,
            )
            last_sig = getattr(agent, "_last_failed_sig", None)
            cnt = getattr(agent, "_failed_preplan_cnt", 0)
            if sig == last_sig:
                cnt += 1
            else:
                cnt = 1
            agent._last_failed_sig = sig
            agent._failed_preplan_cnt = cnt
            if cnt >= 25 and current_task.task_type == "move_box":
                box_color = (
                    State.box_colors[ord(current_task.box_char) - ord("A")]
                    if current_task.box_char else None
                )
                if box_color in self.color_tasks:
                    self.color_tasks[box_color]["future_box_tasks"].append(current_task)
                print(
                    f"  Agent {agent_id}: task failed {cnt}× in a row; "
                    f"requeuing and dropping.",
                    file=sys.stderr, flush=True,
                )
                agent.task = None
                agent._plan = []
                agent._plan_index = 0
                agent._failed_preplan_cnt = 0
                agent._last_failed_sig = None
            return False

        return False

    def _convert_task_to_goal_tuple(
        self, task: Task
    ) -> tuple[int | None, int | None, int, int, str | None]:
        if task.task_type == "move_box":
            return (
                task.object_pos[0],
                task.object_pos[1],
                task.goal_pos[0],
                task.goal_pos[1],
                task.box_char,
            )
        else:
            return (
                None,
                None,
                task.goal_pos[0],
                task.goal_pos[1],
                None,
            )

    def _find_obstacles(
        self, initial_state: State, agent_id: int, failed_task: Task
    ) -> list[tuple[int, int, Color, bool, str]] | None:
        # If the full-world planner already finds a route, there are no
        # blocking obstacles to report.
        direct_plan = solve(
            state=initial_state,
            agent_id=agent_id,
            goal=(
                failed_task.object_pos[0],
                failed_task.object_pos[1],
                failed_task.goal_pos[0],
                failed_task.goal_pos[1],
                failed_task.box_char,
            ),
            constraints=set(),
            dist_map=self.dist_map,
        )
        if direct_plan is not None:
            return []

        # Build a reduced ghost state containing only the current agent and,
        # for move_box tasks, only the target box. This gives us a shortest
        # path in the map topology while ignoring irrelevant objects.
        ghost_agent_rows = [
            row if idx == agent_id else 0
            for idx, row in enumerate(initial_state.agent_rows)
        ]
        ghost_agent_cols = [
            col if idx == agent_id else 0
            for idx, col in enumerate(initial_state.agent_cols)
        ]
        ghost_boxes = [["" for _ in row] for row in initial_state.boxes]
        # Boxes at their final goal positions are kept as permanent obstacles in
        # the ghost state — they will not be moved, so paths must go around them.
        for r in range(len(initial_state.boxes)):
            for c in range(len(initial_state.boxes[r])):
                ch = initial_state.boxes[r][c]
                if ch and State.goals[r][c] == ch:
                    ghost_boxes[r][c] = ch
        if failed_task.task_type == "move_box" and failed_task.box_char is not None:
            box_r, box_c = failed_task.object_pos
            ghost_boxes[box_r][box_c] = failed_task.box_char

        ghost_state = State(ghost_agent_rows, ghost_agent_cols, ghost_boxes)

        goal = self._convert_task_to_goal_tuple(failed_task)
        plan = solve(
            state=ghost_state,
            agent_id=agent_id,
            goal=goal,
            constraints=set(),
            dist_map=self.dist_map,
        )
        if plan is None:
            return None

        obstacles: list[tuple[int, int, Color, bool, str]] = []
        # NOTE: obstacles is a list of tuples such that:
        # - (r, c, color, obstacle_after_box, object_type)

        seen: set[tuple[int, int, str]] = set()
        # NOTE: seen is a set of (r, c, object_type) to avoid adding the same obstacle multiple times if it appears in multiple steps of the path

        def add_obstacle(
            r: int,
            c: int,
            obstacle_color: Color,
            obstacle_after_box: bool,
            obstacle_type: str,
        ) -> None:
            key = (r, c, obstacle_type)
            if key in seen:
                return
            seen.add(key)
            obstacles.append((r, c, obstacle_color, obstacle_after_box, obstacle_type))

        target_box_char = failed_task.box_char
        target_box_pos = (
            failed_task.object_pos if failed_task.task_type == "move_box" else None
        )
        target_box_moved = False

        sim_agent_r = initial_state.agent_rows[agent_id]
        sim_agent_c = initial_state.agent_cols[agent_id]
        sim_target_box_r, sim_target_box_c = (
            target_box_pos if target_box_pos is not None else (None, None)
        )

        def inspect_cell(r: int, c: int, obstacle_after_box: bool) -> None:
            # Check agents first.
            for other_id, (ar, ac) in enumerate(
                zip(initial_state.agent_rows, initial_state.agent_cols)
            ):
                if other_id == agent_id:
                    continue
                if ar == r and ac == c:
                    add_obstacle(
                        r,
                        c,
                        State.agent_colors[other_id],  # type: ignore
                        obstacle_after_box,
                        "agent",
                    )
                    return

            # Then check boxes.
            ch = initial_state.boxes[r][c]
            if not ch:
                return
            if (
                target_box_char is not None
                and failed_task.task_type == "move_box"
                and ch == target_box_char
                and (r, c) == (sim_target_box_r, sim_target_box_c)
            ):
                return

            color = State.box_colors[ord(ch) - ord("A")]
            if color is None:
                return
            add_obstacle(r, c, color, obstacle_after_box, "box")

        for action in plan:
            # Obstacles are considered "after the box" if the target box has
            # already moved, or if this very action moves the target box into
            # the obstacle's field.
            moves_target_box_this_action = False
            if failed_task.task_type == "move_box" and target_box_char is not None:
                if action.type is ActionType.Push:
                    src_r = sim_agent_r + action.agent_row_delta
                    src_c = sim_agent_c + action.agent_col_delta
                    moves_target_box_this_action = (
                        initial_state.boxes[src_r][src_c] == target_box_char
                    )
                elif action.type is ActionType.Pull:
                    box_r = sim_agent_r - action.box_row_delta
                    box_c = sim_agent_c - action.box_col_delta
                    moves_target_box_this_action = (
                        initial_state.boxes[box_r][box_c] == target_box_char
                    )

            obstacle_after_box = (
                target_box_moved or moves_target_box_this_action
                if failed_task.task_type == "move_box"
                else False
            )

            if action.type is ActionType.Move:
                next_r = sim_agent_r + action.agent_row_delta
                next_c = sim_agent_c + action.agent_col_delta
                inspect_cell(next_r, next_c, obstacle_after_box)
                sim_agent_r, sim_agent_c = next_r, next_c
                continue

            if action.type is ActionType.Push:
                src_r = sim_agent_r + action.agent_row_delta
                src_c = sim_agent_c + action.agent_col_delta
                dst_r = src_r + action.box_row_delta
                dst_c = src_c + action.box_col_delta

                inspect_cell(src_r, src_c, obstacle_after_box)
                inspect_cell(dst_r, dst_c, obstacle_after_box)

                # If the target box is the one being pushed, mark the rest of the
                # path as after the box has been reached/moved.
                if (
                    failed_task.task_type == "move_box"
                    and target_box_char is not None
                    and initial_state.boxes[src_r][src_c] == target_box_char
                ):
                    target_box_moved = True
                    sim_target_box_r, sim_target_box_c = dst_r, dst_c

                sim_agent_r, sim_agent_c = src_r, src_c
                continue

            if action.type is ActionType.Pull:
                next_r = sim_agent_r + action.agent_row_delta
                next_c = sim_agent_c + action.agent_col_delta
                box_r = sim_agent_r - action.box_row_delta
                box_c = sim_agent_c - action.box_col_delta

                inspect_cell(next_r, next_c, obstacle_after_box)
                inspect_cell(box_r, box_c, obstacle_after_box)
                inspect_cell(sim_agent_r, sim_agent_c, obstacle_after_box)

                if (
                    failed_task.task_type == "move_box"
                    and target_box_char is not None
                    and initial_state.boxes[box_r][box_c] == target_box_char
                ):
                    target_box_moved = True
                    sim_target_box_r, sim_target_box_c = sim_agent_r, sim_agent_c

                sim_agent_r, sim_agent_c = next_r, next_c

        return obstacles

    def _find_box_current_pos(
        self, joint_state: State, box_char: str, goal_r: int, goal_c: int
    ) -> tuple[int, int] | None:
        """Find the current (row, col) of an UNDELIVERED box with the given
        char. A box is "undelivered" only if its cell is not ANY same-char
        goal (not just the specific target). Without this, multi-instance
        same-char levels (e.g. duckie's 4 U's) end up dragging a U from one
        satisfied U goal to another, un-satisfying the first. Returns the
        off-all-goals box closest to the requested goal."""
        candidates: list[tuple[int, int, int]] = []
        for r, row in enumerate(joint_state.boxes):
            for c, ch in enumerate(row):
                if ch != box_char:
                    continue
                # Skip if (r, c) is a goal for this same char — that box is
                # already delivered; do not steal it.
                if State.goals[r][c] == box_char:
                    continue
                d = abs(r - goal_r) + abs(c - goal_c)
                candidates.append((d, r, c))
        if not candidates:
            return None
        candidates.sort()
        return (candidates[0][1], candidates[0][2])

    def _find_obs_clear_goal(
        self, obs_r: int, obs_c: int, state: "State | None" = None
    ) -> tuple[int, int]:
        """
        Find the best adjacent cell to push the obstacle box to.

        Priority:
          1. Adjacent free cell where the push-from cell (opposite side) is also
             free of walls AND boxes (single-step push immediately feasible).
          2. Any adjacent free cell (push-from may be blocked; plan handles it reactively).

        "Free" means: not a wall, and (if state given) not occupied by a box.
        Agents are not considered static obstacles here.
        """
        num_rows = len(State.walls)
        num_cols = len(State.walls[0]) if num_rows > 0 else 0

        def is_free(r: int, c: int) -> bool:
            if not (0 <= r < num_rows and 0 <= c < num_cols):
                return False
            if State.walls[r][c]:
                return False
            if state is not None and state.boxes[r][c] != "":
                return False
            return True

        def is_chokepoint_for_pending(r: int, c: int) -> bool:
            """True if (r, c) is a box-chokepoint cell whose blocked box
            delivery is still pending — pushing the obstacle here would
            permanently break that other delivery."""
            box_blockers = getattr(self, "box_chokepoint_blockers", None)
            if box_blockers and (r, c) in box_blockers:
                for _br, _bc, b_char in box_blockers[(r, c)]:
                    if self._box_delivery_pending(b_char):
                        return True
            return False

        # Pass 1: dest free AND push-from free AND not a pending-box-chokepoint cell
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dest_r, dest_c = obs_r + dr, obs_c + dc
            push_from_r, push_from_c = obs_r - dr, obs_c - dc
            if (is_free(dest_r, dest_c) and is_free(push_from_r, push_from_c)
                    and not is_chokepoint_for_pending(dest_r, dest_c)):
                return (dest_r, dest_c)

        # Pass 2: BFS through all non-wall cells (including box-occupied ones) to find
        # the nearest box-FREE cell with ≥3 free cardinal neighbors (a true open space).
        # This ensures C doesn't end up in another corridor bottleneck.
        from collections import deque as _dq

        def free_cardinal_count(r: int, c: int) -> int:
            cnt = 0
            for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r2, c2 = r + dr2, c + dc2
                if 0 <= r2 < num_rows and 0 <= c2 < num_cols and not State.walls[r2][c2]:
                    cnt += 1
            return cnt

        bfs_visited: set = {(obs_r, obs_c)}
        bfs_q: _dq = _dq([(obs_r, obs_c)])
        any_free: tuple | None = None

        while bfs_q:
            r, c = bfs_q.popleft()
            if (r, c) != (obs_r, obs_c) and not State.walls[r][c]:
                # Only return this cell if it's currently box-free AND
                # not a pending-box chokepoint
                cell_has_box = state is not None and state.boxes[r][c] != ""
                if not cell_has_box and not is_chokepoint_for_pending(r, c):
                    if any_free is None:
                        any_free = (r, c)
                    if free_cardinal_count(r, c) >= 3:
                        return (r, c)
            # Always traverse through (even box-occupied cells) to reach open space beyond
            for dr, dc in [(1, 0), (0, 1), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (
                    (nr, nc) not in bfs_visited
                    and 0 <= nr < num_rows
                    and 0 <= nc < num_cols
                    and not State.walls[nr][nc]
                ):
                    bfs_visited.add((nr, nc))
                    bfs_q.append((nr, nc))

        return any_free if any_free is not None else (obs_r, obs_c)

    # ------------------------------------------------------------------
    # Negotiation log helpers
    # ------------------------------------------------------------------

    def _record_swap(
        self, requester: int, granter: int, obstacle_box: str,
        obstacle_pos: tuple[int, int], original_task: "Task | None",
        new_task: "Task", same_color: bool,
    ) -> None:
        """Append a TaskSwap entry to the negotiation log."""
        swap = TaskSwap(
            requester=requester,
            granter=granter,
            obstacle_box=obstacle_box,
            obstacle_pos=obstacle_pos,
            original_task=original_task,
            new_task=new_task,
            t_initiated=self.timestep,
            same_color=same_color,
        )
        self.negotiations.append(swap)
        kind = "intra-color" if same_color else "cross-color"
        print(
            f"  [NEGOTIATION] t={self.timestep} {kind}: Agent {requester} "
            f"requested Agent {granter} clear box {obstacle_box} at {obstacle_pos}.",
            file=sys.stderr, flush=True,
        )

    def _detect_blocking_cycle(
        self, state: "State", start_agent_id: int, max_hops: int = 12,
    ) -> list[int] | None:
        """
        Follow the obstacle chain: agent X's task is move BOX_X to GOAL_X;
        GOAL_X is occupied by some other box Y; box Y's owner is the next
        link. Returns the closed cycle of agent_ids when the chain comes
        back to start_agent_id (every agent in the chain blocked by another
        blocked agent — task-swapping cannot make progress). Otherwise None.

        Used to detect 4-corner-swap deadlocks (e.g. MAvis) where the chain
        of obstacle-clear requests rotates around the ring without anyone
        physically moving a box.
        """
        chain: list[int] = []
        current = start_agent_id
        num_rows = len(state.boxes)
        num_cols = len(state.boxes[0]) if num_rows > 0 else 0
        for _ in range(max_hops):
            if current in chain:
                return chain if chain[0] == current else None
            chain.append(current)

            task = self.agents[current].task
            if task is None or task.task_type != "move_box":
                return None
            if task.box_char is None:
                return None
            gr, gc = task.goal_pos
            if not (0 <= gr < num_rows and 0 <= gc < num_cols):
                return None
            blocker = state.boxes[gr][gc]
            if not blocker or blocker == task.box_char:
                return None
            idx = ord(blocker) - ord("A")
            if idx < 0 or idx >= 26:
                return None
            blocker_color = State.box_colors[idx]
            if blocker_color is None:
                return None
            next_owner = None
            for a in self.agents:
                if a.agent_id == current:
                    continue
                if State.agent_colors[a.agent_id] == blocker_color:
                    next_owner = a.agent_id
                    break
            if next_owner is None:
                return None
            current = next_owner
        return None

    def _find_park_cell(
        self,
        obs_r: int,
        obs_c: int,
        state: "State",
    ) -> tuple[int, int] | None:
        """
        Pick a free non-goal cell to temporarily park an obstacle box at, so
        a deadlock cycle can be broken. The cell must be:
          • not a wall,
          • not currently occupied by a box,
          • not any box's goal cell (so we don't strand a different delivery).

        Pass 1 returns an adjacent cell where the immediate push is feasible
        (dest free + push-from free of walls/boxes). Pass 2 BFS-walks the
        reachable region to find any eligible cell.
        """
        num_rows = len(State.walls)
        num_cols = len(State.walls[0]) if num_rows > 0 else 0
        walls = State.walls
        goals = State.goals
        boxes = state.boxes

        def can_host_box(r: int, c: int) -> bool:
            if not (0 <= r < num_rows and 0 <= c < num_cols):
                return False
            if walls[r][c]:
                return False
            if boxes[r][c]:
                return False
            if goals[r][c]:
                return False
            return True

        def is_traversable(r: int, c: int) -> bool:
            if not (0 <= r < num_rows and 0 <= c < num_cols):
                return False
            if walls[r][c]:
                return False
            if boxes[r][c]:
                return False
            return True

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dest_r, dest_c = obs_r + dr, obs_c + dc
            push_from_r, push_from_c = obs_r - dr, obs_c - dc
            if can_host_box(dest_r, dest_c) and is_traversable(push_from_r, push_from_c):
                return (dest_r, dest_c)

        from collections import deque as _dq
        visited: set[tuple[int, int]] = {(obs_r, obs_c)}
        q: _dq = _dq([(obs_r, obs_c)])
        while q:
            r, c = q.popleft()
            if (r, c) != (obs_r, obs_c) and can_host_box(r, c):
                return (r, c)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in visited:
                    continue
                if 0 <= nr < num_rows and 0 <= nc < num_cols and not walls[nr][nc]:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return None

    def _recent_swap_count(
        self, requester: int, granter: int, obstacle_box: str, window: int = 30,
    ) -> int:
        """Count how many times (requester, granter, obstacle_box) has been
        swapped within the recent `window` timesteps. Used to detect
        oscillating negotiation cycles."""
        return sum(
            1 for s in self.negotiations
            if (self.timestep - s.t_initiated) < window
            and s.requester == requester
            and s.granter == granter
            and s.obstacle_box == obstacle_box
        )

    def _swap_foreign_task_with_obstacle(
        self,
        foreign_agent_id: int,
        obstacle: tuple[int, int, Color, bool, str],
        state: "State | None" = None,
        requester_id: int | None = None,
    ) -> bool:
        """
        Cross-color negotiation: reassign a foreign agent's task to clear a
        blocking obstacle box.

        Args:
            foreign_agent_id: ID of the agent of the obstacle's color (the granter)
            obstacle: (obs_r, obs_c, obs_color, obstacle_after_box, obstacle_type)
            state:    optional joint state for clear-goal lookup
            requester_id: agent that requested the swap (the one being blocked).
                          If None, swap is recorded but cycle-detection key is
                          incomplete.
        """
        obs_r, obs_c, obs_color, obstacle_after_box, obstacle_type = obstacle

        current_task = self.agents[foreign_agent_id].task
        # Read the actual char of the obstacle box (handles multi-box-same-
        # color levels where the color's canonical char would point at an
        # unrelated box). Fall back to canonical if state unavailable.
        if (state is not None
            and 0 <= obs_r < len(state.boxes)
            and 0 <= obs_c < len(state.boxes[obs_r])
            and state.boxes[obs_r][obs_c]):
            box_char = state.boxes[obs_r][obs_c]
        else:
            box_char = State.get_box_char_from_color(obs_color)

        # If the obstacle box is at its FINAL goal position, refuse to move it.
        # The blocked agent must find another path.
        if (
            box_char is not None
            and 0 <= obs_r < len(State.goals)
            and 0 <= obs_c < len(State.goals[obs_r])
            and State.goals[obs_r][obs_c] == box_char
        ):
            print(
                f"  Foreign Agent {foreign_agent_id}: box {box_char} at ({obs_r},{obs_c}) "
                f"is at its final goal — refusing to move.",
                file=sys.stderr, flush=True,
            )
            return False

        # If foreign agent already has a task to move this obstacle box, don't swap —
        # just let the requesting agent wait for it.
        if (
            current_task is not None
            and current_task.task_type == "move_box"
            and current_task.box_char is not None
            and current_task.object_pos == (obs_r, obs_c)
        ):
            # Closed-cycle break: if the requester→foreign chain closes
            # back to the requester, every agent in the ring is blocked
            # by another blocked agent and no task swap progresses. Retask
            # the foreign agent to PARK its box at a free non-goal cell,
            # and requeue its original delivery so it resumes once the
            # cycle has unwound.
            if (
                state is not None
                and requester_id is not None
                and box_char is not None
            ):
                cycle = self._detect_blocking_cycle(state, requester_id)
                if cycle and foreign_agent_id in cycle and len(cycle) >= 2:
                    park_cell = self._find_park_cell(obs_r, obs_c, state)
                    if (
                        park_cell is not None
                        and park_cell != (obs_r, obs_c)
                        and park_cell != current_task.goal_pos
                    ):
                        obs_color_local = State.box_colors[ord(box_char) - ord("A")]
                        if obs_color_local in self.color_tasks:
                            self.color_tasks[obs_color_local][
                                "future_box_tasks"
                            ].appendleft(current_task)
                        park_task = Task(
                            task_type="move_box",
                            object_pos=(obs_r, obs_c),
                            goal_pos=park_cell,
                            box_char=box_char,
                            crucial=False,
                        )
                        self.agents[foreign_agent_id].task = park_task
                        self.agents[foreign_agent_id]._plan = []
                        self.agents[foreign_agent_id]._plan_index = 0
                        self._sync_agent_task_state(foreign_agent_id, park_task)
                        print(
                            f"  Cycle break: Agent {foreign_agent_id} retasked to "
                            f"park box {box_char} from ({obs_r},{obs_c}) to "
                            f"{park_cell}; original delivery requeued. "
                            f"Cycle was {cycle}.",
                            file=sys.stderr, flush=True,
                        )
                        self._record_swap(
                            requester=requester_id,
                            granter=foreign_agent_id,
                            obstacle_box=box_char,
                            obstacle_pos=(obs_r, obs_c),
                            original_task=current_task,
                            new_task=park_task,
                            same_color=False,
                        )
                        return True
            print(
                f"  Foreign Agent {foreign_agent_id}: already working on obstacle at ({obs_r},{obs_c}), skip swap.",
                file=sys.stderr, flush=True,
            )
            return True  # treat as success so awaiting gets set

        # Cycle detection: if this exact (requester, granter, box) has been
        # swapped 3+ times in the recent window, the negotiation chain is in
        # oscillation — refuse further swaps to break the cycle. Plan-B:
        # send the requester to an escape cell so the topology has a chance
        # to change before another swap is attempted.
        if box_char is not None and requester_id is not None:
            cycle_count = self._recent_swap_count(
                requester=requester_id,
                granter=foreign_agent_id,
                obstacle_box=box_char,
            )
            if cycle_count >= 3:
                print(
                    f"  Negotiation cycle detected: Agent {requester_id} -> "
                    f"Agent {foreign_agent_id} over box {box_char} ({cycle_count}x); "
                    f"refusing further swaps.",
                    file=sys.stderr, flush=True,
                )
                # Plan-B: route the requester to an escape cell. Rotate the
                # direction bias on consecutive triggers so the agent isn't
                # sent to the same cell every time.
                requester = self.agents[requester_id]
                if state is not None:
                    rot = self.agents_escape_rotation.get(requester_id, 0)
                    self.agents_escape_rotation[requester_id] = rot + 1
                    escape = self._find_escape_cell(state, requester_id, rotation=rot)
                    if escape is not None:
                        if requester.agent_goal is not None:
                            requester._pending_agent_goal = requester.agent_goal
                        requester.agent_goal = escape
                        requester.constraints = set()
                        requester._plan = []
                        requester._plan_index = 0
                        self.agents_awaiting_other_agent[requester_id] = None
                        requester.awaiting_cnt = 0
                        print(
                            f"    Plan-B: Agent {requester_id} routed to escape cell {escape}.",
                            file=sys.stderr, flush=True,
                        )
                return False

        clear_goal = self._find_obs_clear_goal(obs_r, obs_c, state)
        # box_char was set at the top of this function using state lookup;
        # reuse that here so the clearing task points at the actual obstacle
        # box rather than the color's canonical char.
        new_task = Task(
            task_type="move_box",
            object_pos=(obs_r, obs_c),
            goal_pos=clear_goal,
            box_char=box_char,
        )

        # if (
        #     current_task is not None
        #     and current_task.task_type == "move_box"
        #     and current_task.box_char is not None
        #     and current_task.object_pos == (obs_r, obs_c)
        #     and current_task.goal_pos
        #     == (
        #         self.agents[foreign_agent_id].agent_row,
        #         self.agents[foreign_agent_id].agent_col,
        #     )
        #     and State.box_colors[ord(current_task.box_char) - ord("A")] == obs_color
        # ):
        #     print(
        #         f"  Foreign Agent {foreign_agent_id}: current task already matches the obstacle task. No swap needed.",
        #         file=sys.stderr,
        #         flush=True,
        #     )
        #     return False

        for idx, task in enumerate(self.color_tasks[obs_color]["solved_tasks"]):
            if (
                task.task_type == "move_box"
                and task.box_char is not None
                and obs_color == State.box_colors[ord(task.box_char) - ord("A")]
                and task.goal_pos == (obs_r, obs_c)
                and task.crucial is True
            ):
                # move this solved box task back to future since box is now an obstacle that needs to be moved
                task.object_pos = (
                    self.agents[foreign_agent_id].agent_row,
                    self.agents[foreign_agent_id].agent_col,
                )
                self.color_tasks[obs_color]["future_box_tasks"].append(task)
                del self.color_tasks[obs_color]["solved_tasks"][idx]
                print(
                    f"  Foreign Agent {foreign_agent_id}: Obstacle box {obs_color} was previously marked as having solved its box goal, but it is now blocking another agent. Moving its box goal task back to future tasks.",
                    file=sys.stderr,
                    flush=True,
                )
                break
        for idx, task in enumerate(self.color_tasks[obs_color]["future_box_tasks"]):
            if (
                task.task_type == "move_box"
                and task.box_char is not None
                and obs_color == State.box_colors[ord(task.box_char) - ord("A")]
                and task.object_pos == (obs_r, obs_c)
            ):
                # update the future task for the obstacle box to reflect its new position as an obstacle
                task.object_pos = (obs_r, obs_c)
                print(
                    f"  Foreign Agent {foreign_agent_id}: Found future task for obstacle box {obs_color} at ({obs_r}, {obs_c}) in color future tasks at index {idx}. Updating that future task's object position to reflect current obstacle position.",
                    file=sys.stderr,
                    flush=True,
                )
                break

        if current_task is not None:
            if (
                current_task.task_type == "move_box"
                and current_task.box_char is not None
            ):
                current_task.object_pos = (
                    self.agents[foreign_agent_id].agent_row,
                    self.agents[foreign_agent_id].agent_col,
                )
                if current_task.object_pos != current_task.goal_pos:
                    self.color_tasks[obs_color]["future_box_tasks"].appendleft(
                        current_task
                    )
            elif (
                current_task.task_type == "move_agent"
                and current_task.goal_pos is not None
            ):
                # current_task.goal_pos = (obs_r, obs_c)
                # this doesnt matter i think here
                self.color_tasks[obs_color]["future_agent_tasks"].appendleft(
                    current_task
                )

        self.agents[foreign_agent_id].task = new_task
        self._sync_agent_task_state(foreign_agent_id, new_task)

        # Record this cross-color negotiation for cycle detection + analysis
        if box_char is not None and requester_id is not None:
            self._record_swap(
                requester=requester_id,
                granter=foreign_agent_id,
                obstacle_box=box_char,
                obstacle_pos=(obs_r, obs_c),
                original_task=current_task,
                new_task=new_task,
                same_color=False,
            )

        return True

    def _swap_task_with_obstacle(
        self,
        agent_id: int,
        obstacle: tuple[int, int, Color, bool, str],
        state: "State | None" = None,
    ) -> bool:
        """
        Swap current task with an obstacle task (same-colour case): agent
        re-assigns itself to clear the blocking box, deferring its original
        task back to the colour-group queue.

        Cycle detection: if this same (agent, box_char) has been swap-cleared
        3+ times in the recent window, refuse and route the agent to an
        escape cell. Same Plan-B mechanism as cross-colour cycle refusal.
        """
        assert self.profile is not None

        obs_r, obs_c, obs_color, obstacle_after_box, obstacle_type = obstacle

        print(
            f"  Same-colored obstacle detected: box {obs_color} at ({obs_r}, {obs_c}) blocks Agent {agent_id}.",
            file=sys.stderr,
            flush=True,
        )

        # Cycle detection for same-colour swap thrashing. We only count
        # recent swaps where the obstacle was at the SAME (obs_r, obs_c)
        # — distinct positions mean the previous swap actually moved
        # the box, which counts as progress, not a cycle.
        box_char_check = State.get_box_char_from_color(obs_color)
        if box_char_check is not None:
            recent = sum(
                1 for s in self.negotiations
                if (self.timestep - s.t_initiated) < 30
                and s.requester == agent_id
                and s.granter == agent_id
                and s.obstacle_box == box_char_check
                and s.obstacle_pos == (obs_r, obs_c)
            )
            if recent >= 3:
                print(
                    f"  Same-color swap cycle detected: Agent {agent_id} over "
                    f"box {box_char_check} ({recent}x); refusing further swaps.",
                    file=sys.stderr, flush=True,
                )
                # Plan-A: fresh-task recovery. The current task may have been
                # corrupted by previous swap rounds (wrong object_pos/goal_pos).
                # Scan the live state for any undelivered same-color box and
                # build a clean task from scratch.
                if state is not None and self._fresh_same_color_box_task(
                    agent_id, state
                ):
                    return False
                # Plan-B: escape cell.
                if state is not None:
                    rot = self.agents_escape_rotation.get(agent_id, 0)
                    self.agents_escape_rotation[agent_id] = rot + 1
                    escape = self._find_escape_cell(state, agent_id, rotation=rot)
                    if escape is not None:
                        ag = self.agents[agent_id]
                        if ag.agent_goal is not None:
                            ag._pending_agent_goal = ag.agent_goal
                        ag.agent_goal = escape
                        ag.constraints = set()
                        ag._plan = []
                        ag._plan_index = 0
                        self.agents_awaiting_other_agent[agent_id] = None
                        ag.awaiting_cnt = 0
                        print(
                            f"    Plan-B: Agent {agent_id} routed to escape cell {escape}.",
                            file=sys.stderr, flush=True,
                        )
                return False

        current_task = self.agents[agent_id].task
        if current_task is None:
            print(
                f"  Swap failed: Agent {agent_id} has no current task.",
                file=sys.stderr,
                flush=True,
            )
            return False

        if current_task is None or current_task.box_char is None:
            target_color = State.agent_colors[agent_id]
        else:
            target_color = State.box_colors[ord(current_task.box_char) - ord("A")]
        assert target_color is not None

        obstacle_index = next(
            (
                i
                for i, task in enumerate(
                    self.color_tasks[target_color]["future_box_tasks"]
                )
                if task.task_type == "move_box"
                # and task.box_char == obs_char
                and task.box_char is not None
                and obs_color == State.box_colors[ord(task.box_char) - ord("A")]
                and task.object_pos == (obs_r, obs_c)
            ),
            None,
        )

        obstacle_assigned_to_agent_id = None
        for other_agent in self.agents:

            other_current = self.agents[other_agent.agent_id].task
            if (
                other_current is not None
                and other_current.task_type == "move_box"
                # and other_current.box_char == obs_char
                and other_current.box_char is not None
                and obs_color
                == State.box_colors[ord(other_current.box_char) - ord("A")]
                and other_current.object_pos == (obs_r, obs_c)
            ):
                obstacle_assigned_to_agent_id = other_agent.agent_id
                print(
                    f"Obstacle task for box {obs_color} at ({obs_r}, {obs_c}) is currently assigned to Agent {other_agent.agent_id}.",
                    file=sys.stderr,
                    flush=True,
                )
                break

        # if obstacle_index is None and obstacle_assigned_to_agent_id is None:
        if obstacle_assigned_to_agent_id is None:
            if current_task is None:
                # 1. add the task of clearing the obstacle box as current task
                # 2. check if agent was finihsed, if so then pop back that task to future list of color agents
                new_task = Task(
                    task_type="move_box",
                    object_pos=(obs_r, obs_c),
                    goal_pos=(
                        self.agents[agent_id].agent_row,
                        self.agents[agent_id].agent_col,
                    ),
                    box_char=(
                        state.boxes[obs_r][obs_c]
                        if (state is not None
                            and 0 <= obs_r < len(state.boxes)
                            and 0 <= obs_c < len(state.boxes[obs_r])
                            and state.boxes[obs_r][obs_c])
                        else State.get_box_char_from_color(obs_color)
                    ),
                )

                if obstacle_index is not None:
                    print(
                        f"  Found future task for obstacle box {obs_color} at ({obs_r}, {obs_c}) in color future tasks at index {obstacle_index}.",
                        file=sys.stderr,
                        flush=True,
                    )
                    print(
                        f"Updating that future goal to the goal_pos after the swapping: {new_task.goal_pos}",
                        file=sys.stderr,
                        flush=True,
                    )
                    self.color_tasks[target_color]["future_box_tasks"][
                        obstacle_index
                    ].object_pos = new_task.goal_pos

                agent_color = State.agent_colors[agent_id]
                for idx, task in enumerate(
                    self.color_tasks[agent_color]["solved_tasks"]
                ):
                    if (
                        task.task_type == "move_agent"
                        and task.goal_pos
                        == (
                            self.agents[agent_id].agent_row,
                            self.agents[agent_id].agent_col,
                        )
                        and task.crucial is True
                    ):
                        # move this solved agent task back to future since agent is not actually at its goal yet
                        self.color_tasks[agent_color]["future_agent_tasks"].appendleft(
                            task
                        )
                        del self.color_tasks[agent_color]["solved_tasks"][idx]
                        print(
                            f"  Agent {agent_id} was previously marked as having solved its agent goal, but it is now blocked by an obstacle. Moving its agent goal task back to future tasks.",
                            file=sys.stderr,
                            flush=True,
                        )
                        break
            else:
                new_task = Task(
                    task_type="move_box",
                    object_pos=(
                        (current_task.object_pos)
                        if obstacle_after_box
                        else (obs_r, obs_c)
                    ),
                    goal_pos=(
                        self.agents[agent_id].agent_row,
                        self.agents[agent_id].agent_col,
                    ),
                    box_char=(
                        state.boxes[obs_r][obs_c]
                        if (state is not None
                            and 0 <= obs_r < len(state.boxes)
                            and 0 <= obs_c < len(state.boxes[obs_r])
                            and state.boxes[obs_r][obs_c])
                        else State.get_box_char_from_color(obs_color)
                    ),
                )

                if obstacle_index is not None:
                    print(
                        f"  Found future task for obstacle box {obs_color} at ({obs_r}, {obs_c}) in color future tasks at index {obstacle_index}.",
                        file=sys.stderr,
                        flush=True,
                    )
                    print(
                        f"Updating that future goal to the goal_pos after the swapping: {new_task.goal_pos}",
                        file=sys.stderr,
                        flush=True,
                    )
                    self.color_tasks[target_color]["future_box_tasks"][
                        obstacle_index
                    ].object_pos = new_task.goal_pos

                # # remove failed_task from current tasks and add to front of future tasks

                if obstacle_after_box is True:
                    current_task.object_pos = (obs_r, obs_c)

                if (
                    current_task.task_type == "move_box"
                    and current_task.box_char is not None
                ):
                    self.color_tasks[target_color]["future_box_tasks"].appendleft(
                        current_task
                    )
                elif (
                    current_task.task_type == "move_agent"
                    and current_task.goal_pos is not None
                ):
                    self.color_tasks[target_color]["future_agent_tasks"].appendleft(
                        current_task
                    )
                else:
                    print(
                        f"  Swap failed: failed task is not properly formed for re-queuing.",
                        file=sys.stderr,
                        flush=True,
                    )
                    return False

                # 1. check if the obstacle box of was already in place, if so, then append on the right to future box tasks
                # it needs to be put back as it is
                for idx, task in enumerate(
                    self.color_tasks[target_color]["solved_tasks"]
                ):
                    if (
                        task.task_type == "move_box"
                        and task.box_char is not None
                        and obs_color == State.box_colors[ord(task.box_char) - ord("A")]
                        and task.goal_pos == (obs_r, obs_c)
                        and task.crucial is True
                    ):
                        # move this solved box task back to future since box is not actually at its goal yet
                        self.color_tasks[target_color]["future_box_tasks"].append(task)
                        del self.color_tasks[target_color]["solved_tasks"][idx]
                        print(
                            f"  Obstacle box {obs_color} was previously marked as having solved its box goal, but it is now blocking an agent. Moving its box goal task back to future tasks.",
                            file=sys.stderr,
                            flush=True,
                        )
                        break

            self.agents[agent_id].task = new_task
            self._sync_agent_task_state(agent_id, new_task)

            print(
                f"Swapped tasks, was able to reach box directly: {obstacle_after_box}.",
                file=sys.stderr,
                flush=True,
            )

            # Record same-colour swap in the negotiation log so the cycle
            # detector can refuse repeated swaps over the same box.
            if new_task.box_char is not None:
                self._record_swap(
                    requester=agent_id,
                    granter=agent_id,
                    obstacle_box=new_task.box_char,
                    obstacle_pos=(obs_r, obs_c),
                    original_task=current_task,
                    new_task=new_task,
                    same_color=True,
                )

            return True

        print("Swapping edgecase NOT IMPLEMENTED YET", file=sys.stderr, flush=True)
        return False

    # ------------------------------------------------------------------
    # Main loop (called every turn)
    # ------------------------------------------------------------------

    def get_joint_action(self, joint_state: State) -> list[Action]:
        """
        Execute the next timestep: all agents execute their next planned action.
        """
        self._sync_agent_positions(joint_state)

        # Joint A* fast-path: if Joint A* found a complete plan at setup,
        # just pop the next joint action.
        if self._joint_plan:
            joint_action = self._joint_plan.pop(0)
            self.timestep += 1
            return joint_action

        # Reachability validation: if an agent's current box task points to a
        # box it physically cannot reach (walls separate them, e.g. Hungarian
        # assigned a box in another room), push the task back to the queue so
        # a reachable agent can claim it.
        for agent in self.agents:
            t = agent.task
            if t is None or t.task_type != "move_box" or t.box_char is None:
                continue
            ar, ac = joint_state.agent_rows[agent.agent_id], joint_state.agent_cols[agent.agent_id]
            br, bc = t.object_pos
            if (ar, ac) == (br, bc):
                continue
            # BFS reachability
            walls = State.walls
            rows = len(walls)
            cols = len(walls[0]) if walls else 0
            visited = {(ar, ac)}
            q: deque[tuple[int, int]] = deque([(ar, ac)])
            reachable = False
            while q:
                r, c = q.popleft()
                if (r, c) == (br, bc):
                    reachable = True
                    break
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in visited or not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if walls[nr][nc]:
                        continue
                    visited.add((nr, nc))
                    q.append((nr, nc))
            if not reachable:
                box_color = State.box_colors[ord(t.box_char) - ord("A")]
                if box_color in self.color_tasks:
                    self.color_tasks[box_color]["future_box_tasks"].append(t)
                print(
                    f"  Agent {agent.agent_id}: box {t.box_char} at "
                    f"{t.object_pos} unreachable; returning task to queue.",
                    file=sys.stderr, flush=True,
                )
                agent.task = None
                agent._plan = []
                agent._plan_index = 0

        # End-game restoration: once all box tasks are exhausted across all
        # agents and queues, any agent not at its level-defined agent goal
        # should navigate back to it. Catches agents that were evicted by
        # done-agent step-aside and never made it back to their real goal.
        no_future_box = all(
            not b["future_box_tasks"] for b in self.color_tasks.values()
        )
        no_current_box = all(
            a.task is None or a.task.task_type != "move_box" for a in self.agents
        )
        all_box_tasks_done = no_future_box and no_current_box

        # Final-sweep recovery: queues exhausted but the level is NOT in goal
        # state — some box(es) still off-goal because their task was popped
        # then never properly completed (e.g. stale-task pruning was too
        # eager, or a fresh-task assigned the wrong box). Scan the live state
        # and rebuild a task for any undelivered box, queue it back for a
        # reachable color-compatible agent.
        if all_box_tasks_done and not joint_state.is_goal_state():
            # Multi-instance same-char levels (e.g. duckie, lilchal): each
            # off-goal box can only fill ONE goal. Without this set, the
            # nested loop assigns the first off-goal box to every unsatisfied
            # goal of its char, creating contradictory tasks (same source,
            # different destinations) that the planner thrashes on forever.
            assigned_boxes: set[tuple[int, int]] = set()
            # Also pre-exclude boxes that an agent is already committed to.
            already_committed: set[tuple[int, int]] = set()
            for a in self.agents:
                if (a.task is not None
                    and a.task.task_type == "move_box"
                    and a.task.box_char is not None):
                    already_committed.add(a.task.object_pos)
            for gr in range(len(State.goals)):
                for gc in range(len(State.goals[gr])):
                    goal_ch = State.goals[gr][gc]
                    if not goal_ch or not goal_ch.isupper():
                        continue
                    if joint_state.boxes[gr][gc] == goal_ch:
                        continue  # goal already satisfied
                    # Find an UNCLAIMED off-goal box of this char.
                    box_pos = None
                    for r in range(len(joint_state.boxes)):
                        for c in range(len(joint_state.boxes[r])):
                            if (joint_state.boxes[r][c] == goal_ch
                                and State.goals[r][c] != goal_ch
                                and (r, c) not in assigned_boxes
                                and (r, c) not in already_committed):
                                box_pos = (r, c)
                                break
                        if box_pos is not None:
                            break
                    if box_pos is None:
                        continue
                    assigned_boxes.add(box_pos)
                    box_color = State.box_colors[ord(goal_ch) - ord("A")]
                    if box_color not in self.color_tasks:
                        continue
                    # Don't duplicate — only enqueue if no equivalent already pending
                    bucket = self.color_tasks[box_color]
                    already = any(
                        t.box_char == goal_ch and t.object_pos == box_pos
                        and t.goal_pos == (gr, gc)
                        for t in bucket["future_box_tasks"]
                    ) or any(
                        a.task is not None and a.task.box_char == goal_ch
                        and a.task.goal_pos == (gr, gc)
                        for a in self.agents
                    )
                    if already:
                        continue
                    new_task = Task(
                        task_type="move_box",
                        object_pos=box_pos,
                        goal_pos=(gr, gc),
                        box_char=goal_ch,
                        crucial=True,
                    )
                    bucket["future_box_tasks"].append(new_task)
                    print(
                        f"  Final-sweep: rebuilt task {goal_ch} {box_pos} → "
                        f"({gr},{gc}).",
                        file=sys.stderr, flush=True,
                    )
                    # Force the flag to false so end-game doesn't fire
                    all_box_tasks_done = False
                    no_current_box = False  # not strictly needed but keeps state consistent
        level_agent_goals = {
            aid: (gr, gc)
            for gr, gc, aid in (self.profile.agent_goals if self.profile else [])
        }
        if all_box_tasks_done and self.timestep % 5 == 0:
            print(
                f"  ENDGAME at t={self.timestep}: all box tasks done; "
                f"level_agent_goals={level_agent_goals}",
                file=sys.stderr, flush=True,
            )
        # BDI-style task completion handling: move solved tasks aside, then
        # assign and replan if another task is available.
        t = self.timestep
        for agent in self.agents:
            # Restore _pending_agent_goal: when a done agent was stepped aside
            # by conflict-resolution eviction, its real goal was saved here.
            if (
                agent.task is None
                and agent.agent_goal is not None
                and (agent.agent_row, agent.agent_col) == agent.agent_goal
                and agent._pending_agent_goal is not None
            ):
                agent.agent_goal = agent._pending_agent_goal
                agent._pending_agent_goal = None
                gr, gc = agent.agent_goal
                if (agent.agent_row, agent.agent_col) != (gr, gc):
                    back_plan = solve(
                        state=joint_state,
                        agent_id=agent.agent_id,
                        goal=(None, None, gr, gc),
                        constraints=set(),
                        dist_map=self.dist_map,
                    )
                    if back_plan:
                        agent._plan = list(back_plan)
                        agent._plan_index = 0
                        print(
                            f"  Agent {agent.agent_id}: returning to real goal "
                            f"({gr},{gc}), plan len {len(back_plan)}.",
                            file=sys.stderr, flush=True,
                        )

            # End-game cleanup: agent not at its level-defined goal, no
            # remaining plan actions, all deliveries done -> plan a path back.
            no_remaining_plan = agent._plan_index >= len(agent._plan)
            if (
                all_box_tasks_done
                and agent.agent_id in level_agent_goals
                and no_remaining_plan
            ):
                gr, gc = level_agent_goals[agent.agent_id]
                if (agent.agent_row, agent.agent_col) != (gr, gc):
                    back_plan = solve(
                        state=joint_state,
                        agent_id=agent.agent_id,
                        goal=(None, None, gr, gc),
                        constraints=set(),
                        dist_map=self.dist_map,
                    )
                    if back_plan:
                        agent.agent_goal = (gr, gc)
                        agent._pending_agent_goal = None
                        agent._plan = list(back_plan)
                        agent._plan_index = 0
                        agent.abs_constraints = set()
                        agent._noop_until = -1
                        print(
                            f"  Agent {agent.agent_id}: end-game return to "
                            f"({gr},{gc}), plan len {len(back_plan)}.",
                            file=sys.stderr, flush=True,
                        )

            if agent.task is None:
                agent.task = self._pop_next_task_for_agent(agent.agent_id, joint_state)
                if agent.task is not None:
                    self._sync_agent_task_state(agent.agent_id, agent.task)

            self._maybe_advance_completed_task_or_preplan(joint_state, agent.agent_id)
        joint_action = []
        for agent in self.agents:
            # Forced-NoOp window from conflict resolution: hold the agent in
            # place until _noop_until is reached, so the conflict winner has
            # time to clear the corridor.
            if getattr(agent, "_noop_until", -1) > self.timestep:
                joint_action.append(Action.NoOp)
                continue
            action = agent.next_action()
            joint_action.append(action)

        # Stuck-agent recovery: agents that haven't been able to plan for
        # STUCK_THRESHOLD consecutive turns are given a single-turn cardinal
        # nudge from `_pick_recovery_move`. The nudge rotates through
        # directions deterministically (no RNG), so consecutive triggers for
        # the same agent always try different directions.
        STUCK_THRESHOLD = 15
        for agent in self.agents:
            if len(agent._plan) > 0 or agent.task is None:
                self.agents_no_plan_cnt[agent.agent_id] = 0
                continue
            self.agents_no_plan_cnt[agent.agent_id] += 1
            if self.agents_no_plan_cnt[agent.agent_id] >= STUCK_THRESHOLD:
                self.agents_no_plan_cnt[agent.agent_id] = 0
                chosen = self._pick_recovery_move(joint_state, agent.agent_id)
                if chosen is not None:
                    joint_action[agent.agent_id] = chosen
                    print(
                        f"  Agent {agent.agent_id}: stuck for {STUCK_THRESHOLD} turns, "
                        f"nudge {chosen.name_}.",
                        file=sys.stderr, flush=True,
                    )

        # Validate joint action and receive offending agents (if any)
        is_valid, invalid_agents = self.is_joint_action_valid(joint_state, joint_action)
        print(
            f"Joint action valid: {is_valid}; invalid agents: {invalid_agents}",
            file=sys.stderr,
            flush=True,
        )

        if not is_valid:
            # CBS-style conflict resolution: asymmetric yield. The agent with
            # the SHORTEST remaining plan wins — it finishes first and clears
            # the contested corridor. Tiebreak: lowest agent_id. All other
            # conflicting agents NoOp + take a search-time-1 constraint.
            # Requirement: the winner's action must be individually applicable;
            # otherwise the server rejects and the client state drifts.
            invalid_set = set(invalid_agents)
            def remaining(aid: int) -> int:
                ag = self.agents[aid]
                return max(0, len(ag._plan) - ag._plan_index)

            # PRE-STEP: any "done" agent (no plan, no task) in the conflict is
            # almost certainly just sitting in the corridor blocking work. Its
            # NoOp is always applicable, so the standard winner-selection picks
            # it as winner — and the actually-working agent yields forever.
            # Compute and STUFF a real plan into the done agent's _plan so it
            # actually moves aside (BDI won't preplan for a taskless agent).
            level_agent_goal_set = {
                aid: (gr, gc)
                for gr, gc, aid in (self.profile.agent_goals if self.profile else [])
            }
            for aid in list(invalid_set):
                ag = self.agents[aid]
                if remaining(aid) != 0 or ag.task is not None:
                    continue
                # Avoid stepping onto other agents' level goals + the cells
                # adjacent to them (those are likely on the other agent's
                # final approach path).
                avoid: set[tuple[int, int]] = set()
                for other_aid, (gr, gc) in level_agent_goal_set.items():
                    if other_aid == aid:
                        continue
                    avoid.add((gr, gc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        avoid.add((gr + dr, gc + dc))
                # _step_aside_cell is pending-goal-aware (it never parks the
                # agent on a cell that's a pending box-goal target).
                # _find_escape_cell is NOT — using it first risks parking the
                # agent on the cell where the next box needs to go, deadlocking
                # the rest of the level. So try step-aside first, only fall
                # back to far-escape if no safe parking exists nearby.
                escape = self._step_aside_cell(
                    joint_state, aid, min_dist=3, avoid_cells=avoid
                )
                if escape is None or escape == (ag.agent_row, ag.agent_col):
                    escape = self._step_aside_cell(
                        joint_state, aid, avoid_cells=avoid
                    )
                if escape is None or escape == (ag.agent_row, ag.agent_col):
                    escape = self._step_aside_cell(joint_state, aid)
                if escape is None or escape == (ag.agent_row, ag.agent_col):
                    continue
                # Plan A* from current pos → escape cell, install directly.
                escape_plan = solve(
                    state=joint_state,
                    agent_id=aid,
                    goal=(None, None, escape[0], escape[1]),
                    constraints=set(),
                    dist_map=self.dist_map,
                )
                if not escape_plan:
                    continue
                if ag.agent_goal is not None and ag.agent_goal != escape:
                    ag._pending_agent_goal = ag.agent_goal
                ag.agent_goal = escape
                ag.constraints = set()
                ag.abs_constraints = set()
                ag._plan = list(escape_plan)
                ag._plan_index = 0
                ag._noop_until = -1
                # Overwrite this turn's action with the first step of the
                # escape plan so the eviction takes effect immediately.
                if escape_plan and joint_state.is_applicable(aid, escape_plan[0]):
                    joint_action[aid] = escape_plan[0]
                    ag._plan_index = 1
                print(
                    f"    Done-agent {aid} blocking at ({ag.agent_row},"
                    f"{ag.agent_col}): A*-escape to {escape} "
                    f"(plan len {len(escape_plan)}).",
                    file=sys.stderr, flush=True,
                )

            # Re-validate after PRE-STEP eviction. If the evictions resolved
            # the conflict, skip the rest of conflict resolution — otherwise
            # it would clear the eviction plans we just installed.
            is_valid_now, invalid_now = self.is_joint_action_valid(
                joint_state, joint_action
            )
            if is_valid_now:
                print(
                    f"    PRE-STEP eviction resolved conflict.",
                    file=sys.stderr, flush=True,
                )
                invalid_set = set()
                invalid_agents = []
            else:
                invalid_set = set(invalid_now)
                invalid_agents = invalid_now

            # An agent that's "done" is the WORST winner — push done agents to
            # the back of the priority queue.
            def priority(aid: int) -> tuple[int, int, int]:
                ag = self.agents[aid]
                rem = remaining(aid)
                is_done = 1 if (rem == 0 and ag.task is None) else 0
                return (is_done, rem, aid)
            ordered = sorted(invalid_set, key=priority)
            winner = None
            for aid in ordered:
                if joint_state.is_applicable(aid, joint_action[aid]):
                    winner = aid
                    break
            yielding = invalid_set - ({winner} if winner is not None else set())

            # If the winner has a meaningfully shorter plan than the loser(s),
            # the loser will NoOp for several turns instead of immediately
            # replanning. This gives the winner room to actually clear the
            # corridor rather than oscillating one step at a time. The hold
            # count is half the gap, capped at 10 turns.
            if winner is not None and yielding:
                win_rem = remaining(winner)
                for aid in yielding:
                    gap = remaining(aid) - win_rem
                    if gap > 4:
                        hold = min(10, gap // 2)
                        self.agents[aid]._noop_until = self.timestep + hold
            joint_action = [
                action if agent.agent_id not in yielding else Action.NoOp
                for agent, action in zip(self.agents, joint_action, strict=True)
            ]
            for aid in yielding:
                agent = self.agents[aid]
                # Compute the cell the agent was trying to enter
                bad_action = joint_action[aid] if aid not in invalid_agents else None
                # Use the action they attempted (before we set to NoOp)
                attempted_actions = list(joint_action)
                # Recompute attempted destination from agent's _plan
                if agent._plan_index > 0 and agent._plan_index - 1 < len(agent._plan):
                    attempted = agent._plan[agent._plan_index - 1]
                    ar = joint_state.agent_rows[aid]
                    ac = joint_state.agent_cols[aid]
                    # ABSOLUTE-TIME constraint at global timestep T+2.
                    # Why +2: on the next planning turn (global T+1), the agent
                    # is at its current position. A* search-time 0 is "now",
                    # search-time 1 is "after first action". An abs constraint
                    # at T+2 translates to local-time 1 on turn T+1, which
                    # correctly blocks the agent's first move into the bad
                    # cell. (T+1 would translate to local-time 0 — the start
                    # state, which the agent isn't at, so it'd be a no-op.)
                    abs_t = self.timestep + 2
                    if attempted.type is ActionType.Move:
                        bad_r = ar + attempted.agent_row_delta
                        bad_c = ac + attempted.agent_col_delta
                        agent.abs_constraints.add((bad_r, bad_c, abs_t))
                    elif attempted.type is ActionType.Push:
                        # Box destination
                        br = ar + attempted.agent_row_delta
                        bc = ac + attempted.agent_col_delta
                        bdr = br + attempted.box_row_delta
                        bdc = bc + attempted.box_col_delta
                        agent.abs_constraints.add((bdr, bdc, abs_t))
                        agent.abs_constraints.add((br, bc, abs_t))
                    elif attempted.type is ActionType.Pull:
                        nr = ar + attempted.agent_row_delta
                        nc = ac + attempted.agent_col_delta
                        agent.abs_constraints.add((nr, nc, abs_t))
                # Clear plan to force replan with new constraint
                agent._plan = []
                agent._plan_index = 0
                # Cap constraint sets to avoid unbounded growth.
                if len(agent.constraints) > 200:
                    agent.constraints = set(list(agent.constraints)[-100:])
                # Drop abs_constraints that are in the past (already expired)
                # and cap at 200.
                agent.abs_constraints = {
                    c for c in agent.abs_constraints if c[2] >= self.timestep
                }
                if len(agent.abs_constraints) > 200:
                    agent.abs_constraints = set(list(agent.abs_constraints)[-100:])

        # Track invalid-agent set per turn for oscillation detection.
        self._invalid_agent_history.append(
            frozenset(invalid_agents) if not is_valid else frozenset()
        )

        # Oscillation recovery via runtime CBS. Runs BEFORE joint-state cycle
        # detection because alternating-conflict-set oscillation is detectable
        # much sooner (6-8 turns) than full state cycling (40+ turns).
        self._maybe_cbs_recovery(joint_state)

        # Deadlock detection — if we see the same joint state cycling N+ times,
        # trigger recovery: assign escape cells, clear plans, drop awaiting flags.
        self._detect_and_resolve_deadlock(joint_state)

        # Periodic constraint pruning to keep A* fast on long runs.
        if self.timestep % 50 == 0 and self.timestep > 0:
            for agent in self.agents:
                agent.clear_old_constraints(self.timestep)
                if len(agent.constraints) > 300:
                    agent.constraints = set(list(agent.constraints)[-150:])

        self.timestep += 1
        return joint_action

    def _detect_and_resolve_deadlock(self, joint_state: State) -> None:
        """Track recent joint states; if cycling detected, force recovery."""
        # Build a compact signature of agent + box positions
        sig = (
            tuple(joint_state.agent_rows),
            tuple(joint_state.agent_cols),
            tuple(tuple(row) for row in joint_state.boxes),
        )
        self._state_history.append(sig)

        # Cooldown — don't re-trigger every turn. Use a longer cooldown to
        # give the newer oscillation-CBS / done-agent recovery time to work
        # before the older escape-cell mechanism kicks in (the two can fight).
        if self.timestep - self._deadlock_resolved_at < 30:
            return

        # Count occurrences of current sig in the rolling window.
        if len(self._state_history) < 10:
            return
        occurrences = sum(1 for s in self._state_history if s == sig)
        if occurrences < 5:
            return  # not enough repetition to be a cycle

        # We're cycling. Recovery: pick worst-stuck agents (no plan, not at goal),
        # assign each an escape cell, clear their constraints and awaiting flags.
        print(
            f"  Deadlock detected at t={self.timestep} (state seen {occurrences}x). Triggering recovery.",
            file=sys.stderr, flush=True,
        )
        self._deadlock_resolved_at = self.timestep

        stuck = []
        for agent in self.agents:
            has_goal = agent.task is not None or agent.agent_goal is not None or getattr(agent, "tasks", None)
            if has_goal and not agent._plan:
                stuck.append(agent)

        for agent in stuck[:3]:  # only force-escape up to 3 agents to avoid chaos
            escape = self._find_escape_cell(joint_state, agent.agent_id)
            if escape is None:
                continue
            print(
                f"    Agent {agent.agent_id}: forced escape to {escape}.",
                file=sys.stderr, flush=True,
            )
            # Save current positional goal if any, restore after escape
            if agent.agent_goal is not None:
                agent._pending_agent_goal = agent.agent_goal
            agent.agent_goal = escape
            agent.constraints = set()  # drop stale constraints
            agent._plan = []
            agent._plan_index = 0
            self.agents_awaiting_other_agent[agent.agent_id] = None
            agent.awaiting_cnt = 0

    def _maybe_cbs_recovery(self, joint_state: State) -> None:
        """Detect oscillating conflicts and run runtime CBS over all agents
        with open tasks. Triggered when invalid-action sets persist for several
        recent turns, regardless of whether they're identical (per-turn replan
        keeps shifting which agents conflict). Much earlier signal than full
        joint-state cycling."""
        # Cooldown — CBS is expensive; don't re-run constantly.
        if self.timestep - self._cbs_recovered_at < 25:
            return
        if len(self._invalid_agent_history) < 6:
            return

        recent = list(self._invalid_agent_history)[-10:]
        nonempty = [s for s in recent if s]
        # Need persistent invalidity. Lowered to 3 to catch 2-agent alternating
        # patterns where every other turn is valid (4 invalid in 8 vs. our old
        # ≥5 threshold meant we never triggered).
        if len(nonempty) < 3:
            return

        # Require a repeated set — a single agent_id appearing once is noise,
        # but the same set appearing 3+ times in the window is a real cycle.
        from collections import Counter
        set_counts = Counter(nonempty)
        most_common_set, most_common_count = set_counts.most_common(1)[0]
        if most_common_count < 3:
            return

        # Union of all agents seen as invalid recently — they're all suspect.
        suspects: set[int] = set()
        for s in nonempty:
            suspects |= s
        if len(suspects) < 2:
            return

        print(
            f"  Oscillation detected at t={self.timestep} "
            f"(invalid sets {[sorted(s) for s in nonempty]}); "
            f"running runtime CBS recovery on suspects {sorted(suspects)}.",
            file=sys.stderr, flush=True,
        )
        self._cbs_recovered_at = self.timestep

        from searchclient.planner.cbs import anytime_cbs_solve

        # Build agent_goals from CURRENT positions and CURRENT tasks. Only for
        # SUSPECTS — including all 10 agents made CBS branching explode. Other
        # agents continue with their existing plans; instantaneous collisions
        # are caught next turn by is_joint_action_valid and conflict resolution.
        agent_goals: dict[int, tuple] = {}
        for aid in suspects:
            agent = self.agents[aid]
            task = agent.task
            if task is None:
                if agent.agent_goal is not None:
                    gr, gc = agent.agent_goal
                    agent_goals[aid] = (None, None, gr, gc)
                continue

            if task.task_type == "move_box":
                box_pos = self._find_box_current_pos(
                    joint_state, task.box_char, task.goal_pos[0], task.goal_pos[1]
                )
                if box_pos is None:
                    continue
                br, bc = box_pos
                agent_goals[aid] = (
                    br, bc, task.goal_pos[0], task.goal_pos[1], task.box_char,
                )
            else:  # move_agent
                agent_goals[aid] = (
                    None, None, task.goal_pos[0], task.goal_pos[1],
                )

        if len(agent_goals) < 2:
            print("  CBS recovery: <2 plannable agents, skipping.",
                  file=sys.stderr, flush=True)
            return

        # For ≤2 agents, CBS over long plans is too slow (each CBS node is one
        # A* call on a possibly 60+ step plan). Go straight to spatial
        # avoidance — ban the winner's cell zone in the loser's constraints
        # so A* finds an alternate route.
        if len(agent_goals) <= 2:
            print(
                "  ≤2 suspects: skipping CBS, route diversion directly.",
                file=sys.stderr, flush=True,
            )
            self._sequential_routing_recovery(suspects, joint_state)
            return

        # 3+ agents: CBS is genuinely valuable (multi-way coordination).
        # Scale caps with agent count.
        if len(agent_goals) <= 4:
            caps = (100, 500, 2000)
            budget = 5.0
        else:
            caps = (50, 200, 800)
            budget = 3.0
        plans = anytime_cbs_solve(
            state=joint_state,
            agent_goals=agent_goals,
            dist_map=self.dist_map,
            time_budget_s=budget,
            cap_schedule=caps,
        )

        if plans is None:
            print(
                "  CBS recovery failed — falling back to sequential routing.",
                file=sys.stderr, flush=True,
            )
            self._sequential_routing_recovery(suspects, joint_state)
            return

        # Install plans. Clear runtime constraints and noop_until so the agents
        # actually execute the CBS plan instead of being held.
        for aid, plan in plans.items():
            agent = self.agents[aid]
            agent._plan = list(plan)
            agent._plan_index = 0
            agent._noop_until = -1
            agent.abs_constraints = set()
            self.agents_awaiting_other_agent[aid] = None
            agent.awaiting_cnt = 0
            print(
                f"    CBS recovery: Agent {aid} new plan length {len(plan)}.",
                file=sys.stderr, flush=True,
            )

        # Wipe invalid history so we don't re-trigger immediately on stale data.
        self._invalid_agent_history.clear()

    def _fresh_same_color_box_task(self, agent_id: int, joint_state: State) -> bool:
        """Rebuild a clean box task from current world state. Called when the
        same-color cycle refuses further swaps because earlier swap rounds
        corrupted the task's (object_pos, goal_pos) fields. We scan the live
        boxes grid for an undelivered same-color box and pair it with a free
        same-char goal. Returns True if a fresh task was assigned."""
        agent_color = State.agent_colors[agent_id]
        if agent_color is None:
            return False

        # Find every box of this color that is NOT already at a goal of its
        # own char. Pair with whichever same-char goal is still vacant.
        ar = self.agents[agent_id].agent_row
        ac = self.agents[agent_id].agent_col
        candidates: list[tuple[int, int, int, str, tuple[int, int]]] = []
        for r in range(len(joint_state.boxes)):
            for c in range(len(joint_state.boxes[r])):
                ch = joint_state.boxes[r][c]
                if not ch:
                    continue
                if State.box_colors[ord(ch) - ord("A")] != agent_color:
                    continue
                if State.goals[r][c] == ch:
                    continue  # already at one of its goals
                # Find a goal for this char that doesn't yet hold a same-char box.
                free_goal: tuple[int, int] | None = None
                for gr in range(len(State.goals)):
                    for gc in range(len(State.goals[gr])):
                        if State.goals[gr][gc] != ch:
                            continue
                        if joint_state.boxes[gr][gc] == ch:
                            continue  # this goal already satisfied
                        free_goal = (gr, gc)
                        break
                    if free_goal is not None:
                        break
                if free_goal is None:
                    continue
                dist = abs(r - ar) + abs(c - ac)
                candidates.append((dist, r, c, ch, free_goal))

        if not candidates:
            print(
                f"    Fresh-task recovery: no undelivered same-color box left "
                f"for Agent {agent_id}.",
                file=sys.stderr, flush=True,
            )
            return False

        candidates.sort()
        _, br, bc, box_char, (gr, gc) = candidates[0]
        new_task = Task(
            task_type="move_box",
            object_pos=(br, bc),
            goal_pos=(gr, gc),
            box_char=box_char,
            crucial=True,
        )
        agent = self.agents[agent_id]
        agent.task = new_task
        self._sync_agent_task_state(agent_id, new_task)
        # Clear stale plan + constraints so A* starts from scratch.
        agent.constraints = set()
        agent.abs_constraints = set()
        agent._plan = []
        agent._plan_index = 0
        agent._noop_until = -1
        self.agents_awaiting_other_agent[agent_id] = None
        agent.awaiting_cnt = 0
        print(
            f"    Fresh-task recovery: Agent {agent_id} reassigned box "
            f"{box_char} ({br},{bc}) → ({gr},{gc}).",
            file=sys.stderr, flush=True,
        )
        return True

    def _sequential_routing_recovery(self, suspects: set[int], joint_state: State) -> None:
        """Route-diversion fallback when CBS can't reconcile oscillating agents:
        pick the agent with the shortest remaining plan as 'winner' (least
        invested in detouring), then ban the winner's 3x3 cell zone in every
        loser's abs_constraints for many future timesteps. Clearing the
        losers' plans forces A* to find an alternate route (e.g. the southern
        corridor instead of fighting through the winner's lane)."""
        plannable = [
            (len(self.agents[aid]._plan) - self.agents[aid]._plan_index, aid)
            for aid in suspects
            if len(self.agents[aid]._plan) > 0
        ]
        if not plannable:
            return
        plannable.sort()
        winner_rem, winner_id = plannable[0]
        winner = self.agents[winner_id]
        wr, wc = winner.agent_row, winner.agent_col
        # Ban window — long enough for the winner to actually clear the area.
        ban_dt = max(40, min(120, winner_rem + 15))

        diverted: list[int] = []
        for aid in suspects:
            if aid == winner_id:
                continue
            loser = self.agents[aid]
            # Ban the 3x3 zone around the winner for the ban window. A* will
            # find a path that avoids this zone over the next ban_dt turns,
            # which on TwoPlayer-style layouts means routing through the
            # parallel corridor.
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    br, bc = wr + dr, wc + dc
                    for dt in range(1, ban_dt + 1):
                        loser.abs_constraints.add((br, bc, self.timestep + dt))
            # Clear plan so loser re-plans with the new bans applied.
            loser._plan = []
            loser._plan_index = 0
            loser._noop_until = -1  # actively move, don't sit still
            self.agents_awaiting_other_agent[aid] = None
            loser.awaiting_cnt = 0
            diverted.append(aid)
        print(
            f"    Route diversion: winner=Agent {winner_id} (rem={winner_rem}, "
            f"at ({wr},{wc})); diverted {sorted(diverted)} with {ban_dt}-turn "
            f"ban around 3x3 zone.",
            file=sys.stderr, flush=True,
        )

    def _sync_agent_positions(self, joint_state: State) -> None:
        """Cache the latest row/col position for each agent."""
        for agent in self.agents:
            agent.update_position(
                joint_state.agent_rows[agent.agent_id],
                joint_state.agent_cols[agent.agent_id],
            )

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def is_done(self, joint_state: State) -> bool:
        return joint_state.is_goal_state()

    # ------------------------------------------------------------------
    # HCA* — Hierarchical Cooperative A* (ported from felix/structure)
    # ------------------------------------------------------------------

    def _hca_preplan(self, initial_state: State) -> None:
        """Two-pass HCA*. Pass 1: plan ALL tasks per agent in priority order.
        Each agent's full trajectory becomes constraints for lower-priority agents.
        Pass 2: retry failed agents with projected-state (other boxes removed)."""
        # ---- Pass 1 ----
        for agent in self.agents:
            if not getattr(agent, "tasks", None):
                if agent.agent_goal is not None:
                    gr, gc = agent.agent_goal
                    plan = solve(
                        state=initial_state,
                        agent_id=agent.agent_id,
                        goal=(None, None, gr, gc),
                        constraints=agent.constraints,
                        dist_map=self.dist_map,
                    )
                    if plan:
                        agent._plan = plan
                        agent._plan_index = 0
                        self._register_agent_path(
                            agent, initial_state,
                            initial_state.agent_rows[agent.agent_id],
                            initial_state.agent_cols[agent.agent_id],
                            -1, -1, plan,
                        )
                continue

            full_plan, success = self._plan_all_tasks(agent, initial_state, 0)
            if not success or not full_plan:
                print(
                    f"  HCA* pass 1: Agent {agent.agent_id} FAILED.",
                    file=sys.stderr, flush=True,
                )
                continue

            agent._plan = full_plan
            agent._plan_index = 0
            task0 = agent.tasks[0]
            box_r0, box_c0 = agent._find_box(initial_state, task0[4], task0[2], task0[3])
            if box_r0 is None:
                box_r0, box_c0 = -1, -1
            self._register_agent_path(
                agent, initial_state,
                initial_state.agent_rows[agent.agent_id],
                initial_state.agent_cols[agent.agent_id],
                box_r0, box_c0, full_plan,
            )

        # ---- Pass 2 ----
        self._hca_retry_failed(initial_state)

    def _plan_all_tasks(
        self, agent: Agent, initial_state: State, t_base_offset: int
    ) -> tuple[list[Action], bool]:
        """Plan all of an agent's tasks sequentially, chaining simulated states.
        Also chains agent_goal navigation when ≤1 box task."""
        full_plan: list[Action] = []
        current_state = initial_state
        t_offset = t_base_offset

        for task in agent.tasks:
            goal_r, goal_c, box_char = task[2], task[3], task[4]
            box_r, box_c = agent._find_box(current_state, box_char, goal_r, goal_c)
            if box_r is None:
                continue
            shifted = {(r, c, t - t_offset) for r, c, t in agent.constraints if t >= t_offset}
            task_plan = solve(
                state=current_state,
                agent_id=agent.agent_id,
                goal=(box_r, box_c, goal_r, goal_c, box_char),
                constraints=shifted,
                dist_map=self.dist_map,
            )
            if task_plan is None:
                return full_plan, False
            full_plan.extend(task_plan)
            t_offset += len(task_plan)
            current_state = _apply_agent_plan(current_state, agent.agent_id, task_plan)

        # Chain nav to agent_goal if exactly ≤1 box task
        if agent.agent_goal is not None and len(agent.tasks) <= 1:
            gr, gc = agent.agent_goal
            ar = current_state.agent_rows[agent.agent_id]
            ac = current_state.agent_cols[agent.agent_id]
            if not (ar == gr and ac == gc):
                shifted = {(r, c, t - t_offset) for r, c, t in agent.constraints if t >= t_offset}
                nav_plan = solve(
                    state=current_state,
                    agent_id=agent.agent_id,
                    goal=(None, None, gr, gc),
                    constraints=shifted,
                    dist_map=self.dist_map,
                )
                if nav_plan is None:
                    return full_plan, False
                full_plan.extend(nav_plan)

        return full_plan, True

    def _register_agent_path_one(
        self, ar: int, ac: int, bxr: int, bxc: int,
        plan: list[Action], target_agent: Agent,
    ) -> None:
        """Register a single agent's trajectory as constraints for ONE target agent."""
        prev_ar, prev_ac = ar, ac
        for t, action in enumerate(plan):
            target_agent.constraints.add((ar, ac, t))
            if bxr >= 0:
                target_agent.constraints.add((bxr, bxc, t))
            if t > 0:
                target_agent.constraints.add((prev_ar, prev_ac, t))
            prev_ar, prev_ac = ar, ac
            if action.type is ActionType.Move:
                ar += action.agent_row_delta
                ac += action.agent_col_delta
            elif action.type is ActionType.Push:
                new_bxr = bxr + action.box_row_delta
                new_bxc = bxc + action.box_col_delta
                ar, ac = bxr, bxc
                bxr, bxc = new_bxr, new_bxc
            elif action.type is ActionType.Pull:
                new_bxr, new_bxc = ar, ac
                ar = ar + action.agent_row_delta
                ac = ac + action.agent_col_delta
                bxr, bxc = new_bxr, new_bxc
        target_agent.constraints.add((ar, ac, len(plan)))
        if bxr >= 0:
            target_agent.constraints.add((bxr, bxc, len(plan)))

    def _register_agent_path(
        self, agent: Agent, state: State, ar: int, ac: int,
        bxr: int, bxc: int, plan: list[Action],
    ) -> None:
        """Add time-indexed agent and box positions as constraints for
        all lower-priority agents (agent_id > this agent's id)."""
        lower = [a for a in self.agents if a.agent_id > agent.agent_id]
        prev_ar, prev_ac = ar, ac
        for t, action in enumerate(plan):
            for other in lower:
                other.constraints.add((ar, ac, t))
                if bxr >= 0:
                    other.constraints.add((bxr, bxc, t))
                if t > 0:
                    other.constraints.add((prev_ar, prev_ac, t))
            prev_ar, prev_ac = ar, ac
            if action.type is ActionType.Move:
                ar += action.agent_row_delta
                ac += action.agent_col_delta
            elif action.type is ActionType.Push:
                new_bxr = bxr + action.box_row_delta
                new_bxc = bxc + action.box_col_delta
                ar, ac = bxr, bxc
                bxr, bxc = new_bxr, new_bxc
            elif action.type is ActionType.Pull:
                new_bxr, new_bxc = ar, ac
                ar = ar + action.agent_row_delta
                ac = ac + action.agent_col_delta
                bxr, bxc = new_bxr, new_bxc
        for other in lower:
            other.constraints.add((ar, ac, len(plan)))
            if bxr >= 0:
                other.constraints.add((bxr, bxc, len(plan)))

    def _compute_box_trajectory(
        self, initial_state: State, agent_id: int,
        box_r: int, box_c: int, plan: list[Action],
    ) -> list[tuple[int, int, int]]:
        """Return [(r, c, t)] — box position at each timestep during plan."""
        trajectory: list[tuple[int, int, int]] = []
        ar = initial_state.agent_rows[agent_id]
        ac = initial_state.agent_cols[agent_id]
        bxr, bxc = box_r, box_c
        for t, action in enumerate(plan):
            trajectory.append((bxr, bxc, t))
            if action.type is ActionType.Push:
                new_bxr = bxr + action.box_row_delta
                new_bxc = bxc + action.box_col_delta
                ar, ac = bxr, bxc
                bxr, bxc = new_bxr, new_bxc
            elif action.type is ActionType.Pull:
                new_bxr, new_bxc = ar, ac
                ar = ar + action.agent_row_delta
                ac = ac + action.agent_col_delta
                bxr, bxc = new_bxr, new_bxc
            elif action.type is ActionType.Move:
                ar += action.agent_row_delta
                ac += action.agent_col_delta
        trajectory.append((bxr, bxc, len(plan)))
        return trajectory

    def _hca_retry_failed(self, initial_state: State) -> None:
        """Pass 2 of HCA*: retry agents that failed pass 1, using a projected
        state where other agents' boxes are REMOVED, plus:
          - time-indexed box-trajectory constraints (wait until cell clears)
          - permanent constraints for the goal position after plan ends
        """
        needs_retry = [
            a for a in self.agents
            if (getattr(a, "tasks", None) or a.agent_goal is not None) and not a._plan
        ]
        if not needs_retry:
            return

        for agent in needs_retry:
            proj_boxes = [row[:] for row in initial_state.boxes]
            extra_constraints: set[tuple[int, int, int]] = set()

            for other in self.agents:
                if other.agent_id == agent.agent_id:
                    continue
                if not other._plan or not getattr(other, "tasks", None):
                    continue
                other_task = other.tasks[0]
                o_goal_r, o_goal_c, o_char = other_task[2], other_task[3], other_task[4]
                o_box_r, o_box_c = agent._find_box(initial_state, o_char, o_goal_r, o_goal_c)
                if o_box_r is None:
                    continue

                # Time-indexed box-trajectory: wait for box to vacate each cell
                traj = self._compute_box_trajectory(
                    initial_state, other.agent_id, o_box_r, o_box_c, other._plan,
                )
                extra_constraints.update(traj)

                # Remove box from initial static grid
                if 0 <= o_box_r < len(proj_boxes) and 0 <= o_box_c < len(proj_boxes[0]):
                    proj_boxes[o_box_r][o_box_c] = ""

                # Permanent constraints: after the other agent's plan ends,
                # the box stays at the goal cell.
                for t_fut in range(len(other._plan), _MAX_FUTURE):
                    extra_constraints.add((o_goal_r, o_goal_c, t_fut))

            proj_state = State(
                initial_state.agent_rows[:],
                initial_state.agent_cols[:],
                proj_boxes,
            )

            # Plan with combined constraints (agent's own + extra trajectory + permanents)
            saved = agent.constraints
            agent.constraints = agent.constraints | extra_constraints

            if getattr(agent, "tasks", None):
                full_plan, success = self._plan_all_tasks(agent, proj_state, 0)
            elif agent.agent_goal is not None:
                gr, gc = agent.agent_goal
                plan = solve(
                    state=proj_state,
                    agent_id=agent.agent_id,
                    goal=(None, None, gr, gc),
                    constraints=agent.constraints,
                    dist_map=self.dist_map,
                )
                full_plan = plan if plan else []
                success = plan is not None
            else:
                full_plan, success = [], False

            agent.constraints = saved  # restore

            if success and full_plan:
                agent._plan = full_plan
                agent._plan_index = 0
                print(
                    f"  HCA* pass 2: Agent {agent.agent_id} succeeded (length {len(full_plan)}).",
                    file=sys.stderr, flush=True,
                )
            else:
                print(
                    f"  HCA* pass 2: Agent {agent.agent_id} FAILED.",
                    file=sys.stderr, flush=True,
                )

    # ------------------------------------------------------------------
    # Joint A* (ported from felix/structure — optimal for ≤2 agent levels)
    # ------------------------------------------------------------------

    def _joint_astar_plan(
        self, initial_state: State, time_budget_s: float = 5.0,
    ) -> list[list[Action]] | None:
        """A* over the joint state space — optimal but only tractable for small levels.

        Bounded by both a state cap AND a wall-clock budget so we can't hang
        forever on levels where the joint space is large but the gating
        heuristics let it through (e.g. 2 agents + 13 boxes = combinatorial blow-up).
        """
        import time
        assert self.profile is not None
        n_agents = self.profile.num_agents
        max_closed = (
            _JOINT_MAX_CLOSED_2 if n_agents <= 2
            else _JOINT_MAX_CLOSED_3 if n_agents <= 3
            else _JOINT_MAX_CLOSED
        )
        deadline = time.time() + time_budget_s

        h0 = self._joint_h(initial_state)
        if h0 >= max_closed:
            return None

        start = _JNode(f=h0, g=0, state=initial_state)
        heap: list[tuple] = [(h0, 0, start)]
        closed: set[State] = set()
        counter = 0
        depth_limit = 1000 if n_agents <= 2 else 600

        while heap:
            if len(closed) >= max_closed:
                print(f"  Joint A*: hit {max_closed} state limit.", file=sys.stderr, flush=True)
                return None
            # Wall-clock check every 1000 expansions to keep overhead low.
            if (counter & 0x3FF) == 0 and time.time() >= deadline:
                print(
                    f"  Joint A*: time budget {time_budget_s:.1f}s exhausted "
                    f"({len(closed)} closed) — giving up.",
                    file=sys.stderr, flush=True,
                )
                return None

            _, _, node = heapq.heappop(heap)
            if node.state in closed:
                continue
            closed.add(node.state)

            if node.state.is_goal_state():
                actions: list[list[Action]] = []
                n: _JNode | None = node
                while n is not None and n.state.joint_action is not None:
                    actions.append(n.state.joint_action)
                    n = n.parent
                actions.reverse()
                print(
                    f"  Joint A*: solved in {len(actions)} steps ({len(closed)} states).",
                    file=sys.stderr, flush=True,
                )
                return actions

            if node.g >= depth_limit:
                continue

            for succ in node.state.get_expanded_states():
                if succ in closed:
                    continue
                new_g = node.g + 1
                new_h = self._joint_h(succ)
                counter += 1
                heapq.heappush(
                    heap,
                    (new_g + new_h, counter, _JNode(f=new_g + new_h, g=new_g, state=succ, parent=node)),
                )

        return None

    def _joint_h(self, state: State) -> int:
        """Admissible heuristic for Joint A*: sum of per-agent task lower bounds."""
        INF = 10 ** 9
        total = 0
        for agent in self.agents:
            # First unfinished task for this agent (uses agent.tasks list)
            tasks_list = getattr(agent, "tasks", None)
            if not tasks_list:
                # nav-only goal
                if agent.agent_goal is not None and self.dist_map is not None:
                    gr, gc = agent.agent_goal
                    ar, ac = state.agent_rows[agent.agent_id], state.agent_cols[agent.agent_id]
                    total += abs(ar - gr) + abs(ac - gc)
                continue

            for task in tasks_list:
                # task is a 5-tuple (br, bc, gr, gc, char)
                if len(task) < 5:
                    continue
                goal_r, goal_c, box_char = task[2], task[3], task[4]
                if state.boxes[goal_r][goal_c] == box_char:
                    continue
                # find box and add its dist to goal
                for r, row in enumerate(state.boxes):
                    found = False
                    for c, ch in enumerate(row):
                        if ch == box_char and not (r == goal_r and c == goal_c):
                            d = self.dist_map.dist(goal_r, goal_c, r, c) if self.dist_map else INF
                            if d >= INF:
                                d = abs(r - goal_r) + abs(c - goal_c)
                            total += d
                            found = True
                            break
                    if found:
                        break
                break  # only consider first unfinished task
        return total

    # ------------------------------------------------------------------
    # Recovery nudge — pick a single-turn cardinal Move to unstick an
    # agent. Used by both the obstacle-unblock branch and the stuck-
    # agent recovery branch in get_joint_action.
    # ------------------------------------------------------------------

    def _pick_recovery_move(
        self, joint_state: State, agent_id: int,
    ) -> "Action | None":
        """Deterministic single-step nudge for an agent with no plan.

        Rotates through cardinal directions (N → S → E → W) using a
        per-agent counter, returning the first applicable Move. The
        rotation guarantees that consecutive calls for the same agent
        try different directions, so a single unfavourable choice does
        not freeze the agent. Fully deterministic across runs: no RNG,
        no wall-clock, only the agent's id and prior call history.
        """
        candidates = [Action.MoveN, Action.MoveS, Action.MoveE, Action.MoveW]

        # Informed pick: prefer cardinals that move the agent toward an
        # escape cell (BFS-derived target in open space). This is the
        # first-choice attempt; if no escape cell is found or all the
        # informed picks are blocked, we fall back to the rotation
        # below so the agent still makes SOME move.
        escape = self._find_escape_cell(joint_state, agent_id)
        if escape is not None:
            ar = joint_state.agent_rows[agent_id]
            ac = joint_state.agent_cols[agent_id]
            er, ec = escape
            dr = er - ar
            dc = ec - ac
            preferred: list[Action] = []
            # Cardinals ordered by which one reduces row-distance first if
            # the row gap is larger, else column-distance first.
            if abs(dr) >= abs(dc):
                if dr > 0: preferred.append(Action.MoveS)
                elif dr < 0: preferred.append(Action.MoveN)
                if dc > 0: preferred.append(Action.MoveE)
                elif dc < 0: preferred.append(Action.MoveW)
            else:
                if dc > 0: preferred.append(Action.MoveE)
                elif dc < 0: preferred.append(Action.MoveW)
                if dr > 0: preferred.append(Action.MoveS)
                elif dr < 0: preferred.append(Action.MoveN)
            for action in preferred:
                if joint_state.is_applicable(agent_id, action):
                    return action

        # Rotation fallback — informed pick was blocked or escape cell
        # not available. Rotate through cardinals with a per-agent counter
        # so we don't pick the same direction every time.
        rot = self.agents_nudge_rotation.get(agent_id, 0)
        self.agents_nudge_rotation[agent_id] = rot + 1
        start = (rot + agent_id) % 4
        ordered = candidates[start:] + candidates[:start]
        for action in ordered:
            if joint_state.is_applicable(agent_id, action):
                return action
        return None

    # ------------------------------------------------------------------
    # Escape cell (ported from felix/structure — graceful deadlock recovery)
    # ------------------------------------------------------------------

    def _step_aside_cell(
        self, joint_state: State, agent_id: int, max_dist: int = 12,
        min_dist: int = 1, avoid_cells: "set[tuple[int, int]] | None" = None,
    ) -> tuple[int, int] | None:
        """BFS outward to find a safe parking cell. Safe = not a wall, not a
        box, not another agent's cell, not a pending box-goal target, and
        not in `avoid_cells` (e.g. other agents' level-goal squares). BFS
        traverses through unsafe cells but only RETURNS a safe cell at
        distance ≥ min_dist."""
        ar = joint_state.agent_rows[agent_id]
        ac = joint_state.agent_cols[agent_id]
        walls = State.walls
        num_rows = len(walls)
        num_cols = len(walls[0]) if walls else 0
        other_positions = {
            (joint_state.agent_rows[a], joint_state.agent_cols[a])
            for a in range(len(joint_state.agent_rows)) if a != agent_id
        }
        avoid = set(avoid_cells) if avoid_cells else set()
        visited: set[tuple[int, int]] = {(ar, ac)}
        queue: deque[tuple[int, int, int]] = deque([(ar, ac, 0)])
        while queue:
            r, c, d = queue.popleft()
            if d >= max_dist:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in visited:
                    continue
                if not (0 <= nr < num_rows and 0 <= nc < num_cols):
                    continue
                if walls[nr][nc]:
                    continue
                if joint_state.boxes[nr][nc]:
                    continue
                if (nr, nc) in other_positions:
                    continue
                visited.add((nr, nc))
                # Pending goal cell? Traverse through but don't park here.
                goal_char = State.goals[nr][nc]
                is_pending_goal = (
                    bool(goal_char) and joint_state.boxes[nr][nc] != goal_char
                )
                new_d = d + 1
                if (
                    not is_pending_goal
                    and (nr, nc) not in avoid
                    and new_d >= min_dist
                ):
                    return (nr, nc)
                queue.append((nr, nc, new_d))
        return None

    def _find_escape_cell(
        self, joint_state: State, agent_id: int, rotation: int = 0,
    ) -> tuple[int, int] | None:
        """BFS outward from agent's current position to find a safe escape cell.
        Returns a cell ≥6 steps away that is clear of walls/boxes and not near
        other agents. Direction bias by `(agent_id + rotation) % 4` — pass a
        non-zero `rotation` on consecutive calls so the same agent gets a
        different escape cell each time."""
        ar = joint_state.agent_rows[agent_id]
        ac = joint_state.agent_cols[agent_id]
        other_positions = {
            (joint_state.agent_rows[a], joint_state.agent_cols[a])
            for a in range(len(joint_state.agent_rows)) if a != agent_id
        }
        blocked_area: set[tuple[int, int]] = set()
        for r, c in other_positions:
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    blocked_area.add((r + dr, c + dc))

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        n = (agent_id + rotation) % 4
        ordered_dirs = dirs[n:] + dirs[:n]

        walls = State.walls
        num_rows = len(walls)
        num_cols = len(walls[0]) if walls else 0
        visited = {(ar, ac)}
        queue: deque[tuple[int, int, int]] = deque([(ar, ac, 0)])

        while queue:
            r, c, dist = queue.popleft()
            if dist >= 15:
                break
            for dr, dc in ordered_dirs:
                nr, nc = r + dr, c + dc
                if (nr, nc) in visited:
                    continue
                if not (0 <= nr < num_rows and 0 <= nc < num_cols):
                    continue
                if walls[nr][nc] or joint_state.boxes[nr][nc]:
                    continue
                visited.add((nr, nc))
                new_dist = dist + 1
                if new_dist >= 6 and (nr, nc) not in blocked_area:
                    return (nr, nc)
                queue.append((nr, nc, new_dist))
        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def is_joint_action_valid(
        self, joint_state: State, joint_action: list[Action]
    ) -> tuple[bool, list[int]]:
        """
        Validate a joint action comprehensively and return offending agents.

        Checks:
        1. Each action is applicable for its agent
        2. No conflicts between agents (collisions, swaps, etc.)
        3. All agents have valid positions

        Returns a tuple (is_valid, invalid_agent_ids). `invalid_agent_ids` is
        an empty list when `is_valid` is True. Otherwise it contains the IDs
        of agents that appear to be performing invalid or conflicting actions.
        """
        invalid_agents: set[int] = set()

        # Check: correct number of actions
        if len(joint_action) != len(self.agents):
            return False, []

        # Check: each action is applicable for the agent
        for agent_id, action in enumerate(joint_action):
            if not joint_state.is_applicable(agent_id, action):
                invalid_agents.add(agent_id)

        # Prepare maps for conflict detection
        num_agents = len(self.agents)
        # dest: destination coord -> agent_id that will occupy it
        dest: dict[tuple[int, int], int] = {}
        # box_dest: destination coord -> agent_id that will push/pull a box there
        box_dest: dict[tuple[int, int], int] = {}
        agent_destinations: list[tuple[int, int] | None] = [None] * num_agents

        # Current positions map
        cur_pos: dict[tuple[int, int], int] = {}
        for aid in range(num_agents):
            cur_pos[(joint_state.agent_rows[aid], joint_state.agent_cols[aid])] = aid

        # First pass: mark positions of agents doing NoOp (staying in place)
        for agent_id, action in enumerate(joint_action):
            ar, ac = joint_state.agent_rows[agent_id], joint_state.agent_cols[agent_id]
            if action.type is ActionType.NoOp:
                dest[(ar, ac)] = agent_id  # Mark current position as occupied
                agent_destinations[agent_id] = (ar, ac)

        # Second pass: check movement actions against occupied cells and record conflicts
        for agent_id, action in enumerate(joint_action):
            ar, ac = joint_state.agent_rows[agent_id], joint_state.agent_cols[agent_id]

            if action.type is ActionType.NoOp:
                continue

            if action.type is ActionType.Move:
                nr, nc = ar + action.agent_row_delta, ac + action.agent_col_delta
                # Check if destination is already taken by another agent
                if (nr, nc) in dest:
                    invalid_agents.add(agent_id)
                    invalid_agents.add(dest[(nr, nc)])
                # Check if destination will have a box pushed there
                if (nr, nc) in box_dest:
                    invalid_agents.add(agent_id)
                    invalid_agents.add(box_dest[(nr, nc)])
                dest[(nr, nc)] = agent_id
                agent_destinations[agent_id] = (nr, nc)

            elif action.type is ActionType.Push:
                br, bc = ar + action.agent_row_delta, ac + action.agent_col_delta
                bdr, bdc = br + action.box_row_delta, bc + action.box_col_delta
                # Check if box destination is already taken by another box
                if (bdr, bdc) in box_dest:
                    invalid_agents.add(agent_id)
                    invalid_agents.add(box_dest[(bdr, bdc)])
                # Check if box destination will have an agent
                if (bdr, bdc) in dest:
                    invalid_agents.add(agent_id)
                    invalid_agents.add(dest[(bdr, bdc)])
                # Check if agent's entry to box cell conflicts with another agent
                if (br, bc) in dest:
                    invalid_agents.add(agent_id)
                    invalid_agents.add(dest[(br, bc)])
                box_dest[(bdr, bdc)] = agent_id
                dest[(br, bc)] = agent_id
                agent_destinations[agent_id] = (br, bc)

            elif action.type is ActionType.Pull:
                nr, nc = ar + action.agent_row_delta, ac + action.agent_col_delta
                # Check if new agent position conflicts
                if (nr, nc) in dest:
                    invalid_agents.add(agent_id)
                    invalid_agents.add(dest[(nr, nc)])
                # Check if new agent position will have a box destination
                if (nr, nc) in box_dest:
                    invalid_agents.add(agent_id)
                    invalid_agents.add(box_dest[(nr, nc)])
                # Check if agent's old position (where box is pulled) conflicts
                if (ar, ac) in dest:
                    invalid_agents.add(agent_id)
                    invalid_agents.add(dest[(ar, ac)])
                dest[(nr, nc)] = agent_id
                box_dest[(ar, ac)] = agent_id
                agent_destinations[agent_id] = (nr, nc)

        # Check: detect position swaps (two agents exchanging positions is forbidden)
        for i in range(num_agents):
            if agent_destinations[i] is None:
                continue
            di = agent_destinations[i]
            for j in range(i + 1, num_agents):
                if agent_destinations[j] is None:
                    continue
                dj = agent_destinations[j]
                # Swap detected: agent i goes where j was AND j goes where i was
                if di == (
                    joint_state.agent_rows[j],
                    joint_state.agent_cols[j],
                ) and dj == (joint_state.agent_rows[i], joint_state.agent_cols[i]):
                    invalid_agents.add(i)
                    invalid_agents.add(j)

        is_valid = len(invalid_agents) == 0
        return is_valid, sorted(invalid_agents)
