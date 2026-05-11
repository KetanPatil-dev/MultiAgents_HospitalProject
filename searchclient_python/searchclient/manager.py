"""
Manager — coordinates all agents every timestep.

Architecture:
  - setup()           : parse assignment, initialise agents.
  - get_joint_action(): BDI loop (sense → plan_is_sound → replan → act)
                         + priority-based conflict resolution.
  - is_done()         : termination check.

Planning strategy (in order of preference):
  1. Hierarchical Cooperative A* (HCA*) — two passes, priority order.
  2. Joint A* fallback — for small levels where HCA* fails (corridor conflicts).
"""

from __future__ import annotations

import heapq
import sys
from dataclasses import dataclass, field

from searchclient.action import Action, ActionType
from typing import TYPE_CHECKING
from searchclient.agent import Agent
from searchclient.heuristics import DistanceMap
from searchclient.level_parser import LevelProfile
from searchclient.state import State


# Max future timesteps for permanent constraints in HCA*
_MAX_FUTURE = 600
# Joint A* state-space caps (scaled by agent count)
_JOINT_MAX_CLOSED_2  = 1_000_000   # 2-agent levels
_JOINT_MAX_CLOSED_3  = 500_000     # 3-agent levels
_JOINT_MAX_CLOSED    = 300_000     # 4+ agent levels (rarely used)


# ---------------------------------------------------------------------------
# Joint A* node
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _JNode:
    f:      int
    g:      int               = field(compare=False)
    state:  State             = field(compare=False)
    parent: "_JNode | None"   = field(compare=False, default=None)


class Manager:

    def __init__(self) -> None:
        self.agents:       list[Agent] = []
        self.profile:      LevelProfile | None = None
        self.dist_map:     DistanceMap | None = None
        self.timestep:     int = 0
        # Shared space-time reservation table: (r, c, t) → agent_id
        self.reservations: dict[tuple[int, int, int], int] = {}
        # Joint plan (set when joint A* fallback fires)
        self._joint_plan:  list[list[Action]] = []
        # Deadlock detection
        self._recent_pos:    list[tuple] = []
        self._forced_noop:   dict[int, int] = {}  # agent_id → remaining forced-NoOp turns
        self._deadlock_count: dict[int, int] = {}  # agent_id → consecutive deadlock count
        self._noop_streak:   dict[int, int] = {}  # agent_id → consecutive NoOp turns

    # ------------------------------------------------------------------
    # Setup (once after parsing)
    # ------------------------------------------------------------------

    def setup(self, initial_state: State, profile: LevelProfile) -> None:
        self.profile  = profile
        self.dist_map = DistanceMap.from_state(initial_state)

        self.agents = [
            Agent(i, self.dist_map, profile)
            for i in range(profile.num_agents)
        ]

        # ---- Box task assignment ----
        from searchclient.planner.hungarian import assign, agent_goals_assign, subgoal_order

        box_assignment = assign(
            agents=self.agents,
            boxes=profile.real_boxes,
            goals=profile.box_goals,
            dist_map=self.dist_map,
        )

        agent_rows0 = initial_state.agent_rows
        agent_cols0 = initial_state.agent_cols

        for agent in self.agents:
            tasks = box_assignment.get(agent.agent_id, [])
            if len(tasks) > 1:
                tasks = subgoal_order(
                    tasks, self.dist_map,
                    start_r=agent_rows0[agent.agent_id],
                    start_c=agent_cols0[agent.agent_id],
                )
            agent.assign_tasks(tasks)

        # ---- Agent positional goal assignment ----
        ag_goals = agent_goals_assign(self.agents, profile.agent_goals)
        for agent in self.agents:
            agent.agent_goal = ag_goals.get(agent.agent_id)

        # ---- Cooperative pre-planning (HCA*) ----
        self._cooperative_preplan(initial_state)

        # ---- Fix 2: try alternative priority orderings if HCA* left failures ----
        self._try_priority_permutations(initial_state)

        # ---- Fix 4: CBS post-pass to resolve trajectory conflicts ----
        self._cbs_resolve(initial_state)

        # ---- Joint A* (proactive for 2-agent levels; fallback otherwise) ----
        # Only run proactively for 2-agent levels (≤450 cells) since 3+ agent
        # joint A* with boxes is too slow for the planning timeout.
        # For all other cases, only use it when HCA* left an agent without a plan.
        n_cells = profile.num_rows * profile.num_cols
        _small_level = (profile.num_agents <= 2 and n_cells <= 450)
        _hca_incomplete = any(
            (a.tasks or a.agent_goal is not None) and not a._plan
            for a in self.agents
        )
        # Joint A* is only practical for ≤2 agents — 3+ agent joint search
        # with boxes times out on any non-trivial level.
        if _small_level or (_hca_incomplete and profile.num_agents <= 2):
            reason = "small level" if _small_level else "HCA* incomplete"
            print(
                f"Trying joint A* ({reason}).",
                file=sys.stderr, flush=True,
            )
            joint_actions = self._joint_astar_plan(initial_state)
            if joint_actions is not None:
                self._joint_plan = joint_actions
                print(
                    f"Joint A* found plan of length {len(joint_actions)}.",
                    file=sys.stderr, flush=True,
                )
            else:
                print(
                    "Joint A* failed — using HCA* plans / reactive replanning.",
                    file=sys.stderr, flush=True,
                )

        print(
            f"Manager ready: {profile.num_agents} agents, "
            f"{len(profile.real_boxes)} plannable boxes, "
            f"{len(profile.deco_boxes)} decorative.",
            file=sys.stderr, flush=True,
        )
        for agent in self.agents:
            n = len(agent.tasks)
            plan_len = len(agent._plan)
            print(
                f"  Agent {agent.agent_id}: {n} task(s), "
                f"initial plan length={plan_len}, "
                f"joint_plan={'yes' if self._joint_plan else 'no'}.",
                file=sys.stderr, flush=True,
            )

    # ------------------------------------------------------------------
    # Joint A* fallback
    # ------------------------------------------------------------------

    def _joint_astar_plan(self, initial_state: State) -> list[list[Action]] | None:
        """
        A* over the joint state space.  Used when HCA* fails to find plans for
        all agents (typically due to corridor-blocking conflicts).

        Returns list[joint_action] (one joint action per timestep) or None.
        Only practical for small levels (≤3 agents, simple topologies).
        """
        n_agents = len(self.agents) if self.profile is None else self.profile.num_agents
        if n_agents <= 2:
            max_closed = _JOINT_MAX_CLOSED_2
        elif n_agents <= 3:
            max_closed = _JOINT_MAX_CLOSED_3
        else:
            max_closed = _JOINT_MAX_CLOSED

        h0 = self._joint_h(initial_state)
        if h0 >= max_closed:
            return None  # unreachable even to the heuristic

        start = _JNode(f=h0, g=0, state=initial_state)
        heap: list[tuple] = [(h0, 0, start)]
        closed: set[State] = set()
        counter = 0

        while heap:
            if len(closed) >= max_closed:
                print(
                    f"Joint A*: hit {max_closed} state limit.",
                    file=sys.stderr, flush=True,
                )
                break

            _, _, node = heapq.heappop(heap)
            if node.state in closed:
                continue
            closed.add(node.state)

            if node.state.is_goal_state():
                # Extract plan by tracing parents
                actions: list[list[Action]] = []
                n: _JNode | None = node
                while n is not None and n.state.joint_action is not None:
                    actions.append(n.state.joint_action)
                    n = n.parent
                actions.reverse()
                print(
                    f"Joint A*: solved in {len(actions)} steps "
                    f"({len(closed)} states explored).",
                    file=sys.stderr, flush=True,
                )
                return actions

            depth_limit = 1000 if n_agents <= 2 else 600
            if node.g >= depth_limit:
                continue  # depth limit

            for succ in node.state.get_expanded_states():
                if succ in closed:
                    continue
                new_g = node.g + 1
                new_h = self._joint_h(succ)
                counter += 1
                new_node = _JNode(f=new_g + new_h, g=new_g, state=succ, parent=node)
                heapq.heappush(heap, (new_g + new_h, counter, new_node))

        return None

    def _joint_h(self, state: State) -> int:
        """
        Admissible heuristic for joint A*: sum of per-agent lower bounds.
        For each agent's first unfinished task: dist(box → goal).
        """
        INF = 10 ** 9
        total = 0
        for agent in self.agents:
            # Find first unfinished task
            for task in agent.tasks:
                goal_r, goal_c, box_char = task[2], task[3], task[4]
                if state.boxes[goal_r][goal_c] == box_char:
                    continue  # already done
                # Find box in current state
                for r, row in enumerate(state.boxes):
                    for c, ch in enumerate(row):
                        if ch == box_char and not (r == goal_r and c == goal_c):
                            d = self.dist_map.dist(goal_r, goal_c, r, c)
                            if d >= INF:
                                d = abs(r - goal_r) + abs(c - goal_c)
                            total += d
                            break
                    else:
                        continue
                    break
                break  # heuristic based on first unfinished task

            if not agent.tasks and agent.agent_goal is not None:
                gr, gc = agent.agent_goal
                ar, ac = state.agent_rows[agent.agent_id], state.agent_cols[agent.agent_id]
                total += abs(ar - gr) + abs(ac - gc)

        return total

    # ------------------------------------------------------------------
    # Cooperative pre-planning (HCA*)
    # ------------------------------------------------------------------

    def _cooperative_preplan(self, initial_state: State) -> None:
        """
        Two-pass Hierarchical Cooperative A*.

        Pass 1: Plan ALL tasks for each agent in priority order (0 first).
                Each agent's FULL trajectory is registered as space-time
                constraints for lower-priority agents.

        Pass 2: Retry failed agents using a projected state where other
                agents' boxes are REMOVED (not placed at goals) and
                time-indexed box-trajectory constraints are added.
        """
        from searchclient.planner.astar import solve

        # ---- Pass 1: standard priority order, ALL tasks per agent ----
        for agent in self.agents:
            if not agent.tasks:
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
                    f"  Agent {agent.agent_id}: cooperative pre-plan FAILED (pass 1).",
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

        # ---- Pass 2: retry failed agents with projected state ----
        self._retry_failed_agents(initial_state)

    def _plan_all_tasks(
        self,
        agent:         Agent,
        initial_state: State,
        t_base_offset: int,
    ) -> tuple[list[Action], bool]:
        """
        Plan ALL tasks for `agent` sequentially, chaining simulated states.
        Also chains the agent_goal navigation at the end if set, so the
        combined box-delivery+nav plan ensures the agent ends up reachable
        from its final box-delivery position.
        Returns (concatenated_plan, success).
        """
        from searchclient.planner.astar import solve

        full_plan: list[Action] = []
        current_state = initial_state
        t_offset = t_base_offset

        for task in agent.tasks:
            goal_r, goal_c, box_char = task[2], task[3], task[4]

            box_r, box_c = agent._find_box(current_state, box_char, goal_r, goal_c)
            if box_r is None:
                continue  # already at goal in simulated state

            # Shift agent's constraints to local time for this task
            shifted = {
                (r, c, t - t_offset)
                for r, c, t in agent.constraints
                if t >= t_offset
            }

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

        # Chain agent_goal navigation ONLY when there is exactly 1 box task.
        # With multiple boxes the agent navigates between tasks anyway, and
        # chaining all of them makes A* too slow for larger levels.
        if agent.agent_goal is not None and len(agent.tasks) <= 1:
            gr, gc = agent.agent_goal
            ar = current_state.agent_rows[agent.agent_id]
            ac = current_state.agent_cols[agent.agent_id]
            if not (ar == gr and ac == gc):
                shifted = {
                    (r, c, t - t_offset)
                    for r, c, t in agent.constraints
                    if t >= t_offset
                }
                nav_plan = solve(
                    state=current_state,
                    agent_id=agent.agent_id,
                    goal=(None, None, gr, gc),
                    constraints=shifted,
                    dist_map=self.dist_map,
                )
                if nav_plan is None:
                    return full_plan, False  # delivery route leaves agent stranded
                full_plan.extend(nav_plan)

        return full_plan, True

    # ------------------------------------------------------------------

    def _retry_failed_agents(self, initial_state: State) -> None:
        """
        For every agent that still has no plan after pass 1, build a
        projected state where other agents' task boxes are REMOVED entirely
        (not placed at goals) and add:
          - time-indexed box-trajectory constraints (wait until cell clears)
          - permanent constraints for the goal position after plan ends

        Removing boxes (instead of placing at goal) allows A* to route
        through cells that will be vacated in time, while the trajectory
        constraints enforce proper waiting.
        """
        from searchclient.planner.astar import solve

        needs_retry = [
            a for a in self.agents
            if (a.tasks or a.agent_goal is not None) and not a._plan
        ]
        if not needs_retry:
            return

        for agent in needs_retry:
            proj_boxes = [row[:] for row in initial_state.boxes]
            extra_constraints: set[tuple[int, int, int]] = set()

            for other in self.agents:
                if other.agent_id == agent.agent_id:
                    continue
                if not other._plan or not other.tasks:
                    continue

                other_task = other.tasks[0]
                o_goal_r, o_goal_c, o_char = other_task[2], other_task[3], other_task[4]

                o_box_r, o_box_c = agent._find_box(initial_state, o_char, o_goal_r, o_goal_c)
                if o_box_r is None:
                    continue

                # Time-indexed constraints: wait for box to vacate each cell
                traj = self._compute_box_trajectory(
                    initial_state, other.agent_id, o_box_r, o_box_c, other._plan,
                )
                extra_constraints.update(traj)

                # Remove box from initial static grid
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

            all_constraints = agent.constraints | extra_constraints

            if agent.tasks:
                saved = agent.constraints
                agent.constraints = all_constraints
                full_plan, success = self._plan_all_tasks(agent, proj_state, 0)
                agent.constraints = saved

                if not success or not full_plan:
                    print(
                        f"  Agent {agent.agent_id}: retry pre-plan FAILED (pass 2).",
                        file=sys.stderr, flush=True,
                    )
                    continue
                plan = full_plan
            else:
                gr, gc = agent.agent_goal  # type: ignore[misc]
                plan = solve(
                    state=proj_state,
                    agent_id=agent.agent_id,
                    goal=(None, None, gr, gc),
                    constraints=all_constraints,
                    dist_map=self.dist_map,
                )
                if plan is None:
                    print(
                        f"  Agent {agent.agent_id}: retry pre-plan FAILED (pass 2).",
                        file=sys.stderr, flush=True,
                    )
                    continue

            agent._plan = plan
            agent._plan_index = 0
            print(
                f"  Agent {agent.agent_id}: retry pre-plan SUCCESS "
                f"(length={len(plan)}).",
                file=sys.stderr, flush=True,
            )

            if agent.tasks:
                task0 = agent.tasks[0]
                goal_r, goal_c, box_char = task0[2], task0[3], task0[4]
                box_r, box_c = agent._find_box(initial_state, box_char, goal_r, goal_c)
                if box_r is None:
                    box_r, box_c = -1, -1
            else:
                box_r, box_c = -1, -1
            self._register_agent_path(
                agent, initial_state,
                initial_state.agent_rows[agent.agent_id],
                initial_state.agent_cols[agent.agent_id],
                box_r, box_c, plan,
            )

    # ------------------------------------------------------------------
    # Alternative priority orderings (Fix 2)
    # ------------------------------------------------------------------

    def _cooperative_preplan_ordered(
        self,
        initial_state: State,
        order: list[int],
    ) -> None:
        """
        Same as _cooperative_preplan pass 1, but processes agents in `order`
        so that order[0] has highest priority and its trajectory constrains all
        subsequent agents.
        """
        from searchclient.planner.astar import solve

        ordered_agents = [self.agents[i] for i in order]

        for idx, agent in enumerate(ordered_agents):
            lower_priority = ordered_agents[idx + 1:]

            if not agent.tasks:
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
                            constrain_agents=lower_priority,
                        )
                continue

            full_plan, success = self._plan_all_tasks(agent, initial_state, 0)
            if not success or not full_plan:
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
                constrain_agents=lower_priority,
            )

        self._retry_failed_agents(initial_state)

    def _try_priority_permutations(self, initial_state: State) -> None:
        """
        After HCA* (passes 1+2), if agents still lack plans, try alternative
        priority orderings: all rotations, reverse, and failed-agent-first.
        Keeps the first ordering that reduces (or eliminates) failures.
        """
        failed_ids = [
            a.agent_id for a in self.agents
            if (a.tasks or a.agent_goal is not None) and not a._plan
        ]
        if not failed_ids:
            return

        n = len(self.agents)
        base = list(range(n))

        orderings: list[list[int]] = []
        for i in range(1, n):
            orderings.append(base[i:] + base[:i])         # rotations
        orderings.append(list(reversed(base)))             # reverse
        for fid in failed_ids:                             # failed agent first
            o = [fid] + [x for x in base if x != fid]
            if o not in orderings and o != base:
                orderings.append(o)

        for order in orderings:
            saved = {
                a.agent_id: (list(a._plan), a._plan_index, set(a.constraints))
                for a in self.agents
            }
            for a in self.agents:
                a._plan = []
                a._plan_index = 0
                a.constraints = set()

            self._cooperative_preplan_ordered(initial_state, order)

            new_failed = [
                a.agent_id for a in self.agents
                if (a.tasks or a.agent_goal is not None) and not a._plan
            ]
            if len(new_failed) < len(failed_ids):
                print(
                    f"Priority order {order}: {len(new_failed)} failure(s) "
                    f"(was {len(failed_ids)}).",
                    file=sys.stderr, flush=True,
                )
                failed_ids = new_failed
                if not failed_ids:
                    return
            else:
                for a in self.agents:
                    plan, idx, cons = saved[a.agent_id]
                    a._plan = plan
                    a._plan_index = idx
                    a.constraints = cons

    def _compute_box_trajectory(
        self,
        initial_state: State,
        agent_id:      int,
        box_r:         int,
        box_c:         int,
        plan:          list[Action],
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

    def _register_agent_path(
        self,
        agent:            Agent,
        state:            State,
        ar:               int,
        ac:               int,
        bxr:              int,
        bxc:              int,
        plan:             list[Action],
        constrain_agents: "list[Agent] | None" = None,
    ) -> None:
        """
        Add time-indexed agent and box positions as constraints for lower-priority
        agents.  If constrain_agents is None, defaults to agents with agent_id >
        agent.agent_id (original HCA* behaviour).  Pass an explicit list when
        using a non-default priority ordering.
        """
        if constrain_agents is None:
            constrain_agents = [a for a in self.agents if a.agent_id > agent.agent_id]

        prev_ar, prev_ac = ar, ac

        for t, action in enumerate(plan):
            for other in constrain_agents:
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

        for other in constrain_agents:
            other.constraints.add((ar, ac, len(plan)))
            if bxr >= 0:
                other.constraints.add((bxr, bxc, len(plan)))

    # ------------------------------------------------------------------
    # CBS post-pass (Fix 4)
    # ------------------------------------------------------------------

    def _simulate_agent_trajectory(
        self, initial_state: State, agent_id: int, plan: list[Action]
    ) -> list[tuple[int, int]]:
        """(row, col) for the agent at each timestep 0 … len(plan)."""
        ar = initial_state.agent_rows[agent_id]
        ac = initial_state.agent_cols[agent_id]
        traj = [(ar, ac)]
        for action in plan:
            if action.type in (ActionType.Move, ActionType.Push, ActionType.Pull):
                ar += action.agent_row_delta
                ac += action.agent_col_delta
            traj.append((ar, ac))
        return traj

    def _cbs_find_conflict(
        self,
        initial_state: State,
        plans: dict[int, list[Action]],
    ) -> "tuple[int, int, int, int, int] | None":
        """
        Find the first vertex or swap-edge conflict across all agent pairs.
        Returns (agent_i, agent_j, row, col, timestep) or None.
        """
        trajs: dict[int, list[tuple[int, int]]] = {
            aid: self._simulate_agent_trajectory(initial_state, aid, plan)
            for aid, plan in plans.items()
        }
        if not trajs:
            return None

        max_t = max(len(t) for t in trajs.values())
        padded: dict[int, list[tuple[int, int]]] = {
            aid: traj + [traj[-1]] * (max_t - len(traj))
            for aid, traj in trajs.items()
        }

        ids = sorted(trajs.keys())
        for t in range(1, max_t):
            for ii in range(len(ids)):
                ai = ids[ii]
                for jj in range(ii + 1, len(ids)):
                    aj = ids[jj]
                    ri, ci = padded[ai][t]
                    rj, cj = padded[aj][t]
                    if ri == rj and ci == cj:                   # vertex conflict
                        return (ai, aj, ri, ci, t)
                    ri_p, ci_p = padded[ai][t - 1]
                    rj_p, cj_p = padded[aj][t - 1]
                    if ri == rj_p and ci == cj_p and rj == ri_p and cj == ci_p:
                        return (ai, aj, ri, ci, t)              # swap conflict

        return None

    def _cbs_replan_agent(
        self,
        initial_state:     State,
        agent:             Agent,
        extra_constraints: set[tuple[int, int, int]],
    ) -> "list[Action] | None":
        """Replan `agent` with extra_constraints merged into its existing set."""
        saved = agent.constraints
        agent.constraints = saved | extra_constraints
        if agent.tasks:
            full_plan, success = self._plan_all_tasks(agent, initial_state, 0)
            agent.constraints = saved
            return full_plan if success else None
        elif agent.agent_goal is not None:
            from searchclient.planner.astar import solve
            gr, gc = agent.agent_goal
            plan = solve(
                state=initial_state,
                agent_id=agent.agent_id,
                goal=(None, None, gr, gc),
                constraints=agent.constraints,
                dist_map=self.dist_map,
            )
            agent.constraints = saved
            return plan
        else:
            agent.constraints = saved
            return []

    def _cbs_resolve(self, initial_state: State) -> None:
        """
        CBS post-pass: detect and resolve trajectory conflicts between the
        HCA*-generated plans.  Only runs when every active agent already has
        a plan.  Falls back silently if the node budget is exhausted.
        """
        from copy import deepcopy

        if not all(
            a._plan or (not a.tasks and a.agent_goal is None)
            for a in self.agents
        ):
            return  # not all agents planned — nothing to resolve

        plans: dict[int, list[Action]] = {
            a.agent_id: list(a._plan) for a in self.agents
        }
        if self._cbs_find_conflict(initial_state, plans) is None:
            return  # already conflict-free

        print("CBS: resolving trajectory conflicts…", file=sys.stderr, flush=True)

        CBS_LIMIT = min(max(300, 150 * len(self.agents)), 2000)
        uid_counter = [0]

        start_extra: dict[int, set] = {a.agent_id: set() for a in self.agents}
        start_cost = sum(len(p) for p in plans.values())
        heap: list[tuple] = [(start_cost, 0, start_extra, plans)]
        uid_counter[0] = 1

        for _ in range(CBS_LIMIT):
            if not heap:
                break
            cost, _, extra, cur_plans = heapq.heappop(heap)

            conflict = self._cbs_find_conflict(initial_state, cur_plans)
            if conflict is None:
                for a in self.agents:
                    a._plan = list(cur_plans.get(a.agent_id, []))
                    a._plan_index = 0
                print(
                    f"CBS: all conflicts resolved (total length={cost}).",
                    file=sys.stderr, flush=True,
                )
                return

            ai, aj, r, c, t = conflict
            for agent_id in (ai, aj):
                new_extra = deepcopy(extra)
                new_extra[agent_id].add((r, c, t))
                new_plan = self._cbs_replan_agent(
                    initial_state, self.agents[agent_id], new_extra[agent_id]
                )
                if new_plan is None:
                    continue
                new_plans = dict(cur_plans)
                new_plans[agent_id] = new_plan
                new_cost = sum(len(p) for p in new_plans.values())
                uid = uid_counter[0]
                uid_counter[0] += 1
                heapq.heappush(heap, (new_cost, uid, new_extra, new_plans))

        print("CBS: hit node limit — keeping HCA* plans.", file=sys.stderr, flush=True)

    # ------------------------------------------------------------------
    # Main loop (called every turn)
    # ------------------------------------------------------------------

    def get_joint_action(self, joint_state: State) -> list[Action]:
        """
        If a joint plan exists (from joint A* fallback), serve from it.
        Otherwise run the full BDI loop with conflict resolution.
        """
        # --- Joint plan fast path ---
        if self._joint_plan:
            action = self._joint_plan.pop(0)
            self.timestep += 1
            return action

        t = self.timestep

        # Clean up stale constraints
        for agent in self.agents:
            agent.clear_old_constraints(t)

        # --- Decrement forced-NoOp counters ---
        for aid in list(self._forced_noop.keys()):
            self._forced_noop[aid] -= 1
            if self._forced_noop[aid] <= 0:
                del self._forced_noop[aid]

        # --- BDI step ---
        for agent in self.agents:
            agent.sense(joint_state, t)
            if not agent.plan_is_sound(joint_state, t):
                success = agent.replan(joint_state, t)
                if not success:
                    print(
                        f"t={t} Agent {agent.agent_id}: replan failed.",
                        file=sys.stderr, flush=True,
                    )

        # --- Propose actions (respect forced NoOps and escape stagger) ---
        proposed: list[Action] = []
        for agent in self.agents:
            if agent.agent_id in self._forced_noop:
                proposed.append(Action.NoOp)
                agent.step_back()
            elif agent._escape_stagger > 0:
                agent._escape_stagger -= 1
                proposed.append(Action.NoOp)
            else:
                proposed.append(agent.next_action())

        # --- Conflict resolution ---
        resolved, conflicted = self._resolve_conflicts(joint_state, proposed)

        # --- Step back conflicted agents ---
        # Keep the plan for a few turns so agents naturally unblock as peers advance.
        # Only clear (force replan) after YIELD_LIMIT consecutive conflicts.
        YIELD_LIMIT = 5
        for idx in conflicted:
            self.agents[idx].step_back()
            streak = self._noop_streak.get(idx, 0)  # reuse noop_streak as yield counter
            if streak >= YIELD_LIMIT:
                self.agents[idx]._plan = []
                self.agents[idx]._plan_index = 0

        # --- Deadlock detection ---
        stuck = self._detect_deadlock(joint_state, resolved)
        if stuck:
            # Only freeze ONE agent per detection (the most-stuck one).
            # Freezing all simultaneously means ALL block each other — no one moves.
            worst = max(stuck, key=lambda aid: self._deadlock_count.get(aid, 0))
            if worst not in self._forced_noop:
                count = self._deadlock_count.get(worst, 0) + 1
                self._deadlock_count[worst] = count
                ar = joint_state.agent_rows[worst]
                ac = joint_state.agent_cols[worst]

                if count > 15:
                    # Severely stuck: physically move the agent out of the cluster
                    a = self.agents[worst]
                    a.constraints.clear()
                    a._plan = []
                    a._plan_index = 0
                    escape = self._find_escape_cell(joint_state, worst)
                    if a.agent_goal is None and a.tasks:
                        # Box-task agent: assign escape cell as temporary goal
                        if escape is not None:
                            a.agent_goal = escape
                            self._noop_streak[worst] = 0
                            print(
                                f"t={t} Deadlock #{count}: agent {worst} ESCAPE to {escape}.",
                                file=sys.stderr, flush=True,
                            )
                        else:
                            print(
                                f"t={t} Deadlock #{count}: agent {worst} RESET (no escape).",
                                file=sys.stderr, flush=True,
                            )
                    elif not a.tasks and a.agent_goal is not None:
                        # Nav-only agent: escape temporarily, then resume real goal.
                        # Guard: don't overwrite a pending goal already in flight.
                        if escape is not None and a._pending_agent_goal is None:
                            a._pending_agent_goal = a.agent_goal
                            a.agent_goal = escape
                            self._noop_streak[worst] = 0
                            print(
                                f"t={t} Deadlock #{count}: agent {worst} NAV-ESCAPE to {escape}.",
                                file=sys.stderr, flush=True,
                            )
                    wait_turns = 1
                else:
                    wait_turns = min(2 + (count - 1) * 2, 10)
                    # Inject position constraints so A* replans a different route
                    for dt in range(1, wait_turns + 2):
                        self.agents[worst].constraints.add((ar, ac, t + dt))
                    print(
                        f"t={t} Deadlock #{count}: agent {worst} waits {wait_turns} turns.",
                        file=sys.stderr, flush=True,
                    )
                self._forced_noop[worst] = wait_turns
        # Reset deadlock count for active agents that are making progress
        for agent in self.agents:
            has_work = agent.tasks or agent.agent_goal is not None
            if (has_work and agent.agent_id not in stuck
                    and agent.agent_id in self._deadlock_count
                    and agent.agent_id not in self._forced_noop):
                del self._deadlock_count[agent.agent_id]

        # Track per-agent NoOp streaks. If an active agent has been stuck
        # for too long, assign it an escape goal in a different corridor so it
        # physically vacates the congested area and lets others through.
        STUCK_LIMIT = 8
        for i, action in enumerate(resolved):
            agent = self.agents[i]
            if action is Action.NoOp and (agent.tasks or agent.agent_goal is not None):
                self._noop_streak[i] = self._noop_streak.get(i, 0) + 1
            else:
                self._noop_streak[i] = 0

        for agent in self.agents:
            aid = agent.agent_id
            if (self._noop_streak.get(aid, 0) >= STUCK_LIMIT
                    and agent.tasks  # only for box-delivery agents (not nav-only)
                    and agent.agent_goal is None):  # don't override an existing escape goal
                escape = self._find_escape_cell(joint_state, aid)
                agent.constraints.clear()
                agent._plan = []
                agent._plan_index = 0
                self._noop_streak[aid] = 0
                if aid in self._forced_noop:
                    del self._forced_noop[aid]  # release so agent can move immediately
                if escape is not None:
                    agent.agent_goal = escape
                    print(
                        f"t={t} Agent {aid}: escaping to {escape} after {STUCK_LIMIT}-turn streak.",
                        file=sys.stderr, flush=True,
                    )

        self.timestep += 1
        return resolved

    # ------------------------------------------------------------------
    # Conflict resolution (Fix 3: score-based priority)
    # ------------------------------------------------------------------

    def _priority_score(self, agent_id: int, state: State) -> int:
        """
        Commit order for conflict resolution: higher score = commits first.
        Agents with more remaining work and closer to task completion go first.
        Done agents (no tasks, no goal) always score -1 so they yield last.
        """
        a = self.agents[agent_id]
        if not a.tasks and a.agent_goal is None:
            return -1
        score = len(a.tasks) * 100
        if a._plan:
            score += max(0, 50 - len(a.remaining_plan()))
        return score

    def _resolve_conflicts(
        self,
        joint_state:  State,
        joint_action: list[Action],
    ) -> tuple[list[Action], list[int]]:
        """
        Score-based greedy conflict resolution.

        Commits agents in descending _priority_score order (most-work-remaining
        first, done agents last).  If adding an agent's action causes a conflict,
        that agent does NoOp.  EXCEPTION: if the blocker is a done/nav-only agent
        and the active agent needs to pass, the blocker yields instead.
        """
        resolved   = [Action.NoOp] * len(joint_action)
        conflicted: list[int] = []

        # Highest priority commits first
        priority_order = sorted(
            range(len(joint_action)),
            key=lambda i: self._priority_score(i, joint_state),
            reverse=True,
        )
        processed: list[int] = []  # agents assigned to resolved so far

        def _is_done(agent_id: int) -> bool:
            a = self.agents[agent_id]
            return not a.tasks and a.agent_goal is None

        def _should_yield(blocker_id: int, active_id: int) -> bool:
            b = self.agents[blocker_id]
            a = self.agents[active_id]
            if not b.tasks and b.agent_goal is None:
                return True
            if not b.tasks and a.tasks:
                return True
            return False

        for i in priority_order:
            action = joint_action[i]
            resolved[i] = action
            if action is Action.NoOp:
                processed.append(i)
                continue

            if not joint_state.is_applicable(i, action):
                dest_r = joint_state.agent_rows[i] + getattr(action, 'agent_row_delta', 0)
                dest_c = joint_state.agent_cols[i] + getattr(action, 'agent_col_delta', 0)
                blocker = joint_state.agent_at(dest_r, dest_c)
                if blocker is not None:
                    j = ord(blocker) - ord('0')
                    if _should_yield(j, i) and self.agents[j].agent_goal is None:
                        jr, jc = joint_state.agent_rows[j], joint_state.agent_cols[j]
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = jr + dr, jc + dc
                            if joint_state.is_free(nr, nc):
                                self.agents[j].agent_goal = (nr, nc)
                                self.agents[j]._plan = []
                                self.agents[j]._plan_index = 0
                                break

                resolved[i] = Action.NoOp
                conflicted.append(i)
                processed.append(i)
                continue

            if joint_state.is_conflicting(resolved):
                dest_i_r, dest_i_c = self._agent_destination(joint_state, i, action)
                yielder = None
                if not _is_done(i):
                    for j in processed:
                        if resolved[j] is Action.NoOp and _should_yield(j, i):
                            jr, jc = joint_state.agent_rows[j], joint_state.agent_cols[j]
                            if jr == dest_i_r and jc == dest_i_c:
                                yielder = j
                                break

                if yielder is not None:
                    resolved[yielder] = Action.NoOp
                    conflicted.append(yielder)
                    self.agents[yielder]._plan = []
                    self.agents[yielder]._plan_index = 0
                    for dt in range(1, 5):
                        self.agents[yielder].constraints.add(
                            (joint_state.agent_rows[yielder],
                             joint_state.agent_cols[yielder],
                             self.timestep + dt)
                        )
                    if not joint_state.is_conflicting(resolved):
                        processed.append(i)
                        continue

                resolved[i] = Action.NoOp
                conflicted.append(i)

            processed.append(i)

        return resolved, conflicted

    def _agent_destination(
        self, joint_state: State, agent_id: int, action: Action
    ) -> tuple[int, int]:
        ar = joint_state.agent_rows[agent_id]
        ac = joint_state.agent_cols[agent_id]
        if action.type in (ActionType.Move, ActionType.Pull):
            return ar + action.agent_row_delta, ac + action.agent_col_delta
        if action.type is ActionType.Push:
            return ar + action.agent_row_delta, ac + action.agent_col_delta
        return ar, ac

    def _detect_deadlock(self, joint_state: State, resolved: list[Action]) -> list[int]:
        """
        Detect agents stuck in a cycle by tracking joint (agent+box) state.
        Returns agent ids to force into NoOp to break the cycle.
        """
        WINDOW = 12
        # Include relevant box positions in the key so moving a box breaks the cycle
        box_key = tuple(
            (r, c)
            for r, row in enumerate(joint_state.boxes)
            for c, ch in enumerate(row) if ch
        )
        key = (tuple(zip(joint_state.agent_rows, joint_state.agent_cols)), box_key)

        if key in self._recent_pos:
            # Cycle detected — penalise agents that are blocked (no plan, have work)
            stuck = [
                a.agent_id for a in self.agents
                if not a._plan and (a.tasks or a.agent_goal is not None)
                and a.agent_id not in self._forced_noop
            ]
            if stuck:
                return stuck
            # All agents with remaining work have plans but still cycling.
            # Penalise lowest-priority agent that still has work to do.
            # Never penalise "done" agents (no tasks, no goal) — they're idle.
            for agent in reversed(self.agents):
                if agent.agent_id not in self._forced_noop:
                    if agent.tasks or agent.agent_goal is not None:
                        return [agent.agent_id]

        self._recent_pos.append(key)
        if len(self._recent_pos) > WINDOW:
            self._recent_pos.pop(0)
        return []

    def _find_escape_cell(self, joint_state: State, agent_id: int) -> tuple[int, int] | None:
        """
        BFS outward from agent's current position to find a free cell that is:
        - At least 6 steps away (forces agent to actually leave the congested area)
        - Not the current or near-future position of any other agent
        - Not near any other agent's escape destination
        Biases search direction based on agent_id to spread agents apart.
        """
        from collections import deque
        ar = joint_state.agent_rows[agent_id]
        ac = joint_state.agent_cols[agent_id]
        other_positions = {
            (joint_state.agent_rows[a], joint_state.agent_cols[a])
            for a in range(len(joint_state.agent_rows)) if a != agent_id
        }
        # Also exclude cells near other agents (within distance 3)
        blocked_area: set[tuple[int, int]] = set()
        for r, c in other_positions:
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    blocked_area.add((r + dr, c + dc))

        # Rotate BFS direction order so different agent IDs explore different directions first
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        n = agent_id % 4
        ordered_dirs = dirs[n:] + dirs[:n]

        visited = {(ar, ac)}
        queue: deque[tuple[int, int, int]] = deque([(ar, ac, 0)])
        walls = State.walls
        num_rows, num_cols = len(walls), len(walls[0]) if walls else 0
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
                    return nr, nc
                queue.append((nr, nc, new_dist))
        return None

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def is_done(self, joint_state: State) -> bool:
        return joint_state.is_goal_state()


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _apply_agent_plan(state: State, agent_id: int, plan: list[Action]) -> State:
    """Apply a single-agent plan to the joint state (other agents stay put)."""
    current = state
    num_agents = len(state.agent_rows)
    for action in plan:
        joint = [Action.NoOp] * num_agents
        joint[agent_id] = action
        current = current.result(joint)
    return current
