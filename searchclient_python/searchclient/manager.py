"""
Manager — Clean slate for new coordinated multi-agent planning.

Architecture (to be implemented):
  - setup()           : initialize agents, assign tasks
  - get_joint_action(): per-timestep coordination
  - is_done()         : termination check
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from searchclient.action import Action, ActionType
from searchclient.agent import Agent
from searchclient.heuristics import DistanceMap
from searchclient.level_parser import LevelProfile
from searchclient.state import State

if TYPE_CHECKING:
    pass


class Manager:

    def __init__(self) -> None:
        self.agents: list[Agent] = []
        self.profile: LevelProfile | None = None
        self.dist_map: DistanceMap | None = None
        self.timestep: int = 0

    # ------------------------------------------------------------------
    # Setup (once after parsing)
    # ------------------------------------------------------------------

    def setup(self, initial_state: State, profile: LevelProfile) -> None:
        """
        Initialize manager for a new level.

        1. Assign tasks to agents (use Hungarian algorithm)
        """
        self.profile = profile
        self.dist_map = DistanceMap.from_state(initial_state)

        self.agents = [
            Agent(i, self.dist_map, profile) for i in range(profile.num_agents)
        ]

        # --- Task assignment ---
        from searchclient.planner.hungarian import (
            assign,
            agent_goals_assign,
            subgoal_order,
        )

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
                    tasks,
                    self.dist_map,
                    start_r=agent_rows0[agent.agent_id],
                    start_c=agent_cols0[agent.agent_id],
                )
            agent.assign_tasks(tasks)

        # --- Agent positional goals ---
        ag_goals = agent_goals_assign(self.agents, profile.agent_goals)
        for agent in self.agents:
            agent.agent_goal = ag_goals.get(agent.agent_id)

        print(
            f"Manager ready: {profile.num_agents} agents, "
            f"{len(profile.real_boxes)} plannable boxes, "
            f"{len(profile.deco_boxes)} decorative.",
            file=sys.stderr,
            flush=True,
        )
        for agent in self.agents:
            n = len(agent.tasks)
            print(
                f"  Agent {agent.agent_id}: {n} task(s).",
                file=sys.stderr,
                flush=True,
            )

        # --- HCA* Path Planning ---
        self._hca_plan(initial_state)

    def _hca_plan(self, initial_state: State, allow_recovery: bool = True) -> None:
        """
        Hierarchical Cooperative A* (HCA*).
        Plan for each agent sequentially, using previous agents' paths as constraints.
        """
        from searchclient.planner.astar import solve

        # Build constraints hierarchically
        all_constraints: list[set[tuple[int, int, int]]] = [set() for _ in self.agents]

        for agent_idx, agent in enumerate(self.agents):
            if not agent.tasks:
                # No tasks, agent stays put or goes to agent_goal if exists
                if agent.agent_goal:
                    goal = (None, None, agent.agent_goal[0], agent.agent_goal[1])
                    plan = solve(
                        initial_state,
                        agent_idx,
                        goal,
                        constraints=all_constraints[agent_idx],
                        dist_map=self.dist_map,
                    )
                    if plan:
                        agent._plan = plan
                continue

            # Plan for first task
            task = agent.tasks[0]
            box_r, box_c, goal_r, goal_c, box_char = task
            goal = (box_r, box_c, goal_r, goal_c, box_char)

            plan = solve(
                initial_state,
                agent_idx,
                goal,
                constraints=all_constraints[agent_idx],
                dist_map=self.dist_map,
            )

            if plan:
                agent._plan = plan

                # Extract constraints from this agent's plan
                # Simulate the plan execution to get occupied cells at each timestep
                ar, ac = (
                    initial_state.agent_rows[agent_idx],
                    initial_state.agent_cols[agent_idx],
                )
                br, bc = box_r, box_c

                timestep = 0
                for action in plan:
                    if action.type == ActionType.Move:
                        ar += action.agent_row_delta
                        ac += action.agent_col_delta
                    elif action.type == ActionType.Push:
                        ar += action.agent_row_delta
                        ac += action.agent_col_delta
                        br += action.box_row_delta
                        bc += action.box_col_delta
                    elif action.type == ActionType.Pull:
                        ar += action.agent_row_delta
                        ac += action.agent_col_delta
                        br += action.box_row_delta
                        bc += action.box_col_delta

                    timestep += 1
                    # Add agent and box positions as constraints for future agents
                    all_constraints[agent_idx].add((ar, ac, timestep))
                    if not (br == goal_r and bc == goal_c):
                        # Only constrain box position if still moving it
                        all_constraints[agent_idx].add((br, bc, timestep))

                # Propagate constraints to future agents
                for future_agent_idx in range(agent_idx + 1, len(self.agents)):
                    all_constraints[future_agent_idx].update(all_constraints[agent_idx])
            else:
                print(
                    f"Agent {agent_idx}: HCA* failed for first task {agent.tasks[0]}.",
                    file=sys.stderr,
                    flush=True,
                )

        print("HCA* planning complete for all agents.", file=sys.stderr, flush=True)

        # Check for failures and attempt recovery once
        if allow_recovery:
            self._handle_hca_failures(initial_state)

    def _handle_hca_failures(self, initial_state: State) -> None:
        """
        Handle HCA* planning failures.
        - If a task failed due to same-color box obstacle, reorder tasks
        - Complex agent-blocking scenarios deferred to PROJECT FUTURE
        """
        failed_agents = [a for a in self.agents if not a._plan and a.tasks]

        if not failed_agents:
            return

        print(
            f"HCA* recovery: {len(failed_agents)} agents need replanning.",
            file=sys.stderr,
            flush=True,
        )

        # Attempt task reordering for same-color box obstacles
        for agent in failed_agents:
            first_task = agent.tasks[0]
            box_r0, box_c0, goal_r0, goal_c0, box_char = first_task

            # Check if there's another box of same color in the way
            blocking_box = self._find_blocking_same_color_box(
                initial_state, box_r0, box_c0, box_char, agent.agent_id
            )

            if blocking_box:
                blocking_r, blocking_c, blocking_goal_r, blocking_goal_c = blocking_box

                print(
                    f"  Detected same-color box obstacle at ({blocking_r}, {blocking_c}) "
                    f"for Agent {agent.agent_id} targeting ({box_r0}, {box_c0}).",
                    file=sys.stderr,
                    flush=True,
                )

                blocking_task: tuple[int, int, int, int, str] | None = None

                # Find which agent currently owns this blocking box task.
                for other_agent in self.agents:
                    for other_task in list(other_agent.tasks):
                        other_br, other_bc = other_task[0], other_task[1]
                        if other_br == blocking_r and other_bc == blocking_c:
                            blocking_task = other_task
                            other_agent.tasks.remove(other_task)
                            print(
                                f"  Moving blocking task from Agent {other_agent.agent_id} "
                                f"to Agent {agent.agent_id}.",
                                file=sys.stderr,
                                flush=True,
                            )
                            break
                    if blocking_task is not None:
                        break

                if blocking_task is not None:
                    # Put the blocking task in front so this agent clears its own path first.
                    # Keep the original task immediately after it.
                    if agent.tasks and agent.tasks[0] != blocking_task:
                        original_task = agent.tasks[0]
                        agent.tasks = [blocking_task, original_task] + agent.tasks[1:]
                    elif not agent.tasks:
                        agent.tasks = [blocking_task]

                    print(
                        f"  Agent {agent.agent_id} will clear the obstacle first, then resume its original task.",
                        file=sys.stderr,
                        flush=True,
                    )

        # Re-run HCA* after task reordering
        print(
            "HCA* recovery: rerunning with reordered tasks.",
            file=sys.stderr,
            flush=True,
        )
        for agent in self.agents:
            agent._plan = []
            agent._plan_index = 0

        self._hca_plan(initial_state, allow_recovery=False)

    def _find_blocking_same_color_box(
        self,
        state: State,
        target_r: int,
        target_c: int,
        target_char: str,
        agent_id: int,
    ) -> tuple | None:
        """
        Check if there's another box of same color between agent and target box.
        Returns (box_r, box_c, goal_r, goal_c) of blocking box, or None.
        """
        assert self.profile is not None
        agent_r = state.agent_rows[agent_id]
        agent_c = state.agent_cols[agent_id]

        # Find all boxes of same color
        for br, bc, char in self.profile.real_boxes:
            if char != target_char or (br == target_r and bc == target_c):
                continue  # Skip target box itself

            # Check if this box is between agent and target
            # Simple heuristic: if it's roughly on the path
            dy_agent_blocking = abs(agent_r - br)
            dy_blocking_target = abs(br - target_r)
            dx_agent_blocking = abs(agent_c - bc)
            dx_blocking_target = abs(bc - target_c)

            # If blocking box is along the path (within taxicab distance)
            if (
                dy_agent_blocking + dy_blocking_target <= abs(agent_r - target_r) + 2
                and dx_agent_blocking + dx_blocking_target
                <= abs(agent_c - target_c) + 2
            ):

                # Find goal for this box in profile
                for goal_r, goal_c, goal_char in self.profile.box_goals:
                    if goal_char == char:
                        return (br, bc, goal_r, goal_c)

        return None

    # ------------------------------------------------------------------
    # Main loop (called every turn)
    # ------------------------------------------------------------------

    def get_joint_action(self, joint_state: State) -> list[Action]:
        """
        Execute the next timestep: all agents execute their next planned action.
        """
        joint_action = []
        for agent in self.agents:
            action = agent.next_action()
            joint_action.append(action)
        return joint_action

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def is_done(self, joint_state: State) -> bool:
        return joint_state.is_goal_state()
