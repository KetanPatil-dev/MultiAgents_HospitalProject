"""
Manager — Coordinated multi-agent planning with task-based approach.

Architecture:
  - setup()           : initialize agents, build task lists, prearrange HCA*
  - get_joint_action(): per-timestep coordination (BDI cycle)
  - is_done()         : termination check
"""

from __future__ import annotations

from collections import deque
import sys
from typing import TYPE_CHECKING, TypedDict

from searchclient.action import Action, ActionType
from searchclient.agent import Agent
from searchclient.color import Color
from searchclient.heuristics import DistanceMap
from searchclient.level_parser import LevelProfile
from searchclient.state import State
from searchclient.planner.astar import solve
from searchclient.Task import Task

if TYPE_CHECKING:
    pass


class ColorTasksTypedDict(TypedDict):
    future_box_tasks: deque[Task]
    future_agent_tasks: deque[Task]
    solved_tasks: deque[Task]


class Manager:

    def __init__(self) -> None:
        self.agents: list[Agent] = []
        self.profile: LevelProfile | None = None
        self.dist_map: DistanceMap | None = None
        self.timestep: int = 0

        # Task lists by agent color: {Color: {'future': [...], 'current': [...]}}
        self.color_tasks: dict[Color | None, ColorTasksTypedDict] = {}

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
        self._sync_agent_positions(initial_state)

        # # Initialize task lists for each agent
        # for agent in self.agents:
        #     self.agent_tasks[agent.agent_id] = {
        #         "future": deque(),
        #         "current": deque(),
        #         "solved": deque(),
        #     }

        # Initialize task lists for each agent color
        for agent in self.agents:
            agent_color = State.agent_colors[agent.agent_id]
            if agent_color not in self.color_tasks:
                # self.color_tasks[agent_color] = {"future": deque(), "current": deque()}
                self.color_tasks[agent_color] = ColorTasksTypedDict(
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

        # # Step 1c: Assign first task from each color group to respective agents
        # print("Assigning initial tasks to agents...", file=sys.stderr, flush=True)
        # self._assign_initial_tasks()

        # # Step 1d: Perform HCA* preplanning
        # print("Performing HCA* preplanning...", file=sys.stderr, flush=True)
        # self._hca_preplan(initial_state)

        print(
            f"Manager setup complete: {profile.num_agents} agents ready.",
            file=sys.stderr,
            flush=True,
        )

    def _build_box_goal_tasks(self) -> None:
        """
        Find all box goals in the profile and create move_box tasks.
        Add to color group future tasks.
        """
        assert self.profile is not None

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
                if box_char == goal_char:
                    task.object_pos = (br, bc)
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

    # def _assign_initial_tasks(self) -> None:
    #     """
    #     For each agent, assign the first available task from its color group's future tasks.
    #     Move that task from future to current.
    #     """

    #     for agent in self.agents:
    #         agent_color = State.agent_colors[agent.agent_id]
    #         color_bucket = self.color_tasks.get(agent_color)
    #         if color_bucket is not None:
    #             if color_bucket["future_box_tasks"]:
    #                 task = color_bucket["future_box_tasks"].popleft()
    #                 agent.task = task
    #                 self._sync_agent_task_state(agent.agent_id, task)
    #             elif color_bucket["future_agent_tasks"]:
    #                 task = color_bucket["future_agent_tasks"].popleft()
    #                 agent.task = task
    #                 self._sync_agent_task_state(agent.agent_id, task)

    #     # # Assign one agent per color group
    #     # assigned_per_color: dict[Color | None, int] = {}

    #     # for color, task_dict in self.color_tasks.items():
    #     #     if task_dict["future_box_tasks"]:
    #     #         # Find an agent with this color that has no current task
    #     #         task = task_dict["future_box_tasks"].popleft()

    #     #         for agent in self.agents:
    #     #             agent_color = State.agent_colors[agent.agent_id]

    #     #             if (
    #     #                 agent_color == color
    #     #                 and
    #     #             ):
    #     #                 self.agent_tasks[agent.agent_id]["current"].append(task)
    #     #                 color_bucket = self.color_tasks.setdefault(
    #     #                     color, {"future": deque(), "current": deque()}
    #     #                 )
    #     #                 color_bucket["current"].append(task)
    #     #                 self._sync_agent_task_state(agent.agent_id, task)
    #     #                 assigned_per_color[color] = agent.agent_id

    #     #                 color_name = color.name if color else "None"
    #     #                 print(
    #     #                     f"  Assigned task to Agent {agent.agent_id} (color {color_name})",
    #     #                     file=sys.stderr,
    #     #                     flush=True,
    #     #                 )
    #     #                 break

    def _sync_agent_task_state(self, agent_id: int, task: Task) -> None:
        """
        Keep the Agent's internal task/goal fields aligned with the manager task.
        This allows Agent.replan() to work after task completion.
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

    def _pop_next_task_for_agent(self, agent_id: int) -> Task | None:
        """
        Pull the next task for an agent, preferring the agent's color-group
        box tasks first and the agent's positional task last.
        """
        agent_color = State.agent_colors[agent_id]
        color_bucket = self.color_tasks.get(agent_color)

        if color_bucket is not None:
            if color_bucket["future_box_tasks"]:
                task = color_bucket["future_box_tasks"].popleft()
                return task
            elif color_bucket["future_agent_tasks"]:
                # check if this agent has already reached its final goal, if so then dont assign it the next one
                if self.agents[agent_id].has_reached_its_goal:
                    return None

                task = color_bucket["future_agent_tasks"].popleft()
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

        # finished_task = current_tasks.popleft()
        # self.agent_tasks[agent_id]["solved"].append(finished_task)
        self.color_tasks[State.agent_colors[agent_id]]["solved_tasks"].append(
            current_task
        )
        self.agents[agent_id].task = None
        self.agents[agent_id]._plan = []
        self.agents[agent_id]._plan_index = 0

        if current_task.task_type == "move_agent" and current_task.crucial is True:
            self.agents[agent_id].has_reached_its_goal = True

        # if current_task.task_type == "move_box" and current_task.box_char is not None:
        #     finished_color = State.box_colors[ord(current_task.box_char) - ord("A")]
        #     color_bucket = self.color_tasks.get(finished_color)
        #     if color_bucket is not None and color_bucket["current"]:
        #         for idx, task in enumerate(color_bucket["current"]):
        #             if (
        #                 task.task_type == finished_task.task_type
        #                 and task.box_char == finished_task.box_char
        #                 and task.goal_pos == finished_task.goal_pos
        #             ):
        #                 color_bucket["current"].remove(task)
        #                 break

        print(
            f"  Agent {agent_id}: current task solved, moved to solved list.",
            file=sys.stderr,
            flush=True,
        )

        next_task = self._pop_next_task_for_agent(agent_id)
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
        # return self.agents[agent_id].preplan(joint_state, self.timestep)

    # def _hca_preplan(self, initial_state: State) -> None:
    #     """
    #     Perform HCA* preplanning for each agent's current task.
    #     Use heuristics: distance from agent to box, box to goal.
    #     """
    #     from searchclient.planner.astar import solve

    #     for agent in self.agents:
    #         current_tasks = self.agent_tasks[agent.agent_id]["current"]
    #         if not current_tasks:
    #             continue

    #         current_task = current_tasks[0]

    #         # Detect same-colored obstacle before planning and swap tasks immediately.
    #         if current_task.task_type == "move_box":
    #             obstacle_info = self._find_same_color_obstacle(
    #                 initial_state, agent.agent_id, current_task
    #             )
    #             if obstacle_info is not None:
    #                 obs_r, obs_c, obs_char, can_reach_box = obstacle_info
    #                 print(
    #                     f"  Agent {agent.agent_id}: same-colored obstacle detected before planning; attempting swap.",
    #                     file=sys.stderr,
    #                     flush=True,
    #                 )
    #                 if self._swap_task_with_obstacle(
    #                     agent.agent_id,
    #                     current_task,
    #                     (obs_r, obs_c, obs_char),
    #                     can_reach_box,
    #                 ):
    #                     current_task = current_tasks[0]

    #         if current_task.task_type == "move_box":
    #             # Box task: agent must move box to goal
    #             goal_tuple = (
    #                 current_task.object_pos[0],
    #                 current_task.object_pos[1],
    #                 current_task.goal_pos[0],
    #                 current_task.goal_pos[1],
    #                 current_task.box_char,
    #             )
    #         else:
    #             # Agent task: just navigate to goal
    #             goal_tuple = (
    #                 None,
    #                 None,
    #                 current_task.goal_pos[0],
    #                 current_task.goal_pos[1],
    #             )

    #         plan = solve(
    #             state=initial_state,
    #             agent_id=agent.agent_id,
    #             goal=goal_tuple,
    #             constraints=set(),
    #             dist_map=self.dist_map,
    #         )

    #         if plan is None and current_task.task_type == "move_box":
    #             obstacle_info = self._find_same_color_obstacle(
    #                 initial_state, agent.agent_id, current_task
    #             )
    #             if obstacle_info is not None:
    #                 obs_r, obs_c, obs_char, can_reach_box = obstacle_info
    #                 print(
    #                     f"  Agent {agent.agent_id}: same-colored obstacle confirmed during planning; attempting swap.",
    #                     file=sys.stderr,
    #                     flush=True,
    #                 )
    #                 if self._swap_task_with_obstacle(
    #                     agent.agent_id,
    #                     current_task,
    #                     (obs_r, obs_c, obs_char),
    #                     can_reach_box,
    #                 ):
    #                     current_task = current_tasks[0]
    #                     if current_task.task_type == "move_box":
    #                         goal_tuple = (
    #                             current_task.object_pos[0],
    #                             current_task.object_pos[1],
    #                             current_task.goal_pos[0],
    #                             current_task.goal_pos[1],
    #                             current_task.box_char,
    #                         )
    #                     else:
    #                         goal_tuple = (
    #                             None,
    #                             None,
    #                             current_task.goal_pos[0],
    #                             current_task.goal_pos[1],
    #                         )
    #                     plan = solve(
    #                         state=initial_state,
    #                         agent_id=agent.agent_id,
    #                         goal=goal_tuple,
    #                         constraints=set(),
    #                         dist_map=self.dist_map,
    #                     )

    #         if plan:
    #             agent._plan = plan
    #             print(
    #                 f"  Agent {agent.agent_id}: HCA* found plan (length {len(plan)})",
    #                 file=sys.stderr,
    #                 flush=True,
    #             )
    #         else:
    #             print(
    #                 f"  Agent {agent.agent_id}: HCA* FAILED for task {current_task}",
    #                 file=sys.stderr,
    #                 flush=True,
    #             )

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
        if current_task is None:
            return False

        if len(agent._plan) == 0:
            obstacle_info = self._find_same_color_obstacle(
                joint_state, agent.agent_id, current_task
            )

            if obstacle_info is not None:
                managed_to_swap = self._swap_task_with_obstacle(
                    agent.agent_id,
                    current_task,
                    obstacle_info,
                )
                if managed_to_swap and agent.task is not None:
                    current_task = agent.task
                else:
                    print(
                        f"  Agent {agent_id}: failed to swap with obstacle task, will attempt to plan with original task.",
                        file=sys.stderr,
                        flush=True,
                    )

            # other edge cases here i guess

            goal_tuple = self._convert_task_to_goal_tuple(current_task)
            plan = solve(
                state=joint_state,
                agent_id=agent.agent_id,
                goal=goal_tuple,
                constraints=set(),
                dist_map=self.dist_map,
            )

            if plan:
                agent._plan = plan
                agent._plan_index = 0
                print(
                    f"  Agent {agent_id}: preplan found (length {len(plan)})",
                    file=sys.stderr,
                    flush=True,
                )
                return True
            else:
                print(
                    f"  Agent {agent_id}: preplan FAILED for task {current_task}",
                    file=sys.stderr,
                    flush=True,
                )
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

    def _find_same_color_obstacle(
        self,
        initial_state: State,
        agent_id: int,
        failed_task: Task,
    ) -> tuple[int, int, str, bool] | None:
        """
        Return the first blocking box with the same color as failed_task.
        Also check if agent can actually reach the target box by running A* with only same-colored boxes.

        Returns (obs_r, obs_c, obs_char, can_reach_box) where:
        - can_reach_box is True if agent can reach the box (A* finds a path avoiding same-colored boxes)
        - can_reach_box is False if agent cannot reach the box (blocked by same-colored boxes)
        """
        from searchclient.planner.astar import solve

        if failed_task.box_char is None:
            return None

        agent = self.agents[agent_id]
        agent_r = initial_state.agent_rows[agent_id]
        agent_c = initial_state.agent_cols[agent_id]

        obstacle = agent._find_path_obstacle(
            initial_state,
            agent_r,
            agent_c,
            failed_task.object_pos[0],
            failed_task.object_pos[1],
        )
        if obstacle is None:
            return None

        obs_r, obs_c, obs_char = obstacle
        target_color = State.box_colors[ord(failed_task.box_char) - ord("A")]
        obstacle_color = State.box_colors[ord(obs_char) - ord("A")]
        if target_color is None or obstacle_color != target_color:
            return None

        # Check if agent can reach the target box by running A* with only same-colored boxes.
        # Create a modified state with all non-target boxes removed.
        modified_boxes = [row[:] for row in initial_state.boxes]  # Deep copy 2D list

        for r in range(len(modified_boxes)):
            for c in range(len(modified_boxes[r])):
                box_char = modified_boxes[r][c]
                if box_char == "":  # empty slot
                    continue
                box_color = State.box_colors[ord(box_char) - ord("A")]
                if box_color != target_color:
                    modified_boxes[r][c] = ""  # Remove non-target boxes

        modified_state = State(
            agent_rows=initial_state.agent_rows[:],
            agent_cols=initial_state.agent_cols[:],
            boxes=modified_boxes,
        )

        # Run A* from agent to target box position (navigation only, no pushing)
        goal = (None, None, failed_task.object_pos[0], failed_task.object_pos[1])
        plan = solve(
            state=modified_state,
            agent_id=agent_id,
            goal=goal,
            constraints=set(),
            dist_map=self.dist_map,
        )

        can_reach_box = plan is not None

        return (obs_r, obs_c, obs_char, can_reach_box)

    def _swap_task_with_obstacle(
        self,
        agent_id: int,
        failed_task: Task,
        obstacle: tuple[int, int, str, bool],
    ) -> bool:
        """
        Swap current task with an obstacle task if possible.

        Args:
            agent_id: ID of the agent with the failed task
            failed_task: The task that failed to plan
            obstacle: (obs_r, obs_c, obs_char) position and char of the blocking box
            can_reach_box: True if agent can reach the box (obstacle before box),
                          False if agent cannot reach the box (obstacle at box position)
        """
        assert self.profile is not None

        obs_r, obs_c, obs_char, can_reach_box = obstacle

        print(
            f"  Same-colored obstacle detected: box {obs_char} at ({obs_r}, {obs_c}) blocks Agent {agent_id}.",
            file=sys.stderr,
            flush=True,
        )

        # current_tasks = self.agent_tasks[agent_id]["current"]
        # if not current_tasks:
        #     print(
        #         f"  Swap failed: Agent {agent_id} has no current task list.",
        #         file=sys.stderr,
        #         flush=True,
        #     )
        #     return False

        current_task = self.agents[agent_id].task
        if current_task is None:
            print(
                f"  Swap failed: Agent {agent_id} has no current task.",
                file=sys.stderr,
                flush=True,
            )
            return False

        assert failed_task.box_char is not None
        target_color = State.box_colors[ord(failed_task.box_char) - ord("A")]
        assert target_color is not None

        # First try to find that obstacle task in the color future list.
        # color_bucket = self.color_tasks.setdefault(
        #     target_color, {"future": deque(), "current": deque()}
        # )
        # future_tasks = color_bucket["future"]

        obstacle_index = next(
            (
                i
                for i, task in enumerate(
                    self.color_tasks[target_color]["future_box_tasks"]
                )
                if task.task_type == "move_box"
                and task.box_char == obs_char
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
                and other_current.box_char == obs_char
                and other_current.object_pos == (obs_r, obs_c)
            ):
                obstacle_assigned_to_agent_id = other_agent.agent_id
                print(
                    f"Obstacle task for box {obs_char} at ({obs_r}, {obs_c}) is currently assigned to Agent {other_agent.agent_id}.",
                    file=sys.stderr,
                    flush=True,
                )
                break

        if obstacle_index is None and obstacle_assigned_to_agent_id is None:

            new_task = Task(
                task_type="move_box",
                object_pos=(
                    (failed_task.object_pos) if can_reach_box else (obs_r, obs_c)
                ),
                goal_pos=(
                    self.agents[agent_id].agent_row,
                    self.agents[agent_id].agent_col,
                ),
                box_char=obs_char,
            )

            # # remove failed_task from current tasks and add to front of future tasks
            failed_task = current_task
            if can_reach_box is True:
                failed_task.object_pos = (obs_r, obs_c)

            self.color_tasks[target_color]["future_box_tasks"].appendleft(failed_task)

            self.agents[agent_id].task = new_task
            self._sync_agent_task_state(agent_id, new_task)

            print(
                f"Swapped tasks, was able to reach box directly: {can_reach_box}.",
                file=sys.stderr,
                flush=True,
            )
            # print(
            #     f"  No future task found for obstacle box {obs_char} at ({obs_r}, {obs_c}) nor any current task found for other agents. "
            #     + " Just adding a new task to the front of the current task list for Agent {agent_id}.",
            #     file=sys.stderr,
            #     flush=True,
            # )

            return True

        print("Swapping edgecase NOT IMPLEMENTED YET")
        return False

        # if obstacle_index is not None:
        #     obstacle_task = future_tasks.pop(obstacle_index)

        #     obstacle_goal_x, obstacle_goal_y = obstacle_task.goal_pos
        #     failed_goal_x, failed_goal_y = failed_task.goal_pos

        #     obstacle_task.goal_pos = (failed_goal_x, failed_goal_y)
        #     failed_task.goal_pos = (obstacle_goal_x, obstacle_goal_y)

        #     future_tasks.insert(0, failed_task)
        #     current_tasks[0] = obstacle_task
        #     print(
        #         f"  Swapped current task with future obstacle task for Agent {agent_id}.",
        #         file=sys.stderr,
        #         flush=True,
        #     )
        #     return True

        # # NOTE: now we need to consider 2 cases:
        # """
        # 0. obstacle doesnt have a task
        # 1. same coloured agent but different one has that task of the obstacle - we will skip this one for now
        # """

        # for other_agent in self.agents:
        #     if other_agent.agent_id == agent_id:
        #         continue

        #     other_current = self.agent_tasks[other_agent.agent_id]["current"]
        #     for idx, task in enumerate(other_current):
        #         if (
        #             task.task_type == "move_box"
        #             and task.box_char == obs_char
        #             and task.object_pos == (obs_r, obs_c)
        #         ):
        #             print(
        #                 f"  Found another agent {other_agent.agent_id} currently "
        #                 + f"assigned to the obstacle task. This case IS NOT YET IMPLEMENTED, so swap failed for Agent {agent_id}.",
        #                 file=sys.stderr,
        #                 flush=True,
        #             )
        #             return False

        # print(
        #     f"No future task found for obstacle box {obs_char} at ({obs_r}, {obs_c}) nor any current task found for other agents. "
        #     + " Just replacing the current task with the obstacle task",
        #     file=sys.stderr,
        #     flush=True,
        # )
        # self.agent_tasks[agent_id]["current"][0].object_pos = (obs_r, obs_c)

        # return True

        # Fallback: look for another agent that currently owns the obstacle task.
        # for other_agent in self.agents:
        #     other_current = self.agent_tasks[other_agent.agent_id]["current"]
        #     for idx, task in enumerate(other_current):
        # if (
        #     task.task_type == "move_box"
        #     and task.box_char == obs_char
        #             and task.object_pos == (obs_r, obs_c)
        #         ):
        #             other_current[idx] = failed_task
        #             current_tasks[0] = task
        #             print(
        #                 f"  Swapped current tasks between Agent {agent_id} and Agent {other_agent.agent_id}.",
        #                 file=sys.stderr,
        #                 flush=True,
        #             )
        #             return True

        print(
            f"  Swap failed: no task entry found for obstacle box {obs_char} at ({obs_r}, {obs_c}).",
            file=sys.stderr,
            flush=True,
        )

        return False

    # ------------------------------------------------------------------
    # Main loop (called every turn)
    # ------------------------------------------------------------------

    def get_joint_action(self, joint_state: State) -> list[Action]:
        """
        Execute the next timestep: all agents execute their next planned action.
        """
        self._sync_agent_positions(joint_state)

        # BDI-style task completion handling: move solved tasks aside, then
        # assign and replan if another task is available.
        for agent in self.agents:

            if agent.task is None:
                agent.task = self._pop_next_task_for_agent(agent.agent_id)
                if agent.task is not None:
                    self._sync_agent_task_state(agent.agent_id, agent.task)

            self._maybe_advance_completed_task_or_preplan(joint_state, agent.agent_id)

        joint_action = []
        for agent in self.agents:
            action = agent.next_action()
            joint_action.append(action)

        self.timestep += 1
        return joint_action

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
