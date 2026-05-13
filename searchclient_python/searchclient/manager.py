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
        self.agents_awaiting_other_agent: dict[int, int | None] = {}

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
        self._sync_agent_positions(initial_state)

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

                # closest_foreign_color_agent = None
                # closest_distance = float("inf")
                # for other_agent in self.agents:
                #     if other_agent.agent_id == agent_id:
                #         continue
                #     if State.agent_colors[other_agent.agent_id] == box_obstacles[0][2]:
                #         if self.dist_map is None:
                #             print(
                #                 f"  Agent {agent_id}: distance map not initialized, cannot evaluate obstacle proximity.",
                #                 file=sys.stderr,
                #                 flush=True,
                #             )
                #             break
                #         bfs_table = self.dist_map._bfs(
                #             other_agent.agent_row,
                #             other_agent.agent_col,
                #             # box_obstacles[0][0],
                #             # box_obstacles[0][1],
                #         )
                #         dist = bfs_table[box_obstacles[0][0]][box_obstacles[0][1]]
                #         if dist is not None and dist < closest_distance:
                #             closest_distance = dist
                #             closest_foreign_color_agent = other_agent

                # if closest_foreign_color_agent is None:
                #     print(
                #         f"  Agent {agent_id}: no agents of color {box_obstacles[0][2]} found to evaluate obstacle proximity.",
                #         file=sys.stderr,
                #         flush=True,
                #     )
                #     return False

                # print(
                #     f"  Agent {agent_id}: closest agent of color {box_obstacles[0][2]} is Agent {closest_foreign_color_agent.agent_id} at distance {closest_distance}.",
                #     file=sys.stderr,
                #     flush=True,
                # )
                foreign_agent = next(
                    agent
                    for agent in self.agents
                    if agent.agent_id != agent_id
                    and State.agent_colors[agent.agent_id] == box_obstacles[0][2]
                )

                if self.agents_awaiting_other_agent[agent_id] is None:

                    managed_foreign_to_swap = self._swap_foreign_task_with_obstacle(
                        foreign_agent.agent_id,
                        box_obstacles[0],
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
            # elif (
            #     obstacles is not None
            #     and len(obstacles) > 0
            #     and obstacles[0][4] == "agent"
            # ):
            #     print(
            #         f"  Agent {agent_id}: agent obstacle detected at ({obstacles[0][0]}, {obstacles[0][1]}). Will attempt to re-plan with original task and hope to find a way around the agent obstacle.",
            #         file=sys.stderr,
            #         flush=True,
            #     )

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
                self.agents_awaiting_other_agent[agent_id] = None
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

    def _swap_foreign_task_with_obstacle(
        self,
        foreign_agent_id: int,
        obstacle: tuple[int, int, Color, bool, str],
    ) -> bool:
        """
        Swap the task of a foreign agent with an obstacle task if possible.

        Args:
            foreign_agent_id: ID of the agent to attempt the swap on
            obstacle: (obs_r, obs_c, obs_color, obstacle_after_box, obstacle_type) position and color of the blocking box


        """
        obs_r, obs_c, obs_color, obstacle_after_box, obstacle_type = obstacle

        current_task = self.agents[foreign_agent_id].task

        new_task = Task(
            task_type="move_box",
            object_pos=(obs_r, obs_c),
            goal_pos=(
                self.agents[foreign_agent_id].agent_row,
                self.agents[foreign_agent_id].agent_col,
            ),
            box_char=State.get_box_char_from_color(obs_color),
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
                task.object_pos = (obs_r, obs_c)
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

        return True

    def _swap_task_with_obstacle(
        self,
        agent_id: int,
        obstacle: tuple[int, int, Color, bool, str],
    ) -> bool:
        """
        Swap current task with an obstacle task if possible.

        Args:
            agent_id: ID of the agent with the failed task
            failed_task: The task that failed to plan
            obstacle: (obs_r, obs_c, obs_color, obstacle_after_box, obstacle_type) position and color of the blocking box


        """
        assert self.profile is not None

        obs_r, obs_c, obs_color, obstacle_after_box, obstacle_type = obstacle

        print(
            f"  Same-colored obstacle detected: box {obs_color} at ({obs_r}, {obs_c}) blocks Agent {agent_id}.",
            file=sys.stderr,
            flush=True,
        )

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
                    box_char=State.get_box_char_from_color(obs_color),
                )

                if obstacle_index is not None:
                    print(
                        f"  Found future task for obstacle box {obs_color} at ({obs_r}, {obs_c}) in color future tasks at index {obstacle_index}.",
                        file=sys.stderr,
                        flush=True,
                    )
                    print(
                        f"Updating that future goal to the goal_pos after the swapping: {new_task.goal_pos}"
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
                    box_char=State.get_box_char_from_color(obs_color),
                )

                if obstacle_index is not None:
                    print(
                        f"  Found future task for obstacle box {obs_color} at ({obs_r}, {obs_c}) in color future tasks at index {obstacle_index}.",
                        file=sys.stderr,
                        flush=True,
                    )
                    print(
                        f"Updating that future goal to the goal_pos after the swapping: {new_task.goal_pos}"
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

            return True

        print("Swapping edgecase NOT IMPLEMENTED YET")
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
        t = self.timestep
        for agent in self.agents:

            if agent.task is None:
                agent.task = self._pop_next_task_for_agent(agent.agent_id)
                if agent.task is not None:
                    self._sync_agent_task_state(agent.agent_id, agent.task)

            self._maybe_advance_completed_task_or_preplan(joint_state, agent.agent_id)
            # agent.sense(joint_state, t)
            # if not agent.plan_is_sound(joint_state, t):
            #     success = agent.replan(joint_state, t)
            #     if not success:
            #         print(
            #             f"t={t} Agent {agent.agent_id}: replan failed.",
            #             file=sys.stderr,
            #             flush=True,
            #         )
        joint_action = []
        for agent in self.agents:
            action = agent.next_action()
            joint_action.append(action)

        # Validate joint action
        is_valid = not joint_state.is_conflicting(joint_action)
        print(f"Joint action valid: {is_valid}", file=sys.stderr, flush=True)

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
