"""
CAPA 4 — Agente BDI individual.

Cada Agent encapsula:
  - Beliefs: posición actual, caja asignada, plan en curso.
  - Desires: llevar su caja al goal.
  - Intentions: siguiente acción del plan.

El Manager crea un Agent por agente y llama next_action() cada turno.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from searchclient.action import Action

if TYPE_CHECKING:
    from searchclient.state import State
    from searchclient.heuristics import DistanceMap
    from searchclient.level_parser import LevelProfile


class Agent:
    """
    BDI agent.  Planifica independientemente; el Manager coordina conflictos.

    Cycle each turn:
        1. sense()      — update beliefs from joint state
        2. plan_is_sound() — check if current plan is still valid
        3. replan() if needed
        4. next_action() — return next Action from plan
    """

    def __init__(
        self,
        agent_id: int,
        dist_map: DistanceMap,
        profile: LevelProfile,
    ) -> None:
        self.agent_id = agent_id
        self.dist_map = dist_map
        self.profile = profile

        # Current assignment: list of (box_row, box_col, goal_row, goal_col)
        self.tasks: list[tuple[int, int, int, int]] = []

        # Current plan: sequence of Actions for this agent
        self._plan: list[Action] = []
        self._plan_index: int = 0

        # Space-time constraints: set of (row, col, timestep) reserved by others
        self.constraints: set[tuple[int, int, int]] = set()

    # ------------------------------------------------------------------
    # BDI interface called by Manager
    # ------------------------------------------------------------------

    def sense(self, joint_state: State, timestep: int) -> None:
        """Update beliefs from the current joint state."""
        # TODO(P4): update internal belief about agent position from joint_state
        pass

    def plan_is_sound(self, joint_state: State, timestep: int) -> bool:
        """
        Return True iff the current plan is still valid given constraints.
        Returns False → Manager will call replan().
        """
        # TODO(P4): check remaining plan steps against current constraints
        if not self._plan or self._plan_index >= len(self._plan):
            return False
        return True

    def replan(self, joint_state: State, timestep: int) -> bool:
        """
        Invoke planner to compute a new plan.
        Returns True on success, False if no plan found.
        """
        # TODO(P4): select planner (A* or IW) based on profile, call solve()
        # from searchclient.planner.astar import solve
        # plan = solve(joint_state, self.agent_id, self.tasks[0], self.constraints)
        # if plan is None:
        #     return False
        # self._plan = plan
        # self._plan_index = 0
        # return True
        self._plan = []
        self._plan_index = 0
        return False

    def next_action(self) -> Action:
        """Return the next action for this agent (NoOp if plan exhausted)."""
        if self._plan_index < len(self._plan):
            action = self._plan[self._plan_index]
            self._plan_index += 1
            return action
        return Action.NoOp

    def assign_tasks(self, tasks: list[tuple[int, int, int, int]]) -> None:
        """Called by Manager after Hungarian assignment. Clears old plan."""
        self.tasks = tasks
        self._plan = []
        self._plan_index = 0

    def is_done(self) -> bool:
        """True iff all assigned tasks are completed."""
        # TODO(P4): check goal conditions for each assigned box
        return not self.tasks

    def remaining_plan(self) -> list[Action]:
        return self._plan[self._plan_index:]

    def inject_constraint(self, r: int, c: int, t: int) -> None:
        """Add a space-time constraint (r, c, t) and invalidate plan if needed."""
        self.constraints.add((r, c, t))
