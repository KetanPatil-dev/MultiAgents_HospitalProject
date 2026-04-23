"""
CAPA 4 — Manager multiagente.

Responsabilidades:
  - setup(): inicializa agentes, distancias, asignación de tareas.
  - get_joint_action(): BDI loop + resolución de conflictos cada turno.
  - resolve_conflicts(): CBS greedy + social laws + deadlock detection.

Blackboard compartido: planes activos y reservas space-time (r, c, t).
"""

from __future__ import annotations

import sys
from typing import TextIO

from searchclient.action import Action
from searchclient.agent import Agent
from searchclient.heuristics import DistanceMap
from searchclient.level_parser import LevelParser, LevelProfile
from searchclient.state import State


class Manager:
    """
    Coordinates all agents each timestep.

    Usage:
        manager = Manager()
        initial_state, profile = LevelParser.parse(server_messages)
        manager.setup(initial_state, profile)
        while not done:
            joint_action = manager.get_joint_action(current_state, t)
            current_state = current_state.result(joint_action)
            t += 1
    """

    def __init__(self) -> None:
        self.agents: list[Agent] = []
        self.profile: LevelProfile | None = None
        self.dist_map: DistanceMap | None = None

        # Blackboard: space-time reservations (r, c, t) → agent_id
        self.reservations: dict[tuple[int, int, int], int] = {}

        self.timestep: int = 0

    # ------------------------------------------------------------------
    # Setup (once)
    # ------------------------------------------------------------------

    def setup(self, initial_state: State, profile: LevelProfile) -> None:
        """
        Initialise agents, precompute distances, run task assignment.
        Called once after parsing.
        """
        self.profile = profile
        self.dist_map = DistanceMap.from_state(initial_state)

        # Create one Agent per agent in the level
        self.agents = [
            Agent(i, self.dist_map, profile)
            for i in range(profile.num_agents)
        ]

        # Task assignment (Hungarian)
        assignment = self._assign_tasks(initial_state)
        for agent_id, tasks in assignment.items():
            self.agents[agent_id].assign_tasks(tasks)

        print(
            f"Manager setup: {profile.num_agents} agents, "
            f"{len(profile.real_boxes)} plannable boxes, "
            f"{len(profile.deco_boxes)} decorative.",
            file=sys.stderr, flush=True,
        )

    # ------------------------------------------------------------------
    # Main loop (called every turn)
    # ------------------------------------------------------------------

    def get_joint_action(self, joint_state: State) -> list[Action]:
        """
        BDI loop for all agents → joint action for this timestep.
        """
        t = self.timestep
        joint_action: list[Action] = []

        for agent in self.agents:
            # Sense
            agent.sense(joint_state, t)

            # Plan if needed
            if not agent.plan_is_sound(joint_state, t):
                success = agent.replan(joint_state, t)
                if not success:
                    print(
                        f"Agent {agent.agent_id} could not find a plan at t={t}.",
                        file=sys.stderr, flush=True,
                    )

            joint_action.append(agent.next_action())

        # Resolve conflicts before returning
        joint_action = self._resolve_conflicts(joint_state, joint_action)

        self.timestep += 1
        return joint_action

    # ------------------------------------------------------------------
    # Conflict resolution (CBS greedy + social laws)
    # ------------------------------------------------------------------

    def _resolve_conflicts(
        self, joint_state: State, joint_action: list[Action]
    ) -> list[Action]:
        """
        Detect and resolve conflicts in the proposed joint action.

        Resolution cascade (cheapest first):
          1. Social laws in corridors (lower-id agent yields).
          2. CBS greedy: find first conflict → constrain lower-priority agent → replan.
          3. Force NoOp as last resort to break deadlock.
        """
        # TODO(P4): implement full CBS greedy resolution
        # For now: if the joint action is conflicting, make all agents NoOp except agent 0
        if joint_state.is_conflicting(joint_action):
            print(
                f"Conflict detected at t={self.timestep} — stub: NoOp all but agent 0.",
                file=sys.stderr, flush=True,
            )
            for i in range(1, len(joint_action)):
                joint_action[i] = Action.NoOp

        return joint_action

    def _detect_deadlock(self) -> list[int]:
        """
        Return list of agent ids in a wait cycle (DFS on wait graph).
        Empty list if no deadlock.
        """
        # TODO(P4): build wait graph from current plans and detect cycles
        return []

    # ------------------------------------------------------------------
    # Task assignment
    # ------------------------------------------------------------------

    def _assign_tasks(
        self, initial_state: State
    ) -> dict[int, list[tuple[int, int, int, int]]]:
        """
        Calls Hungarian planner to assign boxes to agents.
        Returns dict {agent_id: [(box_r, box_c, goal_r, goal_c), ...]}.
        """
        # TODO(P5): replace stub with real Hungarian assignment
        # from searchclient.planner.hungarian import assign
        # return assign(self.agents, self.profile.real_boxes, self.profile.box_goals, self.dist_map)
        return {i: [] for i in range(len(self.agents))}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_done(self, joint_state: State) -> bool:
        return joint_state.is_goal_state()
