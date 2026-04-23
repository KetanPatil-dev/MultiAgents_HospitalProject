"""
Joint state of all agents and boxes.

Class-level attributes (walls, goals, agent_colors, box_colors) are set once by
LevelParser.parse() and never mutated afterwards.
"""

from __future__ import annotations

import random
from typing import ClassVar

from searchclient.action import Action, ActionType
from searchclient.color import Color


class State:
    _RNG = random.Random(1)

    # Set once at startup by LevelParser.parse() — read-only after that.
    agent_colors: ClassVar[list[Color | None]] = []
    box_colors: ClassVar[list[Color | None]] = []
    walls: ClassVar[list[list[bool]]] = []
    goals: ClassVar[list[list[str]]] = []  # 'A'-'Z' | '0'-'9' | ''

    def __init__(
        self,
        agent_rows: list[int],
        agent_cols: list[int],
        boxes: list[list[str]],
    ) -> None:
        self.agent_rows = agent_rows  # indexed by agent id
        self.agent_cols = agent_cols
        self.boxes = boxes            # boxes[row][col] in 'A'-'Z' | ''
        self.parent: State | None = None
        self.joint_action: list[Action] | None = None
        self.g: int = 0
        self._hash: int | None = None

    # ------------------------------------------------------------------
    # Applicability / conflict (joint-action level)
    # ------------------------------------------------------------------

    def is_applicable(self, agent: int, action: Action) -> bool:
        ar, ac = self.agent_rows[agent], self.agent_cols[agent]

        if action.type is ActionType.NoOp:
            return True

        if action.type is ActionType.Move:
            nr, nc = ar + action.agent_row_delta, ac + action.agent_col_delta
            return self.is_free(nr, nc)

        if action.type is ActionType.Push:
            br, bc = ar + action.agent_row_delta, ac + action.agent_col_delta
            bdr, bdc = br + action.box_row_delta, bc + action.box_col_delta
            if not self.boxes[br][bc]:
                return False
            box_idx = ord(self.boxes[br][bc]) - ord('A')
            if not Color.compatible(State.agent_colors[agent], State.box_colors[box_idx]):
                return False
            return self.is_free(bdr, bdc)

        if action.type is ActionType.Pull:
            nr, nc = ar + action.agent_row_delta, ac + action.agent_col_delta
            br, bc = ar - action.box_row_delta, ac - action.box_col_delta
            if not self.is_free(nr, nc):
                return False
            if not self.boxes[br][bc]:
                return False
            box_idx = ord(self.boxes[br][bc]) - ord('A')
            return Color.compatible(State.agent_colors[agent], State.box_colors[box_idx])

        return False

    def is_conflicting(self, joint_action: list[Action]) -> bool:
        dest: set[tuple[int, int]] = set()
        box_dest: set[tuple[int, int]] = set()

        for agent, action in enumerate(joint_action):
            ar, ac = self.agent_rows[agent], self.agent_cols[agent]

            if action.type is ActionType.NoOp:
                continue

            if action.type is ActionType.Move:
                nr, nc = ar + action.agent_row_delta, ac + action.agent_col_delta
                if (nr, nc) in dest:
                    return True
                dest.add((nr, nc))

            elif action.type is ActionType.Push:
                br, bc = ar + action.agent_row_delta, ac + action.agent_col_delta
                bdr, bdc = br + action.box_row_delta, bc + action.box_col_delta
                if (bdr, bdc) in box_dest:
                    return True
                box_dest.add((bdr, bdc))
                dest.add((br, bc))

            elif action.type is ActionType.Pull:
                nr, nc = ar + action.agent_row_delta, ac + action.agent_col_delta
                if (nr, nc) in dest:
                    return True
                dest.add((nr, nc))
                box_dest.add((ar, ac))

        return False

    def result(self, joint_action: list[Action]) -> State:
        rows = self.agent_rows[:]
        cols = self.agent_cols[:]
        boxes = [row[:] for row in self.boxes]

        for agent, action in enumerate(joint_action):
            ar, ac = rows[agent], cols[agent]

            if action.type is ActionType.Move:
                rows[agent] = ar + action.agent_row_delta
                cols[agent] = ac + action.agent_col_delta

            elif action.type is ActionType.Push:
                br, bc = ar + action.agent_row_delta, ac + action.agent_col_delta
                bdr, bdc = br + action.box_row_delta, bc + action.box_col_delta
                boxes[bdr][bdc] = boxes[br][bc]
                boxes[br][bc] = ""
                rows[agent], cols[agent] = br, bc

            elif action.type is ActionType.Pull:
                nr, nc = ar + action.agent_row_delta, ac + action.agent_col_delta
                br, bc = ar - action.box_row_delta, ac - action.box_col_delta
                boxes[ar][ac] = boxes[br][bc]
                boxes[br][bc] = ""
                rows[agent], cols[agent] = nr, nc

        s = State(rows, cols, boxes)
        s.parent = self
        s.joint_action = joint_action[:]
        s.g = self.g + 1
        return s

    def is_goal_state(self) -> bool:
        for r, row in enumerate(State.goals):
            for c, g in enumerate(row):
                if 'A' <= g <= 'Z' and self.boxes[r][c] != g:
                    return False
                if '0' <= g <= '9':
                    aid = ord(g) - ord('0')
                    if self.agent_rows[aid] != r or self.agent_cols[aid] != c:
                        return False
        return True

    # ------------------------------------------------------------------
    # Expansion — O(b^n_agents), only for small n or legacy use
    # ------------------------------------------------------------------

    def get_expanded_states(self) -> list[State]:
        num_agents = len(self.agent_rows)
        applicable = [
            [a for a in Action if self.is_applicable(ag, a)]
            for ag in range(num_agents)
        ]

        joint_action = [Action.NoOp] * num_agents
        perm = [0] * num_agents
        expanded: list[State] = []

        while True:
            for ag in range(num_agents):
                joint_action[ag] = applicable[ag][perm[ag]]

            if not self.is_conflicting(joint_action):
                expanded.append(self.result(joint_action))

            done = False
            for ag in range(num_agents):
                if perm[ag] < len(applicable[ag]) - 1:
                    perm[ag] += 1
                    break
                else:
                    perm[ag] = 0
                    if ag == num_agents - 1:
                        done = True
            if done:
                break

        State._RNG.shuffle(expanded)
        return expanded

    def extract_plan(self) -> list[list[Action]]:
        plan: list[list[Action]] = []
        s: State | None = self
        while s is not None and s.joint_action is not None:
            plan.append(s.joint_action)
            s = s.parent
        plan.reverse()
        return plan

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_free(self, r: int, c: int) -> bool:
        return (
            0 <= r < len(State.walls)
            and 0 <= c < len(State.walls[0])
            and not State.walls[r][c]
            and not self.boxes[r][c]
            and self.agent_at(r, c) is None
        )

    def agent_at(self, r: int, c: int) -> str | None:
        for i, (ar, ac) in enumerate(zip(self.agent_rows, self.agent_cols)):
            if ar == r and ac == c:
                return chr(i + ord('0'))
        return None

    # ------------------------------------------------------------------
    # Hash / equality — only mutable fields
    # ------------------------------------------------------------------

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((
                tuple(self.agent_rows),
                tuple(self.agent_cols),
                tuple(tuple(row) for row in self.boxes),
            ))
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return False
        return (
            self.agent_rows == other.agent_rows
            and self.agent_cols == other.agent_cols
            and self.boxes == other.boxes
        )

    def __repr__(self) -> str:
        lines = []
        for r, row in enumerate(self.boxes):
            line = []
            for c, cell in enumerate(row):
                if cell:
                    line.append(cell)
                elif State.walls[r][c]:
                    line.append("+")
                else:
                    a = self.agent_at(r, c)
                    line.append(a if a is not None else " ")
            lines.append("".join(line))
        return "\n".join(lines)
