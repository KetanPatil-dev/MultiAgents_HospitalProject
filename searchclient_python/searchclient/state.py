import random
from typing import ClassVar

from searchclient.action import Action, ActionType
from searchclient.color import Color


class State:
    _RNG = random.Random(1)

    agent_colors: ClassVar[list[Color | None]]
    walls: ClassVar[list[list[bool]]]
    box_colors: ClassVar[list[Color | None]]
    goals: ClassVar[list[list[str]]]

    def __init__(self, agent_rows: list[int], agent_cols: list[int], boxes: list[list[str]]) -> None:
        """
        Constructs an initial state.
        Arguments are not copied, and therefore should not be modified after being passed in.

        The lists walls, boxes, and goals are indexed from top-left of the level, row-major order (row, col).
               Col 0  Col 1  Col 2  Col 3
        Row 0: (0,0)  (0,1)  (0,2)  (0,3)  ...
        Row 1: (1,0)  (1,1)  (1,2)  (1,3)  ...
        Row 2: (2,0)  (2,1)  (2,2)  (2,3)  ...
        ...

        For example, State.walls[2] is a list of booleans for the third row.
        State.walls[row][col] is True if there's a wall at (row, col).

        The agent rows and columns are indexed by the agent number.
        For example, State.agent_rows[0] is the row location of agent '0'.

        Note: The state should be considered immutable after it has been hashed, e.g. added to a dictionary or set.
        """
        self.agent_rows = agent_rows
        self.agent_cols = agent_cols
        self.boxes = boxes
        self.parent: State | None = None
        self.joint_action: list[Action] | None = None
        self.g = 0
        self._hash: int | None = None

    # def result(self, joint_action: list[Action]) -> "State":
    #     """
    #     Returns the state resulting from applying joint_action in this state.
    #     Precondition: Joint action must be applicable and non-conflicting in this state.
    #     """

    #     # Copy this state.
    #     copy_agent_rows = self.agent_rows[:]
    #     copy_agent_cols = self.agent_cols[:]
    #     copy_boxes = [row[:] for row in self.boxes]

    #     # Apply each action.
    #     for agent, action in enumerate(joint_action):
    #         if action.type is ActionType.NoOp:
    #             pass

    #         elif action.type is ActionType.Move:
    #             copy_agent_rows[agent] += action.agent_row_delta
    #             copy_agent_cols[agent] += action.agent_col_delta

    #     copy_state = State(copy_agent_rows, copy_agent_cols, copy_boxes)

    #     copy_state.parent = self
    #     copy_state.joint_action = joint_action.copy()
    #     copy_state.g = self.g + 1

    #     return copy_state


    def result(self, joint_action: list[Action]) -> "State":
        copy_agent_rows = self.agent_rows[:]
        copy_agent_cols = self.agent_cols[:] 
        copy_boxes = [row[:] for row in self.boxes] # 2D matrix of cells

        for agent, action in enumerate(joint_action): # joint_action list of actions of all agents
            if action.type is ActionType.Move:
                copy_agent_rows[agent] += action.agent_row_delta
                copy_agent_cols[agent] += action.agent_col_delta

            elif action.type is ActionType.Push:
                # Move box first
                box_row = copy_agent_rows[agent] + action.agent_row_delta
                box_col = copy_agent_cols[agent] + action.agent_col_delta
                box_dest_row = box_row + action.box_row_delta
                box_dest_col = box_col + action.box_col_delta
                
                # Change the matrix (erase old put new)
                box_char = copy_boxes[box_row][box_col]
                copy_boxes[box_dest_row][box_dest_col] = box_char
                copy_boxes[box_row][box_col] = ""
                
                # Move agent to box's old position
                copy_agent_rows[agent] = box_row
                copy_agent_cols[agent] = box_col

            elif action.type is ActionType.Pull:
                # Move box to agent's old position
                box_row = copy_agent_rows[agent] - action.box_row_delta
                box_col = copy_agent_cols[agent] - action.box_col_delta
                box_dest_row = copy_agent_rows[agent]
                box_dest_col = copy_agent_cols[agent]
                
                box_char = copy_boxes[box_row][box_col]
                copy_boxes[box_dest_row][box_dest_col] = box_char
                copy_boxes[box_row][box_col] = ""
                
                # Move agent to its destination
                copy_agent_rows[agent] += action.agent_row_delta
                copy_agent_cols[agent] += action.agent_col_delta

        copy_state = State(copy_agent_rows, copy_agent_cols, copy_boxes)
        copy_state.parent = self
        copy_state.joint_action = joint_action.copy()
        copy_state.g = self.g + 1
        return copy_state
    

    # def result(self, joint_action: list[Action]) -> "State":
    #     copy_agent_rows = self.agent_rows[:]
    #     copy_agent_cols = self.agent_cols[:]
    #     copy_boxes = [row[:] for row in self.boxes]

    #     for agent, action in enumerate(joint_action):
    #         if action.type is ActionType.Move:
    #             copy_agent_rows[agent] += action.agent_row_delta
    #             copy_agent_cols[agent] += action.agent_col_delta

    #         elif action.type is ActionType.Push:
    #             # Usar posición ORIGINAL del agente
    #             agent_row = self.agent_rows[agent]
    #             agent_col = self.agent_cols[agent]
                
    #             box_row = agent_row + action.agent_row_delta
    #             box_col = agent_col + action.agent_col_delta
    #             box_dest_row = box_row + action.box_row_delta
    #             box_dest_col = box_col + action.box_col_delta
                
    #             box_char = copy_boxes[box_row][box_col]
    #             copy_boxes[box_dest_row][box_dest_col] = box_char
    #             copy_boxes[box_row][box_col] = ""
                
    #             copy_agent_rows[agent] = box_row
    #             copy_agent_cols[agent] = box_col

    #         elif action.type is ActionType.Pull:
    #             # Usar posición ORIGINAL del agente
    #             agent_row = self.agent_rows[agent]
    #             agent_col = self.agent_cols[agent]
                
    #             # Caja en dirección OPUESTA al movimiento del agente
    #             box_row = agent_row - action.agent_row_delta  # ← CAMBIO
    #             box_col = agent_col - action.agent_col_delta  # ← CAMBIO
                
    #             # Destino de la caja: donde estaba el agente
    #             box_dest_row = agent_row
    #             box_dest_col = agent_col
                
    #             box_char = copy_boxes[box_row][box_col]
    #             copy_boxes[box_dest_row][box_dest_col] = box_char
    #             copy_boxes[box_row][box_col] = ""
                
    #             # Mover agente
    #             copy_agent_rows[agent] = agent_row + action.agent_row_delta
    #             copy_agent_cols[agent] = agent_col + action.agent_col_delta

    #     copy_state = State(copy_agent_rows, copy_agent_cols, copy_boxes)
    #     copy_state.parent = self
    #     copy_state.joint_action = joint_action.copy()
    #     copy_state.g = self.g + 1
    #     return copy_state


    def is_goal_state(self) -> bool:
        for row in range(len(State.goals)):
            for col in range(len(State.goals[row])):
                goal = State.goals[row][col]

                if "A" <= goal <= "Z" and self.boxes[row][col] != goal:
                    return False
                if "0" <= goal <= "9" and not (
                    self.agent_rows[ord(goal) - ord("0")] == row and self.agent_cols[ord(goal) - ord("0")] == col
                ):
                    return False
        return True

    def get_expanded_states(self) -> list["State"]:
        num_agents = len(self.agent_rows)

        # Determine list of applicable action for each individual agent.
        applicable_actions = [
            [action for action in Action if self.is_applicable(agent, action)] for agent in range(num_agents)
        ]

        # Iterate over joint actions, check conflict and generate child states.
        joint_action = [Action.NoOp for _ in range(num_agents)]
        actions_permutation = [0 for _ in range(num_agents)]
        expanded_states = []
        while True:
            for agent in range(num_agents):
                joint_action[agent] = applicable_actions[agent][actions_permutation[agent]]

            if not self.is_conflicting(joint_action):
                expanded_states.append(self.result(joint_action))

            # Advance permutation.
            done = False
            for agent in range(num_agents):
                if actions_permutation[agent] < len(applicable_actions[agent]) - 1:
                    actions_permutation[agent] += 1
                    break
                else:  # noqa: RET508
                    actions_permutation[agent] = 0
                    if agent == num_agents - 1:
                        done = True

            # Last permutation?
            if done:
                break

        State._RNG.shuffle(expanded_states)
        return expanded_states

    # def is_applicable(self, agent: int, action: Action) -> bool:
    #     agent_row = self.agent_rows[agent]
    #     agent_col = self.agent_cols[agent]
    #     _agent_color = State.agent_colors[agent]

    #     if action.type is ActionType.NoOp:
    #         return True

    #     if action.type is ActionType.Move:
    #         destination_row = agent_row + action.agent_row_delta
    #         destination_col = agent_col + action.agent_col_delta
    #         return self.is_free(destination_row, destination_col)

    #     assert False, f"Not implemented for action type {action.type}."

    def is_applicable(self, agent: int, action: Action) -> bool:
        agent_row = self.agent_rows[agent]
        agent_col = self.agent_cols[agent]

        if action.type is ActionType.NoOp:
            return True

        elif action.type is ActionType.Move:
            destination_row = agent_row + action.agent_row_delta
            destination_col = agent_col + action.agent_col_delta
            return self.is_free(destination_row, destination_col)

        elif action.type is ActionType.Push:
            # 1. Agent destination (where box is currently)
            box_row = agent_row + action.agent_row_delta
            box_col = agent_col + action.agent_col_delta
            # 2. Box destination (where box will land)
            box_dest_row = box_row + action.box_row_delta
            box_dest_col = box_col + action.box_col_delta
            
            # Check if there's a box at (box_row, box_col)
            box_char = self.boxes[box_row][box_col]
            if box_char == "":
                return False
            # Check if colors match
            if State.agent_colors[agent] != State.box_colors[ord(box_char) - ord('A')]:
                return False
            # Check if box destination is free
            return self.is_free(box_dest_row, box_dest_col)

        elif action.type is ActionType.Pull:
            # 1. Agent destination
            agent_dest_row = agent_row + action.agent_row_delta
            agent_dest_col = agent_col + action.agent_col_delta
            # 2. Box current position (relative to agent's STARTING position)
            box_row = agent_row - action.box_row_delta
            box_col = agent_col - action.box_col_delta
            
            # Check if agent's destination is free
            if not self.is_free(agent_dest_row, agent_dest_col):
                return False
            # Check if there's a box at (box_row, box_col)
            box_char = self.boxes[box_row][box_col]
            if box_char == "":
                return False
            # Check if colors match
            return State.agent_colors[agent] == State.box_colors[ord(box_char) - ord('A')]

        return False
    

    # def is_conflicting(self, joint_action: list[Action]) -> bool:
    #     num_agents = len(self.agent_rows)

    #     destination_rows = [-1 for _ in range(num_agents)]  # row of new cell to become occupied by action
    #     destination_cols = [-1 for _ in range(num_agents)]  # column of new cell to become occupied by action
    #     box_rows = [-1 for _ in range(num_agents)]  # current row of box moved by action
    #     box_cols = [-1 for _ in range(num_agents)]  # current column of box moved by action

    #     # Collect cells to be occupied and boxes to be moved.
    #     for agent in range(num_agents):
    #         action = joint_action[agent]
    #         agent_row = self.agent_rows[agent]
    #         agent_col = self.agent_cols[agent]

    #         if action.type is ActionType.NoOp:
    #             pass

    #         elif action.type is ActionType.Move:
    #             destination_rows[agent] = agent_row + action.agent_row_delta
    #             destination_cols[agent] = agent_col + action.agent_col_delta
    #             box_rows[agent] = agent_row  # Distinct dummy value.
    #             box_cols[agent] = agent_col  # Distinct dummy value.

    #     for a1 in range(num_agents):
    #         if joint_action[a1] is Action.NoOp:
    #             continue

    #         for a2 in range(a1 + 1, num_agents):
    #             if joint_action[a2] is Action.NoOp:
    #                 continue

    #             # Moving into same cell?
    #             if destination_rows[a1] == destination_rows[a2] and destination_cols[a1] == destination_cols[a2]:
    #                 return True

    #     return False

    def is_conflicting(self, joint_action: list[Action]) -> bool:
        num_agents = len(self.agent_rows)

        dest = set()
        box_dest = set()

        for agent in range(num_agents):
            action = joint_action[agent]
            agent_row = self.agent_rows[agent]
            agent_col = self.agent_cols[agent]

            if action.type is ActionType.NoOp:
                continue

            if action.type is ActionType.Move:
                r = agent_row + action.agent_row_delta
                c = agent_col + action.agent_col_delta
                if (r, c) in dest:
                    return True
                dest.add((r, c))

            elif action.type is ActionType.Push:
                box_row = agent_row + action.agent_row_delta
                box_col = agent_col + action.agent_col_delta
                box_dest_row = box_row + action.box_row_delta
                box_dest_col = box_col + action.box_col_delta

                if (box_dest_row, box_dest_col) in box_dest:
                    return True

                box_dest.add((box_dest_row, box_dest_col))
                dest.add((box_row, box_col))  # agent moves here

            elif action.type is ActionType.Pull:
                agent_dest_row = agent_row + action.agent_row_delta
                agent_dest_col = agent_col + action.agent_col_delta

                if (agent_dest_row, agent_dest_col) in dest:
                    return True

                dest.add((agent_dest_row, agent_dest_col))
                box_dest.add((agent_row, agent_col))  # box moves here

        return False

    def is_free(self, row: int, col: int) -> bool:
        return not State.walls[row][col] and not self.boxes[row][col] and self.agent_at(row, col) is None

    def agent_at(self, row: int, col: int) -> str | None:
        for agent in range(len(self.agent_rows)):
            if self.agent_rows[agent] == row and self.agent_cols[agent] == col:
                return chr(agent + ord("0"))
        return None

    def extract_plan(self) -> list[list[Action]]:
        plan = []
        state: State | None = self
        while state is not None and state.joint_action is not None:
            plan.append(state.joint_action)
            state = state.parent
        plan.reverse()
        return plan

    def __hash__(self) -> int:
        if self._hash is None:
            prime = 31
            h = 1
            h = h * prime + hash(tuple(self.agent_rows))
            h = h * prime + hash(tuple(self.agent_cols))
            h = h * prime + hash(tuple(State.agent_colors))
            h = h * prime + hash(tuple(tuple(row) for row in self.boxes))
            h = h * prime + hash(tuple(State.box_colors))
            h = h * prime + hash(tuple(tuple(row) for row in State.goals))
            h = h * prime + hash(tuple(tuple(row) for row in State.walls))
            self._hash = h
        return self._hash

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, State):
            return False
        if self.agent_rows != other.agent_rows:
            return False
        if self.agent_cols != other.agent_cols:
            return False
        if State.agent_colors != other.agent_colors:
            return False
        if State.walls != other.walls:
            return False
        if self.boxes != other.boxes:
            return False
        if State.box_colors != other.box_colors:
            return False
        return State.goals == other.goals

    def __repr__(self) -> str:
        lines = []
        for row in range(len(self.boxes)):
            line = []
            for col in range(len(self.boxes[row])):
                if self.boxes[row][col]:
                    line.append(self.boxes[row][col])
                elif State.walls[row][col] is not None:
                    line.append("+")
                elif (agent := self.agent_at(row, col)) is not None:
                    line.append(agent)
                else:
                    line.append(" ")
            lines.append("".join(line))
        return "\n".join(lines)
