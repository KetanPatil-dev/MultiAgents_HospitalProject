from dataclasses import dataclass


@dataclass
class Task:
    """
    Unified task representation.

    task_type: 'move_agent' (agent navigation) or 'move_box' (box pushing)
    object_pos: (row, col) of agent or box
    goal_pos: (row, col) target position
    agent_end_pos: optional (dr, dc) relative position for agent after task
                   e.g. (0, 1) = agent ends to the right of the box
    """

    task_type: str  # 'move_agent' or 'move_box'
    object_pos: tuple[int, int]
    goal_pos: tuple[int, int]
    agent_end_pos: tuple[int, int] | None = None
    box_char: str | None = None  # for move_box only
    crucial: bool = False
