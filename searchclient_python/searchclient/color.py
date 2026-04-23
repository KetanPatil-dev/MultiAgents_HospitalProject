from enum import Enum, unique


@unique
class Color(Enum):
    Blue = 0
    Red = 1
    Cyan = 2
    Purple = 3
    Green = 4
    Orange = 5
    Pink = 6
    Grey = 7
    Lightblue = 8
    Brown = 9

    @staticmethod
    def from_string(s: str) -> "Color | None":
        return _STR_TO_COLOR.get(s.lower())

    @staticmethod
    def compatible(agent_color: "Color | None", box_color: "Color | None") -> bool:
        """An agent can manipulate a box iff they share the same color."""
        return agent_color is not None and agent_color == box_color


_STR_TO_COLOR = {
    "blue": Color.Blue,
    "red": Color.Red,
    "cyan": Color.Cyan,
    "purple": Color.Purple,
    "green": Color.Green,
    "orange": Color.Orange,
    "pink": Color.Pink,
    "grey": Color.Grey,
    "lightblue": Color.Lightblue,
    "brown": Color.Brown,
}
