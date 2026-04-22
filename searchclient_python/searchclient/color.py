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
