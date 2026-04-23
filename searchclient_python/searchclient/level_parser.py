"""
CAPA 1 — Parsing e inicialización.

LevelParser.parse():
  - Lee el nivel del servidor (stdin).
  - Filtra cajas decorativas (cajas sin goal del mismo tipo → obstáculos estáticos).
  - Rellena State.{walls, goals, agent_colors, box_colors}.
  - Devuelve (initial_state, profile).

LevelProfile contiene métricas derivadas usadas por las capas superiores para
elegir qué planificador lanzar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TextIO

from searchclient.color import Color
from searchclient.state import State


@dataclass
class LevelProfile:
    """Métricas de nivel computadas una vez en el parsing."""

    num_rows: int = 0
    num_cols: int = 0
    num_agents: int = 0

    # Cajas que tienen goal correspondiente (planificables)
    real_boxes: list[tuple[int, int, str]] = field(default_factory=list)  # (row, col, char)
    # Cajas sin goal — tratadas como paredes dinámicas
    deco_boxes: list[tuple[int, int, str]] = field(default_factory=list)

    # Goals de caja: list[(row, col, char)]
    box_goals: list[tuple[int, int, str]] = field(default_factory=list)
    # Goals de agente: list[(row, col, agent_id)]
    agent_goals: list[tuple[int, int, int]] = field(default_factory=list)

    # Ratio cajas planificables / agentes
    boxes_per_agent: float = 0.0

    # Corredores: celdas libres con exactamente 2 vecinos libres (cardinalidad)
    corridor_cells: set[tuple[int, int]] = field(default_factory=set)

    # Grupos de color aislados: cada entrada es (color, frozenset de celdas)
    # Si dos grupos no comparten espacio → planificación independiente
    color_regions: dict[Color, frozenset[tuple[int, int]]] = field(default_factory=dict)

    # Flags de alto nivel para elegir planificador
    has_long_corridors: bool = False    # corredor crítico > 80 celdas
    is_warehouse: bool = False          # ratio cajas/agente > 20
    has_deco_trap: bool = False         # >10% cajas son decorativas


class LevelParser:

    @staticmethod
    def parse(server_messages: TextIO) -> tuple[State, LevelProfile]:
        """
        Lee el nivel desde server_messages y devuelve (initial_state, profile).

        Efecto secundario: rellena State.walls, State.goals,
        State.agent_colors, State.box_colors.
        """
        # --- domain ---
        server_messages.readline()  # #domain
        server_messages.readline()  # hospital

        # --- levelname ---
        server_messages.readline()  # #levelname
        server_messages.readline()  # <name>

        # --- colors ---
        server_messages.readline()  # #colors
        agent_colors: list[Color | None] = [None] * 10
        box_colors: list[Color | None] = [None] * 26
        line = server_messages.readline()
        while not line.startswith("#"):
            split = line.split(":")
            color = Color.from_string(split[0].strip())
            entities = [e.strip() for e in split[1].split(",")]
            for e in entities:
                if "0" <= e <= "9":
                    agent_colors[ord(e) - ord("0")] = color
                elif "A" <= e <= "Z":
                    box_colors[ord(e) - ord("A")] = color
            line = server_messages.readline()

        # --- initial state ---
        # line is currently "#initial"
        num_rows = 0
        num_cols = 0
        level_lines: list[str] = []
        line = server_messages.readline()
        while not line.startswith("#"):
            level_lines.append(line)
            num_cols = max(num_cols, len(line))
            num_rows += 1
            line = server_messages.readline()

        num_agents = 0
        agent_rows: list[int] = [-1] * 10
        agent_cols: list[int] = [-1] * 10
        walls = [[False] * num_cols for _ in range(num_rows)]
        boxes_initial = [[""] * num_cols for _ in range(num_rows)]

        for row, ln in enumerate(level_lines):
            for col, c in enumerate(ln):
                if "0" <= c <= "9":
                    agent_rows[ord(c) - ord("0")] = row
                    agent_cols[ord(c) - ord("0")] = col
                    num_agents += 1
                elif "A" <= c <= "Z":
                    boxes_initial[row][col] = c
                elif c == "+":
                    walls[row][col] = True

        del agent_rows[num_agents:]
        del agent_cols[num_agents:]

        # --- goal state ---
        # line is currently "#goal"
        goals_raw = [[""] * num_cols for _ in range(num_rows)]
        line = server_messages.readline()
        row = 0
        while not line.startswith("#"):
            for col, c in enumerate(line):
                if "0" <= c <= "9" or "A" <= c <= "Z":
                    goals_raw[row][col] = c
            row += 1
            line = server_messages.readline()
        # line is now "#end"

        # --- filter decorative boxes ---
        # A box type is "plannable" only if it appears in goals_raw as a box goal.
        goal_box_types: set[str] = set()
        for r in range(num_rows):
            for c in range(num_cols):
                g = goals_raw[r][c]
                if "A" <= g <= "Z":
                    goal_box_types.add(g)

        boxes = [[""] * num_cols for _ in range(num_rows)]
        deco_positions: list[tuple[int, int, str]] = []
        real_positions: list[tuple[int, int, str]] = []

        for r in range(num_rows):
            for c in range(num_cols):
                ch = boxes_initial[r][c]
                if not ch:
                    continue
                if ch in goal_box_types:
                    boxes[r][c] = ch
                    real_positions.append((r, c, ch))
                else:
                    # Decorative box: treat as static wall in the boxes grid
                    # (kept in boxes array so collision detection works)
                    boxes[r][c] = ch
                    deco_positions.append((r, c, ch))

        # --- populate State class attributes ---
        State.agent_colors = agent_colors
        State.box_colors = box_colors
        State.walls = walls
        State.goals = goals_raw

        initial_state = State(agent_rows, agent_cols, boxes)

        # --- build LevelProfile ---
        profile = LevelProfile(
            num_rows=num_rows,
            num_cols=num_cols,
            num_agents=num_agents,
            real_boxes=real_positions,
            deco_boxes=deco_positions,
        )

        # Collect box/agent goals
        for r in range(num_rows):
            for c in range(num_cols):
                g = goals_raw[r][c]
                if "A" <= g <= "Z":
                    profile.box_goals.append((r, c, g))
                elif "0" <= g <= "9":
                    profile.agent_goals.append((r, c, ord(g) - ord("0")))

        profile.boxes_per_agent = (
            len(real_positions) / num_agents if num_agents > 0 else 0.0
        )

        LevelParser._compute_corridors(profile, walls, num_rows, num_cols)
        LevelParser._compute_flags(profile)

        return initial_state, profile

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _free_neighbors(r: int, c: int, walls: list[list[bool]], num_rows: int, num_cols: int) -> int:
        count = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < num_rows and 0 <= nc < num_cols and not walls[nr][nc]:
                count += 1
        return count

    @staticmethod
    def _compute_corridors(profile: LevelProfile, walls: list[list[bool]], num_rows: int, num_cols: int) -> None:
        """Mark cells with exactly 2 free cardinal neighbors as corridor cells."""
        corridor: set[tuple[int, int]] = set()
        for r in range(num_rows):
            for c in range(num_cols):
                if not walls[r][c] and LevelParser._free_neighbors(r, c, walls, num_rows, num_cols) == 2:
                    corridor.add((r, c))
        profile.corridor_cells = corridor

        # Find longest connected corridor segment via BFS/DFS
        visited: set[tuple[int, int]] = set()
        max_len = 0
        for start in corridor:
            if start in visited:
                continue
            component: list[tuple[int, int]] = []
            stack = [start]
            while stack:
                cell = stack.pop()
                if cell in visited:
                    continue
                visited.add(cell)
                component.append(cell)
                r, c = cell
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb = (r + dr, c + dc)
                    if nb in corridor and nb not in visited:
                        stack.append(nb)
            max_len = max(max_len, len(component))

        profile.has_long_corridors = max_len > 80

    @staticmethod
    def _compute_flags(profile: LevelProfile) -> None:
        profile.is_warehouse = profile.boxes_per_agent > 20
        total_boxes = len(profile.real_boxes) + len(profile.deco_boxes)
        profile.has_deco_trap = (
            total_boxes > 0 and len(profile.deco_boxes) / total_boxes > 0.10
        )
