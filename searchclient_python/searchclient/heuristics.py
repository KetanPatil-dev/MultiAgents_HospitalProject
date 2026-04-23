"""
CAPA 1 — Heurísticas y distancias precomputadas.

DistanceMap:
  - BFS desde cada celda goal → tabla dist[goal_cell][row][col].
  - Consulta O(1): dist_map.dist(goal_row, goal_col, from_row, from_col).
  - Se construye una vez al inicio sobre el mapa de paredes.
"""

from __future__ import annotations

from collections import deque

from searchclient.state import State


class DistanceMap:
    """
    Precomputes BFS distances from every goal cell to every reachable cell.

    Usage:
        dist_map = DistanceMap(goals, walls)
        d = dist_map.dist(goal_r, goal_c, from_r, from_c)  # O(1)
    """

    INF = 10 ** 9

    def __init__(
        self,
        goals: list[tuple[int, int]],  # list of (row, col) goal positions to precompute from
        walls: list[list[bool]],
    ) -> None:
        self._walls = walls
        self._num_rows = len(walls)
        self._num_cols = len(walls[0]) if walls else 0
        # _table[(gr, gc)] = 2-D list of distances from every cell to (gr, gc)
        self._table: dict[tuple[int, int], list[list[int]]] = {}
        for gr, gc in goals:
            self._table[(gr, gc)] = self._bfs(gr, gc)

    def dist(self, goal_r: int, goal_c: int, from_r: int, from_c: int) -> int:
        """Return BFS distance from (from_r, from_c) to (goal_r, goal_c). INF if unreachable."""
        table = self._table.get((goal_r, goal_c))
        if table is None:
            return self.INF
        return table[from_r][from_c]

    def has_goal(self, goal_r: int, goal_c: int) -> bool:
        return (goal_r, goal_c) in self._table

    # ------------------------------------------------------------------

    def _bfs(self, start_r: int, start_c: int) -> list[list[int]]:
        dist = [[self.INF] * self._num_cols for _ in range(self._num_rows)]
        if self._walls[start_r][start_c]:
            return dist
        dist[start_r][start_c] = 0
        queue: deque[tuple[int, int]] = deque([(start_r, start_c)])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < self._num_rows
                        and 0 <= nc < self._num_cols
                        and not self._walls[nr][nc]
                        and dist[nr][nc] == self.INF):
                    dist[nr][nc] = dist[r][c] + 1
                    queue.append((nr, nc))
        return dist

    @classmethod
    def from_state(cls, state: State) -> "DistanceMap":
        """Convenience constructor: precompute from all box goals in State.goals."""
        goals: list[tuple[int, int]] = []
        for r, row in enumerate(State.goals):
            for c, g in enumerate(row):
                if "A" <= g <= "Z":
                    goals.append((r, c))
        return cls(goals, State.walls)
