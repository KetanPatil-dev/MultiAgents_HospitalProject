from abc import ABC, abstractmethod

from searchclient.state import State


class Heuristic(ABC):
    """Clase abstracta base para todas las heurísticas"""
    
    def __init__(self, initial_state: State) -> None:
        # Aquí las subclases pueden hacer preprocessing si lo necesitan
        pass

    @abstractmethod
    def h(self, state: State) -> int:
        """Función heurística h(n) - debe ser implementada por subclases"""
        ...

    @abstractmethod
    def f(self, state: State) -> int:
        """Función de evaluación f(n) - debe ser implementada por subclases"""
        ...

    @abstractmethod
    def __repr__(self) -> str:
        """Representación en string de la heurística"""
        ...

class HeuristicGoalCount(Heuristic):
    def __init__(self, initial_state: State) -> None:
        super().__init__(initial_state)
    
    def h(self, state: State) -> int:
        count = 0
        
        # Iterate for all the cells
        for row in range(len(State.goals)):
            for col in range(len(State.goals[row])):
                goal = State.goals[row][col]
                
                if "A" <= goal <= "Z":
                    if state.boxes[row][col] != goal:
                        count += 1  
        
        return count

class HeuristicAdvanced(Heuristic):
    def __init__(self, initial_state: State) -> None:
        super().__init__(initial_state)

        self.distances = {}
        relevant_positions = self._get_relevant_positions(initial_state)

        for pos in relevant_positions:
            self.distances[pos] = self._bfs_from(pos[0], pos[1])
    
    def _get_relevant_positions(self, initial_state: State):
        positions = set() 
        
        for row in range(len(State.goals)):
            for col in range(len(State.goals[row])):
                goal = State.goals[row][col]  
                
                if "A" <= goal <= "Z":
                    positions.add((row, col)) 
        
        for row in range(len(initial_state.boxes)):
            for col in range(len(initial_state.boxes[row])):
                if initial_state.boxes[row][col]:
                    positions.add((row, col)) 
        
        agent_row = initial_state.agent_rows[0]  
        agent_col = initial_state.agent_cols[0]  
        positions.add((agent_row, agent_col))
        
        return positions  
    
    def _bfs_from(self, start_row: int, start_col: int):
        distances = {} 
        
        queue = [(start_row, start_col, 0)]
        
        visited = {(start_row, start_col)}
        
        while queue:
            row, col, dist = queue.pop(0)
            
            distances[(row, col)] = dist

            for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                new_row = row + dr  
                new_col = col + dc  

                if (new_row, new_col) not in visited:
                    if 0 <= new_row < len(State.walls) and 0 <= new_col < len(State.walls[0]):
                        if not State.walls[new_row][new_col]:
                            visited.add((new_row, new_col))
                            queue.append((new_row, new_col, dist + 1))

        return distances
  
    def h(self, state: State) -> int:
        total = 0  

        agent_row = state.agent_rows[0]
        agent_col = state.agent_cols[0]

        min_agent_dist = float('inf')

        BOX_WEIGHT = 3   # eso fuerte para cajas
        AGENT_WEIGHT = 1 # peso suave para agente

        for row in range(len(State.goals)):
            for col in range(len(State.goals[row])):
                goal_char = State.goals[row][col]

                if "A" <= goal_char <= "Z":
                    box_row, box_col = self._find_box(state, goal_char)

                    if box_row is not None:

                        # Distancia caja → objetivo
                        if (box_row, box_col) in self.distances and \
                        (row, col) in self.distances[(box_row, box_col)]:
                            box_dist = self.distances[(box_row, box_col)][(row, col)]
                        else:
                            box_dist = abs(box_row - row) + abs(box_col - col)

                        # Peso fuerte
                        total += BOX_WEIGHT * box_dist

                        # istancia agente → caja (solo si no está ya en goal)
                        if (box_row, box_col) != (row, col):
                            if (agent_row, agent_col) in self.distances and \
                            (box_row, box_col) in self.distances[(agent_row, agent_col)]:
                                agent_dist = self.distances[(agent_row, agent_col)][(box_row, box_col)]
                            else:
                                agent_dist = abs(agent_row - box_row) + abs(agent_col - box_col)

                            min_agent_dist = min(min_agent_dist, agent_dist)

        if min_agent_dist < float('inf'):
            total += AGENT_WEIGHT * min_agent_dist

        return total


    # #### USUAL, NO FALLA ####
    # def h(self, state: State) -> int:
    #     total = 0  

    #     agent_row = state.agent_rows[0]
    #     agent_col = state.agent_cols[0]

    #     min_agent_dist = float('inf')

    #     for row in range(len(State.goals)):
    #         for col in range(len(State.goals[row])):
    #             goal_char = State.goals[row][col]  

    #             if "A" <= goal_char <= "Z":
    #                 box_row, box_col = self._find_box(state, goal_char)

    #                 if box_row is not None and (box_row, box_col) != (row, col):
                        
    #                     # Dist box-objective

    #                     if (box_row, box_col) in self.distances:
    #                         if (row, col) in self.distances[(box_row, box_col)]:
    #                             box_dist = self.distances[(box_row, box_col)][(row, col)]
    #                         else:
    #                             box_dist = abs(box_row - row) + abs(box_col - col)
    #                     else:
    #                         box_dist = abs(box_row - row) + abs(box_col - col)

    #                     total += box_dist
                        
    #                     # Dist agent-box

    #                     if (agent_row, agent_col) in self.distances:
    #                         if (box_row, box_col) in self.distances[(agent_row, agent_col)]:
    #                             agent_dist = self.distances[(agent_row, agent_col)][(box_row, box_col)]
    #                         else:
    #                             agent_dist = abs(agent_row - box_row) + abs(agent_col - box_col)
    #                     else:
    #                         agent_dist = abs(agent_row - box_row) + abs(agent_col - box_col)

    #                     min_agent_dist = min(min_agent_dist, agent_dist)
        
    #     if min_agent_dist < float('inf'):
    #         total += min_agent_dist
        
    #     return total  
    
    #### POCHA ####
    # def h(self, state: State) -> int:
    #     agent_row = state.agent_rows[0]
    #     agent_col = state.agent_cols[0]
        
    #     box_distances = []  # Lista de distancias
    #     min_agent_dist = float('inf')
        
    #     for row in range(len(State.goals)):
    #         for col in range(len(State.goals[row])):
    #             goal_char = State.goals[row][col]
                
    #             if "A" <= goal_char <= "Z":
    #                 box_row, box_col = self._find_box(state, goal_char)
                    
    #                 if box_row is not None and (box_row, box_col) != (row, col):
                        
    #                     if (box_row, box_col) in self.distances and (row, col) in self.distances[(box_row, box_col)]:
    #                         box_dist = self.distances[(box_row, box_col)][(row, col)]
    #                     else:
    #                         box_dist = abs(box_row - row) + abs(box_col - col)
                        
    #                     box_distances.append(box_dist)
                        
    #                     if (agent_row, agent_col) in self.distances and (box_row, box_col) in self.distances[(agent_row, agent_col)]:
    #                         agent_dist = self.distances[(agent_row, agent_col)][(box_row, box_col)]
    #                     else:
    #                         agent_dist = abs(agent_row - box_row) + abs(agent_col - box_col)
                        
    #                     min_agent_dist = min(min_agent_dist, agent_dist)
        
    #     # *** HYBRID HEURISTICS ***
    #     if box_distances:
    #         # Ordenar distancias de mayor a menor
    #         box_distances.sort(reverse=True)
            
    #         # Peso completo a la más lejana, peso reducido a las demás
    #         total = box_distances[0]  
            
    #         for dist in box_distances[1:]:
    #             total += dist * 0.3 
    #     else:
    #         total = 0
        
    #     if min_agent_dist < float('inf'):
    #         total += min_agent_dist
        
    #     return int(total)
    
    def _find_box(self, state: State, box_char: str):
        for row in range(len(state.boxes)):
            for col in range(len(state.boxes[row])):
                if state.boxes[row][col] == box_char:
                    return (row, col)  

        return (None, None)        


class HeuristicAStar(HeuristicAdvanced):
    def __init__(self, initial_state: State) -> None:
        super().__init__(initial_state)

    def f(self, state: State) -> int:
        return state.g + self.h(state)

    def __repr__(self) -> str:
        return "A* evaluation"


class HeuristicWeightedAStar(HeuristicAdvanced):
    def __init__(self, initial_state: State, w: int) -> None:
        super().__init__(initial_state)
        self.w = w

    def f(self, state: State) -> int:
        return state.g + self.w * self.h(state)

    def __repr__(self) -> str:
        return f"WA*({self.w}) evaluation"


class HeuristicGreedy(HeuristicAdvanced):
    def __init__(self, initial_state: State) -> None:
        super().__init__(initial_state)

    def f(self, state: State) -> int:
        return self.h(state)

    def __repr__(self) -> str:
        return "greedy evaluation"


