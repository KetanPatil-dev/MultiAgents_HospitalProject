import sys
import time

from searchclient import memory
from searchclient.action import Action
from searchclient.frontier import Frontier
from searchclient.state import State

start_time = time.perf_counter()


def search(initial_state: State, frontier: Frontier) -> list[list[Action]] | None:
    output_fixed_solution = False

    if output_fixed_solution:
        # Part 1:
        # The agents will perform the sequence of actions returned by this method.
        # Try to solve a few levels by hand, enter the found solutions below, and run them:

        return [
            [Action.MoveS],
            [Action.MoveE],
            [Action.MoveE],
            [Action.MoveS],
            [Action.MoveE],
            [Action.MoveE],
            [Action.MoveE],
            [Action.MoveE],
            [Action.MoveE],
            [Action.MoveE],
            [Action.MoveE],
            [Action.MoveE],
            [Action.MoveS],
            [Action.MoveS],
        ]

    # Part 2:
    # Now try to implement the Graph-Search algorithm from R&N figure 3.7
    # In the case of "failure to find a solution" you should return None.
    # Some useful methods on the state class which you will need to use are:
    # state.is_goal_state() - Returns true if the state is a goal state.
    # state.extract_plan() - Returns the list of actions used to reach this state.
    # state.get_expanded_states() - Returns a list containing the states reachable from the current state.
    # You should also take a look at frontier.py to see which methods the Frontier interface exposes
    #
    # print_search_status(expanded, frontier): As you can see below, the code will print out status
    # (#expanded states, size of the frontier, #generated states, total time used) for every 1000th node
    # generated.
    # You should also make sure to print out these stats when a solution has been found, so you can keep
    # track of the exact total number of states generated!!

    iterations = 0

    ### Set initial Frontier
    frontier.add(initial_state)     # Start in the Ini state
    explored: set[State] = set()    

    ### Looking for Goal WHILE still More to Explore
    while not frontier.is_empty():
        iterations += 1
        if iterations % 50000 == 0:
            print_search_status(explored, frontier)

        if memory.get_usage() > memory.max_usage:  # type: ignore
            print_search_status(explored, frontier)
            print("Maximum memory usage exceeded.", file=sys.stderr, flush=True)
            return None

        ## Select last state from the queue as next state
        current_state = frontier.pop()
        explored.add(current_state)

        if current_state.is_goal_state():
            print_search_status(explored, frontier)
            return current_state.extract_plan()

        ## Expand from current state
        for next_state in current_state.get_expanded_states():
            if not frontier.contains(next_state) and next_state not in explored:
                frontier.add(next_state)    # Add the newly discovered states

    print_search_status(explored, frontier)
    return None


def print_search_status(explored: set[State], frontier: Frontier) -> None:
    elapsed_time = time.perf_counter() - start_time
    print(
        f"#Expanded: {len(explored):8,}, #Frontier: {frontier.size():8,}, "
        f"#Generated: {len(explored) + frontier.size():8,}, Time: {elapsed_time:3.3f} s\n"
        f"[Alloc: {memory.get_usage():4.2f} MB, MaxAlloc: {memory.max_usage:4.2f} MB]",
        file=sys.stderr,
        flush=True,
    )
