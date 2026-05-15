"""
Task swap — implicit negotiation between agents.

When agent A's plan is blocked by a box that agent B can move, A reassigns
B's current task to "move the obstacle out of the way." This is the same
mechanism as cooperative auctions in multi-agent literature, materialised
here as a single typed event.

The Manager appends every swap to its `negotiations` log. The log is used
for two things:
  1. Cycle detection — if (requester, granter, obstacle_box) has been
     swapped repeatedly in a short window, the swap chain is in oscillation
     and further swaps are refused.
  2. Post-hoc analysis — the log can be replayed offline to visualise the
     decision chain (useful for the video presentation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from searchclient.Task import Task


@dataclass
class TaskSwap:
    requester:     int                # agent whose plan was blocked
    granter:       int                # agent re-tasked to clear the obstacle
    obstacle_box:  str                # char of the box being cleared
    obstacle_pos:  tuple[int, int]    # initial (row, col) of the obstacle
    original_task: "Task | None"      # what `granter` was doing before the swap
    new_task:      "Task"             # the obstacle-clearing task now assigned to `granter`
    t_initiated:   int                # manager timestep when the swap was recorded
    same_color:    bool               # True if requester == granter color group (intra-color);
                                      # False if cross-color negotiation
