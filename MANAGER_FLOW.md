# Manager Flow Reference

## Overview
Manager coordinates multi-agent pathfinding in two phases:
1. **SETUP** (once): Plan generation, conflict resolution, fallback attempts
2. **MAIN LOOP** (every turn): BDI cycle, real-time conflict resolution, deadlock recovery

---

## SETUP PHASE

### Step 1: Task Assignment (Hungarian Algorithm)
```
Input:
  - Agents (with colors)
  - Boxes (with positions & colors)
  - Goals (with positions & characters)

Algorithm:
  1. Match boxes ↔ goals by character (greedy min-cost)
  2. Assign pairs to agents:
     - Only if agent color matches box color
     - Load-balanced: agent with fewest tasks gets it
  3. Order multiple tasks per agent by proximity

Output:
  Agent 0: [(box_r, box_c, goal_r, goal_c, char), ...]
  Agent 1: [(box_r, box_c, goal_r, goal_c, char), ...]
  ...
```

---

### Step 2: HCA* PASS 1 - Priority Planning

**Key Idea**: Plan in **priority order** (0 → N-1). Each higher-priority agent constrains lower ones.

```
╔════════════════════════════════════════════════════════════╗
║ FOR EACH AGENT (0 → N-1):                                 ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  1. Try to plan ALL tasks (sequential subgoals)           ║
║                                                            ║
║  2. If PLAN FOUND:                                        ║
║     ├─ Store plan in agent._plan                          ║
║     └─ Register path as (r, c, t) constraints             ║
║        for all LOWER-priority agents                      ║
║        (they can't occupy these cells at these times)     ║
║                                                            ║
║  3. If PLAN FAILED:                                       ║
║     ├─ Print "Agent N: cooperative pre-plan FAILED"       ║
║     └─ Move to next agent (no constraints added)          ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

**Example (3 agents, all have boxes)**:

```
INITIAL STATE:
  Agent 0 at (1,1), Agent 1 at (3,3), Agent 2 at (8,8)
  Box A at (2,2), Box B at (5,5), Box C at (9,9)

Planning (Priority: 0 > 1 > 2):

  Agent 0:
    Plan: (1,1) → (2,2) [push] → (0,1) [goal A]
    Constraints added for Agents 1,2:
      t=0: (1,1) blocked, (2,2) blocked
      t=1: (2,2) blocked, (0,1) blocked
      ...
  
  Agent 1:
    Constraints from Agent 0:
      Must avoid (1,1) at t=0, (2,2) at t=0-1, (0,1) at t=1...
    Plan: (3,3) → wait → (5,5) [can't pass Agent 0 early]
    Success! Constraints added for Agent 2.
  
  Agent 2:
    Constraints from Agents 0,1:
      Must avoid both paths
    Plan: (8,8) → (9,9) [push] → ...
    Success!
```

**What causes "cooperative pre-plan FAILED"?**
- A* can't find path around constraints
- Higher-priority agent blocks the only route to lower agent's box
- Problem: **level geometry incompatible with priority order**

---

### Step 3: HCA* PASS 2 - Projected State Retry

**For agents that failed Pass 1**: Try again with modified state.

```
╔════════════════════════════════════════════════════════════╗
║ FOR EACH AGENT WITHOUT PLAN FROM PASS 1:                  ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  1. BUILD PROJECTED STATE:                                ║
║     ├─ Copy initial boxes                                 ║
║     ├─ REMOVE other agents' boxes (set to "")             ║
║     └─ Keep walls, agent positions                        ║
║                                                            ║
║  2. BUILD EXTRA CONSTRAINTS:                              ║
║     ├─ Time-indexed box trajectories                      ║
║     │  (when boxes move, block cells temporarily)         ║
║     └─ Permanent goal blocks                              ║
║        (goals stay blocked after agents' plans end)       ║
║                                                            ║
║  3. REPLAN with all constraints:                          ║
║     └─ If successful: store plan                          ║
║        If failed: print "retry pre-plan FAILED (pass 2)"  ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

**Why does Pass 2 work sometimes?**
- Pass 1: Agent 1 blocked by Agent 0's BOXES on grid
- Pass 2: Remove Agent 0's boxes, add TIME-INDEXED constraints
  - Allows A* to route through cells Agent 0 will vacate in time
  - Example: Cell (5,5) occupied by Agent 0's box t=3-7, but Agent 1 can wait and use it at t=8

---

### Step 4: Try Alternative Priority Orders (If still failures)

```
IF agents still lack plans after Pass 1 + 2:

Try these orderings (keep first that reduces failures):
  ├─ Rotations:     [0,1,2] → [1,2,0] → [2,0,1]
  ├─ Reverse:       [2,1,0]
  └─ Failed first:  [agent_1, agent_0, agent_2, ...]

FOR EACH ORDERING:
  Reset all plans & constraints
  Run HCA* Passes 1 + 2
  Count failures
  
If fewer failures than previous best:
  Keep this order
  Try next
Else:
  Revert to best so far
```

**Output**: `Priority order [1, 0, 2]: 1 failure(s) (was 2).` → improved!

---

### Step 5: CBS Conflict Resolution

**Purpose**: Fix trajectory conflicts in finalized plans.

```
WHILE conflicts exist AND haven't exceeded node limit:
  
  1. DETECT CONFLICT:
     ├─ Vertex: two agents at (r, c) at time t
     └─ Swap: agents swap positions between t-1 and t
  
  2. RESOLVE:
     ├─ Create constraint: (r, c, t) forbidden for one agent
     └─ Replan that agent with constraint
  
  3. CHECK: no conflicts? DONE. Else, repeat.

LIMIT: ~300-2000 iterations (don't spend forever)
If exceed: keep HCA* plans as-is
```

---

### Step 6: Joint A* Fallback

**When**: If ≤2 agents AND (small level OR HCA* incomplete)

```
SEARCH SPACE: (All agent positions, all box positions)

FOR EACH STATE:
  ├─ All agents move simultaneously
  ├─ Track heuristic: sum(box → goal distances)
  └─ Expand if f = g + h < best_f_so_far

STOP WHEN:
  ├─ Goal reached: ALL boxes at goals ✓
  ├─ State limit hit: 1M states (2-agent), 500K (3-agent), etc.
  └─ Timeout

OUTPUT:
  ├─ If success: store in manager._joint_plan
  │  (main loop will serve pre-computed actions)
  └─ If failed: print "Joint A* failed — using HCA* plans"
```

**Why only ≤2 agents?**
- 3+ agent joint space explodes: O(b^n_agents)
- Practically: timeouts on any non-trivial level

---

## MAIN LOOP (Every Timestep)

### Overview
```
FOR EACH TIMESTEP:
  
  IF joint_plan exists:
    Pop next joint action → return (FAST PATH)
  
  ELSE:
    ├─ BDI loop (per agent):
    │  ├─ Sense (pop completed tasks)
    │  ├─ Plan sound? (plan still valid?)
    │  ├─ Replan if needed (A* or IW)
    │  └─ Next action (pop from plan)
    │
    ├─ Conflict resolution (score-based greedy)
    │  └─ Override conflicting agents to NoOp
    │
    ├─ Deadlock detection (cycle check)
    │  └─ Force NoOps / escape goals if stuck
    │
    └─ Return resolved joint action
```

---

### Step 1: BDI Cycle (Per Agent)

```
╔═══════════════════════════════════════════════════════════╗
║ FOR EACH AGENT (0 → N-1):                                ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  1. SENSE:                                               ║
║     If first task's box is at goal:                       ║
║       ├─ Pop task from task list                         ║
║       ├─ Clear plan                                       ║
║       └─ Repeat until uncompleted task or empty          ║
║                                                           ║
║  2. PLAN SOUND?                                          ║
║     ├─ If no tasks AND no goal: True (done)              ║
║     ├─ If have tasks/goal AND plan exists                ║
║     │  AND _plan_index < len(plan): True                 ║
║     └─ Else: False (need replan)                         ║
║                                                           ║
║  3. REPLAN (if not sound):                               ║
║     Tries (in order):                                     ║
║     ├─ IW(1): Fast novelty-based search                  ║
║     ├─ Ghost A*: Ignore other agents                     ║
║     ├─ Obstacle clear: Push blocking box then plan       ║
║     └─ If all fail: replan failed (print warning)        ║
║                                                           ║
║  4. NEXT ACTION:                                         ║
║     └─ Pop agent._plan[agent._plan_index++]              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

---

### Step 2: Action Proposal

```
FOR EACH AGENT:
  IF agent in forced_noop[agent_id]:
    proposed_action = NoOp
    (deadlock recovery — wait turn)
  
  ELIF agent._escape_stagger > 0:
    proposed_action = NoOp
    (post-escape cooldown)
  
  ELSE:
    proposed_action = agent.next_action()
```

---

### Step 3: Conflict Resolution (Score-based Priority)

```
╔═══════════════════════════════════════════════════════════╗
║ SCORE = work_remaining * 100 + progress_to_goal          ║
║ Done agents (no tasks, no goal): score = -1              ║
║                                                           ║
║ SORT agents by score DESC (highest → lowest)             ║
║                                                           ║
║ FOR EACH AGENT (descending score):                       ║
║   IF proposed_action causes conflict:                    ║
║     ├─ Check if blocker should yield instead              ║
║     │  (if blocker is done or nav-only vs active)        ║
║     ├─ If blocker yields:                                ║
║     │  └─ Blocker does NoOp, add to conflicted list      ║
║     └─ Else:                                             ║
║        └─ This agent does NoOp, add to conflicted list   ║
║   ELSE:                                                   ║
║     └─ Commit action to resolved list                    ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

**Result**: `resolved = [Action, Action, NoOp, Action, ...]`

---

### Step 4: Deadlock Detection & Recovery

```
╔═══════════════════════════════════════════════════════════╗
║ WINDOW = last 12 timesteps                               ║
║ KEY = (agent_positions_tuple, box_positions_tuple)       ║
║                                                           ║
║ IF key seen in window:                                   ║
║   → CYCLE DETECTED                                       ║
║   → Stuck agents need intervention                       ║
║                                                           ║
║ FOR EACH STUCK AGENT:                                    ║
║   deadlock_count[agent_id] += 1                          ║
║   count = deadlock_count[agent_id]                       ║
║                                                           ║
║   IF count <= 15:                                        ║
║     ├─ wait_turns = min(2 + (count-1)*2, 10)             ║
║     ├─ Add position constraints (force different route)  ║
║     └─ Force NoOp for wait_turns                         ║
║   ELSE (count > 15, severely stuck):                     ║
║     ├─ Find escape cell (BFS, ≥6 steps away)             ║
║     ├─ Assign as temporary agent_goal                    ║
║     ├─ Clear plan & constraints                          ║
║     └─ Print "ESCAPE to (r, c)"                          ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

**Messages**:
- `"t=X Deadlock #1: agent Y waits 2 turns."` → force NoOp
- `"t=X Deadlock #7: agent Y ESCAPE to (15, 8)."` → break deadlock

---

### Step 5: NoOp Streak Tracking

```
FOR EACH AGENT:
  IF proposed_action is NoOp AND has_work:
    noop_streak[agent_id] += 1
  ELSE:
    noop_streak[agent_id] = 0

IF noop_streak >= 8 AND has_box_tasks AND no escape goal yet:
  ├─ Find escape cell (BFS)
  ├─ Set as agent_goal
  ├─ Clear plan & constraints
  └─ Print "Agent X: escaping after 8-turn streak"
```

**Purpose**: Proactively move stuck agents out of congested areas.

---

## Error Reference

| Message | Phase | Cause | Fix |
|---------|-------|-------|-----|
| `"Agent N: cooperative pre-plan FAILED (pass 1)."` | Setup | Higher-priority agents block path | Try Pass 2, alternative orders |
| `"Agent N: retry pre-plan FAILED (pass 2)."` | Setup | Even with boxes removed, no path | Change level topology or priorities |
| `"Priority order [X,Y,Z]: 1 failure(s)"` | Setup | Recovery attempt improving | Continuing search |
| `"Joint A* failed"` | Setup | Small level, but no solution found | Use HCA* + runtime replanning |
| `"t=X Agent N: replan failed"` | Main loop | Runtime obstacle invalidated plan | Expect deadlock recovery to help |
| `"t=X Deadlock #M: agent Y waits T turns"` | Main loop | Cycle detected | Waiting, adding constraints |
| `"t=X Agent Y: escaping after 8-turn streak"` | Main loop | Severe congestion | Moving to different area |

---

## Level Design Tips

**For HCA* to work:**
- Agents shouldn't block each other's box access paths
- Box goals should be reachable from agent starting positions
- Avoid narrow corridors with multiple boxes

**If pre-planning fails:**
- Check: Is each agent's box reachable?
- Check: Is each box goal reachable from agent?
- Check: Can agents execute sequentially (priority order)?

**If runtime deadlocks**:
- Level has tight space where many agents compete
- Manager will try escape cells and forced waits
- If too severe: might timeout (need level redesign)
