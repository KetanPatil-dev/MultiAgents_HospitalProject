# New Manager Design Document

**Author**: Damian  
**Date**: [Fill in]  
**Status**: Planning / In Progress / Complete

---

## 1. Overall Philosophy

**Goal**: [Describe the high-level approach. What is fundamentally different from the current design?]

**Key Principles**:
- it should use a combination of pre-planning but also detect obstacles and replan at runtime
- it should handle targets reassignemnet,
- each agent should have one task at a time and should not be assigned a new task until it finishes the current one
- it should be able to detect deadlocks and allow some agents to take over the priority and tell the other agents what to do
- there should be a list of tasks that each agent type needs to do and a list of tasks that each agent should do (some takss should be shared between aganets of the same type - boxes - but there should be some tasks of moving specifc aganet and box to the goal that should be assigned to specific agents)
- once the agent is assigned a box -> path planning should consider the box and agent as one entity and not allow for any more pushing or pulling of other objects, however if another box of the same colour if found on the way then the agent should switch between the two boxes and push pull the other box insread of the original, the task should be switched and the other task should be removed from the list of tasks for that agent and perhaps removed from the list of tasks for that agent type as well if there is no more box goals for that colour (if there are 3 red boxes and only 2 red goals then the first goal of moving that box somewhere should be overriden by another one)

**Assumptions**:
- 
- 

---

## 2. Setup Phase

### 2.1 Task Assignment

**Input**:
- at the begiging i would like the preplanning to happen in  the following state, i want the hungarina alogrithm to find boxes to solve but only consider the boxes that are within the reach of no obstacles (ignore the aganets but do not ignore boxes as obstacles)
- make a list of all goals and assigne each agent to one goal but keep a list of goals for each aganet type and additioanl goals for a speicifc aganets, give them all a priority of 1 so that in the futture we can manually assign higher priority to those specicigc agents and their goals

**Algorithm**:

```
1. [Step 1]
2. [Step 2]
3. [Step 3]
...
```

**Output**:
- 

**Edge Cases**:
- 

---

### 2.2 Initial Planning

**Strategy**: [Describe your approach to pre-planning. HCA*? Joint A*? Something else?]

**Algorithm**:

```
1. [Step 1]
2. [Step 2]
...
```

try to do hca* with the following heuristics:
- distance from the box to the goal
- distance from the agent to the box
- distance from the agent to the goal

if at any point any hca* will completely yeild no plan then know that you need to solve the obstacle problem

**When to fallback**: [What triggers moving to next strategy?]

**Fallback strategies** (in order):
1. [Strategy 1]
2. [Strategy 2]
3. [Strategy 3]

---

### 2.3 Conflict Detection & Resolution (Setup)

**Purpose**: [What are you resolving here?]
if there is conflict then it can be different cases

1 there is 

**Algorithm**:

```
1. [Step 1]
2. [Step 2]
...
```

---

## 3. Main Loop (Per Timestep)

### 3.1 Agent Sensing

**Purpose**: [What do agents learn about world state each turn?]

**Per Agent**:

```
1. [Check 1]
2. [Check 2]
...
```

---

### 3.2 Plan Validity Check

**Purpose**: [When do agents need to replan?]

**Conditions to replan**:
- 
- 
- 

---

### 3.3 Replanning

**When triggered**: [From plan validity check]

**Planners tried** (in order):
1. **[Planner 1]**
   - Speed: [fast/medium/slow]
   - Success rate: [what types of problems it solves]
   - Fallback when fails: →  [Planner 2]

2. **[Planner 2]**
   - Speed: 
   - Success rate: 
   - Fallback when fails: → [Planner 3]

3. **[Planner 3]**
   - Speed: 
   - Success rate: 
   - If all fail: [What then?]

---

### 3.4 Action Proposal

**Per Agent**:

```
IF [condition 1]:
  action = [action type]
ELIF [condition 2]:
  action = [action type]
ELSE:
  action = [pop from plan]
```

---

### 3.5 Conflict Resolution (Runtime)

**Purpose**: [How do you handle simultaneous actions that conflict?]

**Priority scoring**:

```
score(agent_id) = [formula]
  - Higher score = commits first
  - Lower score = yields first
```

**Conflict detection**:
- [Conflict type 1]
- [Conflict type 2]
- [Conflict type 3]

**Resolution strategy**:

```
1. [Step 1]
2. [Step 2]
...
```

**Exception handling**:
- If [situation]: [what happens]
- If [situation]: [what happens]

---

### 3.6 Deadlock Detection

**Window size**: [N timesteps]

**Detection method**:

```
Track: [What state do you track?]

If [condition]:
  → Deadlock detected
```

**Recovery**:

```
FOR EACH stuck agent:
  1. [Step 1]
  2. [Step 2]
  ...

Escape strategy: [How do agents escape?]
```

---

## 4. Data Structures

### Required

```python
class Manager:
    # [Describe each field]
    # Example:
    # agents: list[Agent]  — all agents
    # timestep: int        — current timestep
    
```

### Per-Agent

```python
class Agent:
    # [Describe relevant fields]
    # tasks: list[tuple]   — remaining box tasks
    # _plan: list[Action]  — current plan
    
```

### Constraints/Reservations

```
Type: [data structure]
Usage: [when is it written/read?]
Example: [concrete example]
```

---

## 5. Edge Cases & Special Handling

| Scenario | Current Behavior | Your Approach |
|----------|------------------|---|
| Agent has no tasks assigned | | |
| Box already at goal | | |
| Dead-end reachable only by one agent | | |
| Multiple agents conflicting at same cell | | |
| Agent stuck in cycle | | |
| Level unsolvable from start | | |

---

## 6. Error Handling

### Setup Phase Failures

| Failure Type | Current Message | Your Handling |
|---|---|---|
| Agent can't plan task | `"cooperative pre-plan FAILED"` | |
| All agents fail | | |
| Resource limit hit | | |

### Runtime Failures

| Failure Type | Current Message | Your Handling |
|---|---|---|
| Replan returns None | `"replan failed"` | |
| Deadlock detected | `"Deadlock #N"` | |
| Timeout reached | | |

---

## 7. Key Algorithms / Components

### Algorithm 1: [Name]

**Purpose**: [What does it do?]

**Input**: [parameters]

**Output**: [result]

**Complexity**: [Time/space]

**Implementation notes**:
- 
- 

---

### Algorithm 2: [Name]

**Purpose**: 

**Input**: 

**Output**: 

**Complexity**: 

**Implementation notes**:

---

## 8. Testing Strategy

**Levels to test** (in order of complexity):
1. [Simple level]
2. [Medium level]
3. [Complex level]

**Metrics to track**:
- Setup time
- Main loop iterations until solved
- Memory usage
- Deadlock count
- Replanning frequency

**Expected behavior**:
- Level 1: Should complete in under [X] steps
- Level 2: Should handle without deadlock
- Level 3: Should recover from deadlock via [mechanism]

---

## 9. Known Limitations / Trade-offs

| Issue | Trade-off | Rationale |
|---|---|---|
| | | |

---

## 10. Implementation Checklist

- [ ] Task assignment logic
- [ ] Setup planning (Pass 1)
- [ ] Setup fallback (Pass 2)
- [ ] Alternative orderings
- [ ] Conflict resolution (setup)
- [ ] Main loop skeleton
- [ ] BDI cycle per agent
- [ ] Runtime conflict resolution
- [ ] Deadlock detection
- [ ] Escape mechanism
- [ ] Testing on simple levels
- [ ] Testing on complex levels
- [ ] Performance optimization
- [ ] Memory guards
- [ ] Error logging

---

## 11. Notes / Ideas for Future

[Free-form space for design thoughts, potential optimizations, alternative approaches to revisit]

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | [Date] | [Initial design] |
