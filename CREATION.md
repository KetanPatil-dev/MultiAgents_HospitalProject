# MAS WardRush


## Introduction

In this file i will be initally describing what kind of problem needs to be solved and then you should try to implement it.
We will keep track here of what was implemented and what needs to be implemented now. I will use DESIGN.md for myself as a jouranal of my thoughts.

## TODO

This is what needs to be implemented.

I want you to do the following algorithm:
 
step 1 - preplanning:
for each agent colour create a list of tasks that need to be solved in the future by that agent group (i will call it an agent group or agent colour)
for each of specifc agents create a list of tasks that need to be solved in the future by that specifc agent
for each specifc agent create a list of tasks that are currentl  being solved by that agent 
for each specifc agent create a list of tasks that have been solved by that agent

as far as i know there are different types for tasks that need to be solved - moving boxes and positional placemnet, i want to unify them into one Task class that will have the following atributes i think ( i will be updating it on a way or if you have some ideas about it then do accordingly):
- task type (moving agent to a specific location - just agent, moving box to a specific location - with optional argument that would indicate where the agent needs to end up relative to the box - None or (x,y) for example if the agent needs to end up on the left of the box then it would be (-1,0) and if it needs to end up on the right of the box then it would be (1,0) and so on)
- object position (x,y)
- goal position (x,y)

next you need to find all final box placements and final agents positions.
for each final box placement add that task to the agent ggroup future list. for each specifc agent add its final position to the specifc agent future list as well.

now for each agent take a first task from coulour group future tasks list (just do it iteratively for now) and add it to the specific agent current task list. there should be 1 current task at the time (i might want to edit it later or in special edge cases but for now i want to keep it simple)

now perform hca* for each agent with the following heuristics:
- distance from the box to the goal
- distance from the agent to the box
- distance from the agent to the goal

if at any point any hca* will completely yeild no plan then know that you need to figure out why was it not able to find a plan.
for now we will consider the following problems:
- same coloured obstacle - if we are trying to put some box in its goal but we cannot find any possible path because there is another box of the same colour that is blocking the way, then we will call this problem the same coloured obstacle problem and we will need to solve it by switching the task of moving the original box to the goal with the task of moving the other box to the goal. we will do it by updating the current task from current tasks of that specifc agent to the task of moving the other box to that goal, then we will check if that obstacle box had its own task in the future tasks of that agent group and if it did then we will update it to the task of moving the original box to the goal.

this way each agent will have their tasks preplanned. now we can move to the main loop.

for each agent do the following:
- check if any of the current tasks of that agent is done and if it is then move it to the solved tasks list and remove it from the current tasks list
- if the agent has no current task then assign it a new task from the future tasks list and perform initial preplanning for that task
- since each agent has now the plan then perform BDI cycle 




## IMPLEMENTED

This is where the implemented features will be listed.

## PROJECT FUTURE

This is where future ideas will be implemented once the script matures enough.


## LEVEL SUBGOALS

This is trying to solve the levels in order to solve the whole problem. THe levels will increase with difficulty.

Levels:

- [] SingularWardRush.lvl
- [] EasyWardRush.lvl
- [] WardRush.lvl
