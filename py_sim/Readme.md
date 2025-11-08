# BAR Py Sim
## Overview
This is a simple simulation in python that is used to do branching searches in the BAR makespace. 
Based on the Work presented in "Build Order Optimizatio in StarCraft" by David Churchill and Micahel Buro. This is a look ahead simulation that works based on the following functions:

 1. S' <- Sim(S, t): Return state S' based on simulating from State S for t timestep.s
 2. t <- When(S, R): Return t seconds from state S when requirements R will be met. E.g. I need 600 metal, at 30 metal/s and 450 metal currently, I will have 600 metal in 5 seconds. 
 3. S' <- Do(S, a): Issue action a in state S assuming all required resources are available. This subtracts used resources and flags resources as borrowed. (This description is based on the SC sim described in the paper, and will have slightly different semantics for BAR). 

 We define a state as S = <t, R, P, I> where:
   - t is the time in seconds
   - R is the resource vector - current metal and energy, as well as current metal and energy/s 
   - 

Not sure about the above, I think we can step faster if we build a bit differently:
There are 3 resources:
 - metal
 - energy
 - build power

### Resource

Everything has a cost in metal, energy, and build power. The metal or energy used in a second of building is:
min(current metal, metal cost/build power cost * applied build power)

#### Metal and Energy
Metal and energy are produced in a "stream" where player owned structures add x metal/s each second. Similarily, metal and energy are consumed 1 second at a time based on the equation above.

#### Build Power
Build power is owned by specific units, such as construction robots and construction turrents. Each has a build power value, whcih is the maximum build power it can provide per second. A contruction bot has 80 bp/s.

#### Build time Example
Assume a construction bot wants to build a wind turbine. The cost of the turbine is:
43 Metal
175 Energy
1680 Build Power

The construction bot has 80 Build Power.
Build power applied = 80
Metal cost/s = 43*(80/1680) = ~2.04 m/s
Energy Cost/s = 175*(80/1680) = ~8.3 e/s
Build Time = 1680/80 = 21

If at any timestep, we do not have 2.04 to spend, then the build power applied will be reduced. If instead we only had 1.5 m, the build power consumed would be 60. In the case that both metal and energy were limited, we would spend build power equal to the min(bp*(m/s actual)/(m/s target), bp * (e/s actual)/(e/s target))

Build power can be stacked, so two constuction bots can work together to build the wind turbine, at which point the build power is just additive, so max build power applied to the turbine is 160, max metal/s is 4.08, max energy/s is 16.6. 

Build power has a location and a range. Construction bots are at a location, and can move to another location. They can build things a small distance away. Construction turrents have greater range and more build power, but can't move. So the position of build power matters a lot. And the placement of buildings matters, to ensure maximum build power can be applied. 

## Simulation Goals
Given a set of build actions underway (this includes building and units, since the cost formula and resources are identical), each second calculate the new state. State is then:
S = (t, R, P, I) where:
 - t is the current time
 - R is the resource vector, specifically units and building that have build power, and their location.s
 - Build actions in progress, the total cost, location, and percentage complete.
 - Income and current resource holding for metal and energy. Very simply, total metal, metal/s, total energy, energy/s

### Simulation step
For any given step, do the following:
 1. Calculate total resource expenditure this time step.
 2. Calculate total income this time step.
 3. Update position of any units that move this time step. (Only construction bots and the commander will move.)
 4. Update the build percentage of each p in P this timestep.
 5. Update current resources as total = total + income - expenditure. 