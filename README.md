# Scalable Operator Allocation and Scheduling Algorithm for Semi-Autonomous Robot Fleets

This repository contains the minimal code necessary to reproduce the results presented in the following publication:

[A. Dahiya, N. Akbarzadeh, A. Mahajan, and S. L. Smith, "Scalable Operator Allocation for Multirobot Assistance: A Restless Bandit Approach](https://ieeexplore.ieee.org/abstract/document/9721692)," published in IEEE Transactions on Control of Network Systems, Volume 9, Issue 3, pp. 1397-1408, September 2022. DOI: 10.1109/TCNS.2022.3153872.

# Abstract
In this paper, we consider the problem of allocating human operators in a system with multiple semiautonomous robots. Each robot is required to perform an independent sequence of tasks, subjected to a chance of failing and getting stuck in a fault state at every task. If and when required, a human operator can assist or teleoperate a robot. Conventional MDP techniques used to solve such problems face scalability issues due to exponential growth of state and action spaces with the number of robots and operators. In this paper we derive conditions under which the operator allocation problem is indexable, enabling the use of the Whittle index heuristic. The conditions can be easily checked to verify indexability, and we show that they hold for a wide range of problems of interest. Our key insight is to leverage the structure of the value function of individual robots, resulting in conditions that can be verified separately for each state of each robot. We apply these conditions to two types of transitions commonly seen in remote robot supervision systems. Through numerical simulations, we demonstrate the efficacy of Whittle index policy as a near-optimal and scalable approach that outperforms existing scalable methods.

# File Structure

- Run `main_Multi-Robot-Teleoperation_Fast.jl` to generate test results shown in the paper.

- `SingleRobot_Teleoperation_MDP.jl` and `MultiRobot_Teleoperation_MDP.jl` create problem instances for single and multi-robot cases respectively.

- `Custom_Policies_Teleoperation_MDP.jl` contains implementation of various baseline policies, including Benefit-maximization policy, L-step greedy policy and Reactive allocation policy.

- `WhittleIndex_TeleopMDP.jl` contains the implementation of the Whittle Index policy.

Please refer to the publication for detailed insights into the algorithm and its application in the context of scalable operator allocation and scheduling for semi-autonomous robot fleets.

## Requirements

To run the code, ensure you have Julia version 1.7.0 installed. Additionally, install the required Julia packages listed below:

```plaintext
Alert==1.2.0
BasicPOMCP==0.3.6
BeliefUpdaters==0.2.2
BenchmarkTools==1.2.0
Colors==0.12.8
Combinatorics==1.0.2
DelimitedFiles
DiscreteValueIteration==0.4.5
Distributions==0.24.18
FIB==0.4.3
LinearAlgebra
MCTS==0.4.7
POMDPModelTools==0.3.9
POMDPModels==0.4.14
POMDPSimulators==0.3.13
POMDPs==0.9.3
Parameters==0.12.3
Plots==1.25.0
Printf
Profile
QMDP==0.1.6
Random
SparseArrays
StaticArrays==1.2.13
StatsBase==0.33.13
TickTock==1.1.0
Traceur==0.3.1
```
