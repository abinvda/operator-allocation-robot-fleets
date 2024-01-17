using POMDPs
using POMDPModels
using POMDPModelTools
using POMDPSimulators
using BeliefUpdaters
using QMDP
using MCTS
using Parameters
using DiscreteValueIteration
using Random
using StaticArrays
using StatsBase
using SparseArrays
using Colors
using Plots
using FIB
using Printf
using Distributions # For Normal , norm functions
using LinearAlgebra
using Combinatorics
using BasicPOMCP
using Profile
using Traceur
using TickTock
using BenchmarkTools
using Alert
using DelimitedFiles # For saving variables and results

include("SingleRobot_Teleoperation_MDP.jl")
include("MultiRobot_Teleoperation_MDP.jl")
include("WhittleIndex_TeleopMDP.jl")
include("Wrapped_Teleoperation_MDP.jl")
include("Custom_Policies_Teleoperation_MDP.jl")
# include("utilsPOMDP.jl")

mdp = TeleopMDP()

struct Transition_Dist
    [Array{MultiRobot_State}, Array{Int64}]
end
## Select Solver
solver = SparseValueIterationSolver(max_iterations = 100000, belres = 1e-15, verbose = false)

run_time_matrix = []
offline_run_time_matrix = zeros(10, 10, 4) # (n_robots, n_operators, n_policies)
for n_opr in [1,2]
    for n_robots in [1,2,3,4,5]
        # Initilize some system parameters
        mx = 7
        discount_factor = 0.99
        lambda = -0.75
        map_size = [mx, 2]
        rho = - 2.0
        rho_e = - 4.0

        ## Initialize k robots and obtain individual optimal policies
        temp_time = zeros(5, 1)
        policy_eval_time = 0.0
        passive_policy_eval_time = 0.0
        WI_time = 0.0
        AG_time = 0.0
        max_itr = 1
        for itr = 1:max_itr
            robots = TeleopMDP[]
            policies = ValueIterationPolicy[]
            passive_policies = Vector{ValueIterationPolicy}(undef, n_robots)
            all_probabilities = Any[]
            allowed_transition_types = [2, 3]
            for k = 1:n_robots
                probabilities = get_transition_probabilities(map_size[1], allowed_transition_types)
                robot = TeleopMDP(
                    size_x = map_size[1],
                    r_operate = lambda,
                    r_not_goal = rho,
                    r_error_state = rho_e,
                    p0_ii = probabilities[1],
                    p0_ie = probabilities[2],
                    p1_ii = probabilities[3],
                    p1_ie = probabilities[4],
                    p1_ei = probabilities[5],
                    p1_ee = probabilities[6],
                    discount_factor = discount_factor,
                )
                policy_eval_time += @belapsed solve($solver, $robot)
                policy = solve(solver, robot)
                push!(robots, robot)
                push!(policies, policy)

                passive_robot = TeleopMDP(
                    size_x = map_size[1],
                    r_operate = lambda,
                    r_not_goal = rho,
                    r_error_state = rho_e,
                    p0_ii = probabilities[1],
                    p0_ie = probabilities[2],
                    p1_ii = probabilities[3],
                    p1_ie = probabilities[4],
                    p1_ei = probabilities[5],
                    p1_ee = probabilities[6],
                    discount_factor = discount_factor,
                    is_passive = 1,
                )
                passive_policy_eval_time += @belapsed solve($solver, $passive_robot)
                policy = solve(solver, passive_robot)
                passive_policies[k] = policy
            end

            ## Index Policy
            whittleindex_matrix = getWhittleIndices_Offline(robots, solver, -150.0, 10.0) # Whitle indices using binary search
            AG_matrix = -adaptive_greedy(robots) # Whittle indices using Adaptive Geedy algorithm
            current_states = fill(MDP_State(1, 1), n_robots) # Initialize the starting state
            
            mr_act1 = get_action_Index_policy_individual(policies, robots, current_states, whittleindex_matrix, n_opr)
            mr_act2 = get_action_Index_policy_individual(policies, robots, current_states, AG_matrix, n_opr)

            WI_time += @belapsed whittleindex_matrix = getWhittleIndices_Offline($robots, $solver, -150.0, 10.0)
            whittleindex_matrix = getWhittleIndices_Offline(robots, solver, -150.0, 10.0)
            current_states = fill(MDP_State(1, 1), n_robots)
            AG_time += @belapsed adaptive_greedy($robots)
            temp_time[1] += @belapsed mr_act = get_action_Index_policy_individual(
                $policies,
                $robots,
                $current_states,
                $whittleindex_matrix,
                $n_opr,
            )
            temp_time[2] += @belapsed mr_act = get_action_L_step_myopic($current_states, $robots, $n_opr, 1, $passive_policies)
            mr_act = @belapsed get_action_Greedy6_policy_individual($current_states, $robots, $n_opr, $passive_policies, 1)
            temp_time[3] += @belapsed mr_act = get_action_L_step_myopic($current_states, $robots, $n_opr, 2, $passive_policies)
            temp_time[4] += @belapsed mr_act = get_action_D_lambda_policy_individual($policies, $current_states, $n_opr)
            temp_time[5] += @belapsed mr_act = get_action_reactive_policy($current_states, $n_opr)
        end
        avg_time = (temp_time / max_itr)
        push!(run_time_matrix, [n_opr, n_robots, avg_time, WI_time/max_itr, AG_time/max_itr, passive_policy_eval_time/max_itr, policy_eval_time/max_itr])
        @show n_opr, n_robots, avg_time
        @show WI_time/max_itr, AG_time/max_itr, passive_policy_eval_time/max_itr, policy_eval_time/max_itr
        offline_run_time_matrix[n_robots, n_opr, :] = [WI_time/max_itr, AG_time/max_itr, passive_policy_eval_time/max_itr, policy_eval_time/max_itr]

        println(" ")
    end
end

@show run_time_matrix
@show offline_run_time_matrix