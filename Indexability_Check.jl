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
using BasicPOMCP
using Profile
using Traceur
using TickTock
using BenchmarkTools
using Alert
using DelimitedFiles

include("SingleRobot_Teleoperation_MDP.jl")
include("MultiRobot_Teleoperation_MDP.jl")
include("WhittleIndex_TeleopMDP.jl")
include("Wrapped_Teleoperation_MDP.jl")
include("Custom_Policies_Teleoperation_MDP.jl")

mdp = TeleopMDP()

struct Transition_Dist
    [Array{MultiRobot_State}, Array{Int64}]
end
## Select Solver
solver = SparseValueIterationSolver(max_iterations = 10000, belres = 1e-15, verbose = false)

## Checking terminal condition
function all_goals_reached(robots, current_states)
    all_done = true
    for k=1:size(robots,1)
        all_done = all_done && (current_states[k].x == robots[k].size_x || current_states[k].x == -1) # TODO: Implement this as inline function
    end
    return all_done
end

## Outer loop
tick()
# Initilize some system parameters
map_size = [7,2]
rho = -2.0 
rho_e = -4.0
allowed_transition_types = [2,3]

all_discounts = [0.90, 0.95, 0.99]
iters_failed = zeros(length(all_discounts))
condition_satisfied_count = zeros(length(all_discounts))
all_failed_probabilities = []


total_iters = 100
for iter = 1:total_iters
for (d_ind,discount_factor) in enumerate(all_discounts)
    probabilities = get_transition_probabilities(map_size[1], allowed_transition_types, :AllRandom, :No, discount_factor)
    # Initialize change matrix
        condition_satisfied_count[d_ind] += check_conditions(probabilities, discount_factor)
        change_matrix = zeros(map_size[1],3) # First column is the change counter, second column is the previous optimal action, third column is the lambda at which latest change occured.

        lambda_min = -1000.0
        lambda_max = 1000.0
        for lambda = lambda_min:5.0:lambda_max
            robot = TeleopMDP(size_x=map_size[1], r_operate=lambda, r_not_goal=rho, r_error_state=rho_e, p0_ii = probabilities[1], p0_ie = probabilities[2], p1_ii = probabilities[3], p1_ie = probabilities[4], p1_ei = probabilities[5], p1_ee = probabilities[6], discount_factor = discount_factor)
            policy = solve(solver, robot)

            instance_failed = 0
            # check optimal action at each individual state and update change_matrix
            for i = 1:map_size[1]-1

                if (action(policy, MDP_State(i,1)) == :active ? 1 : 0) != change_matrix[i,2]
                    change_matrix[i,1] += 1
                    change_matrix[i,2] = action(policy, MDP_State(i,1)) == :active ? 1 : 0
                    change_matrix[i,3] = lambda
                end
                if change_matrix[i,1] > 1
                    instance_failed = 1
                    break
                end
            end
            if instance_failed == 136
                break
            end
        end
        if any(x -> x>1, change_matrix[:,1])
            iters_failed[d_ind] += 1
            push!(all_failed_probabilities, probabilities)
        end
end
Base.GC.gc()
end
println("Check complete.")
iters_passed = total_iters .- iters_failed
println("$iters_passed of $total_iters samples were found to be indexabile for gamma = $all_discounts.")
println("Sufficient conditions were satisfied for $condition_satisfied_count samples.")
tock()
