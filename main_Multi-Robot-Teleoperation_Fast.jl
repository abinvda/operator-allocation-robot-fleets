## This is a faster version of multi-robot teleoperation simulations. Here, we don't formulate the big multi-robot problem initially. Instead, we use the single robot definitions to get our work done.
# This also means that under this method, we cannot simulate the optimal multi-robot policy.
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

mdp = TeleopMDP() # Initialize a single-robot teleoperation MDP problem instance with default variables

struct Transition_Dist
    [Array{MultiRobot_State}, Array{Int64}]
end
## Select Solver
solver = SparseValueIterationSolver(max_iterations = 100000, belres = 1e-15, verbose = false)

## Function for checking terminal condition
function all_goals_reached(robots, current_states)
    all_done = true
    for k=1:size(robots,1)
        all_done = all_done && (current_states[k].x == robots[k].size_x || current_states[k].x == -1) # TODO: Implement this as inline function
    end
    return all_done
end

## Outer loop # Number of problem instances to simulate
outer_itr = 100 

max_robots = 60
for n_opr in [1,2,3,5] # Total number of operators
    tick()
    # Initialize variables for storing results for each condition (number of robots) under each policy
    all_averageCosts = zeros(max_robots, 6)
    all_StdDevs = zeros(max_robots, 6)
    all_averageCosts_normalized = zeros(max_robots, 6)
    all_StdDevs_normalized = zeros(max_robots, 6)
    all_averageCosts_count = zeros(max_robots * outer_itr, 6)
    all_StdDevs_count = zeros(max_robots * outer_itr, 6)
    for n_robots in n_opr*[1,3,5,10]
        if n_opr > n_robots && n_opr != 1
            continue
        end
        # Initialize variables for storing results for each iteration for each policy (which will be averaged later)
        total_reward_Benefit = zeros(outer_itr, 1)
        total_reward_WI = zeros(outer_itr, 1)
        total_reward_Greedy = zeros(outer_itr, 1)
        total_reward_MR_Greedy = zeros(outer_itr, 1)
        total_reward_Myopic_1 = zeros(outer_itr, 1)
        total_reward_Myopic_2 = zeros(outer_itr, 1)
        total_reward_Optimal = zeros(outer_itr, 1)
        total_reward_Reactive = zeros(outer_itr, 1)
        println("At $n_robots robots and $n_opr operators.")

        # Initilize some system parameters
        discount_factor = 0.99
        n_inner_itr = 500 # Number of iterations for each problem instance
        lambda = -0.75
        max_allowed_time_sec = 10
        
        for itr = 1:outer_itr
            @show itr
            # alert("Iteration $itr with $n_robots robots.")
            map_size = [7, 2] # Define number of waypoints. 7 waypoints, 2 internal states each
            rho = -2.0
            rho_e = -4.0

            ## Initialize k robots and obtain individual optimal policies
            robots = Vector{TeleopMDP}(undef, n_robots) # A vector of single-robot MDPs
            policies = Vector{ValueIterationPolicy}(undef, n_robots) # A vector of policies for those single-robot problems
            passive_policies = Vector{ValueIterationPolicy}(undef, n_robots)
            all_probabilities = Any[]
            for k = 1:n_robots # Initialize transition probabilities for all waypoints
                allowed_transition_types = [2, 3] # These type-2 and 3 correspond to type-1 and 2 from the paper, respectively.
                push!(all_probabilities, get_transition_probabilities(map_size[1], allowed_transition_types))
            end

            for k = 1:n_robots
                probabilities = all_probabilities[k]
                robot = TeleopMDP(size_x = map_size[1], r_operate = lambda, r_not_goal = rho, r_error_state = rho_e, p0_ii = probabilities[1], p0_ie = probabilities[2], p1_ii = probabilities[3], p1_ie = probabilities[4], p1_ei = probabilities[5], p1_ee = probabilities[6], discount_factor = discount_factor)
                policy = solve(solver, robot)
                robots[k] = robot
                policies[k] = policy

                passive_robot = TeleopMDP(size_x = map_size[1], r_operate = lambda, r_not_goal = rho, r_error_state = rho_e, p0_ii = probabilities[1], p0_ie = probabilities[2], p1_ii = probabilities[3], p1_ie = probabilities[4], p1_ei = probabilities[5], p1_ee = probabilities[6], discount_factor = discount_factor, is_passive = 1)
                policy = solve(solver, passive_robot)
                passive_policies[k] = policy
            end

            ## Simulate with Index policy
            temp_reward_WI = zeros(n_inner_itr, 1)
            whittleindex_matrix = getWhittleIndices_Offline(robots, solver, -150.0, 2.0) # This is the binary search method of finding indices. Will replace it with the Adaptice Greedy implementation in the future.
            time_exceeded = false
            for inner_itr = 1:n_inner_itr
                robots_rewards = 0.0
                current_states = fill(MDP_State(1,1), n_robots)
                t0 = time_ns()
                while !all_goals_reached(robots, current_states)
                    mr_act = get_action_Index_policy_individual(policies, robots, current_states, whittleindex_matrix, n_opr)
                    robots_actions = [in(x,mr_act) ? :active : :passive for x in 1:n_robots]
                    ## Simulate for 1 step with selected actions and obtain next states
                    for k=1:n_robots
                        sim(robots[k], max_steps=2, initialstate=current_states[k]) do s
                            current_states[k] = s
                            a = robots_actions[k]
                            return a
                        end
                        robots_rewards += reward(robots[k], current_states[k], robots_actions[k])
                    end
            
                    if ((time_ns() - t0) % Int64)*10^-9 > max_allowed_time_sec
                        println("Time exceeded by Index Policy.")
                        @show ((time_ns() - t0) % Int64)*10^-9
                        time_exceeded = true
                        break
                    end
                end
                if time_exceeded
                    break
                end
                temp_reward_WI[inner_itr] = robots_rewards/n_robots
            end
            total_reward_WI[itr] = mean(temp_reward_WI)
            whittleindex_matrix = Nothing # To clear it from memory
            temp_reward_WI = Nothing

            ## Simulate with Benefit Max. policy 
            temp_reward_Benefit = zeros(n_inner_itr, 1)
            time_exceeded = false
            for inner_itr = 1:n_inner_itr
                robots_rewards = 0.0
                current_states = fill(MDP_State(1,1), n_robots)
                t0 = time_ns()
                while !all_goals_reached(robots, current_states)
                    mr_act = get_action_Benefit_policy_individual(policies, current_states, n_opr)
                    robots_actions = [in(x,mr_act) ? :active : :passive for x in 1:n_robots]
                    ## Simulate for 1 step with selected actions and obtain next states
                    for k=1:n_robots
                        sim(robots[k], max_steps=2, initialstate=current_states[k]) do s
                            current_states[k] = s
                            a = robots_actions[k]
                            return a
                        end
                        robots_rewards += reward(robots[k], current_states[k], robots_actions[k])
                    end
                    if ((time_ns() - t0) % Int64)*10^-9 > max_allowed_time_sec
                        println("Time exceeded by Benefit Max. Policy.")
                        @show ((time_ns() - t0) % Int64)*10^-9
                        time_exceeded = true
                        break
                    end
                end
                if time_exceeded
                    break
                end
                temp_reward_Benefit[inner_itr] = robots_rewards/n_robots
            end
            total_reward_Benefit[itr] = mean(temp_reward_Benefit)
            temp_reward_Benefit = Nothing

            ## Simulate with Multi-Robot Greedy Policy
            temp_reward_MR_Greedy = zeros(n_inner_itr, 1)
            for inner_itr = 1:n_inner_itr
                robots_rewards = 0.0
                current_states = fill(MDP_State(1,1), n_robots)
                while !all_goals_reached(robots, current_states)
                    mr_act = get_action_Greedy5_policy_individual(current_states, robots, n_opr)
                    robots_actions = [in(x, mr_act) ? :active : :passive for x in 1:n_robots]
                    ## Simulate for 1 step with selected actions and obtain next states
                    for k=1:n_robots
                        sim(robots[k], max_steps=2, initialstate=current_states[k]) do s
                            current_states[k] = s
                            a = robots_actions[k]
                            return a
                        end
                        robots_rewards += reward(robots[k], current_states[k], robots_actions[k])
                    end
                end
                temp_reward_MR_Greedy[inner_itr] = robots_rewards/n_robots
            end
            total_reward_MR_Greedy[itr] = mean(temp_reward_MR_Greedy)
            temp_reward_MR_Greedy = Nothing

            ## Simulate with 1-Step Myopic Policy
            temp_reward_Myopic_1 = zeros(n_inner_itr, 1)
            time_exceeded = false
            for inner_itr = 1:n_inner_itr
                robots_rewards = 0.0
                current_states = fill(MDP_State(1, 1), n_robots)
                t0 = time_ns()
                while !all_goals_reached(robots, current_states)
                    mr_act = get_action_Greedy6_policy_individual(current_states, robots, n_opr, passive_policies, 1)
                    robots_actions = [in(x, mr_act) ? :active : :passive for x = 1:n_robots]
                    ## Simulate for 1 step with selected actions and obtain next states
                    for k = 1:n_robots
                        sim(robots[k], max_steps = 2, initialstate = current_states[k]) do s
                            current_states[k] = s
                            a = robots_actions[k]
                            return a
                        end
                        robots_rewards += reward(robots[k], current_states[k], robots_actions[k])
                    end

                    if ((time_ns() - t0) % Int64) * 10^-9 > max_allowed_time_sec
                        println("Time exceeded by 1 Step Greedy Policy.")
                        time_exceeded = true
                        break
                    end
                end
                temp_reward_Myopic_1[inner_itr] = robots_rewards / n_robots
                if time_exceeded
                    break
                end
            end
            total_reward_Myopic_1[itr] = mean(temp_reward_Myopic_1)
            temp_reward_Myopic_1 = Nothing

            ## Simulate with 2-Step Myopic Policy
            temp_reward_Myopic_2 = zeros(n_inner_itr, 1)
            time_exceeded = false
            for inner_itr = 1:n_inner_itr
                robots_rewards = 0.0
                current_states = fill(MDP_State(1,1), n_robots)
                t0 = time_ns()
                while !all_goals_reached(robots, current_states)
                    mr_act = get_action_L_step_myopic(current_states, robots, n_opr, 2, passive_policies)
                    robots_actions = [in(x, mr_act) ? :active : :passive for x in 1:n_robots]
                    ## Simulate for 1 step with selected actions and obtain next states
                    for k=1:n_robots
                        sim(robots[k], max_steps=2, initialstate=current_states[k]) do s
                            current_states[k] = s
                            a = robots_actions[k]
                            return a
                        end
                        robots_rewards += reward(robots[k], current_states[k], robots_actions[k])
                    end
                    if ((time_ns() - t0) % Int64)*10^-9 > max_allowed_time_sec
                        println("Time exceeded by 2 Step Greedy Policy.")
                        time_exceeded = true
                        break
                    end
                end
                temp_reward_Myopic_2[inner_itr] = robots_rewards/n_robots
                if time_exceeded
                    break
                end
            end
            total_reward_Myopic_2[itr] = mean(temp_reward_Myopic_2)
            temp_reward_Myopic_2 = Nothing

            ## Simulate with Reactive Policy
            temp_reward_Reactive = zeros(n_inner_itr, 1)
            for inner_itr = 1:n_inner_itr
                robots_rewards = 0.0
                current_states = fill(MDP_State(1, 1), n_robots)
                while !all_goals_reached(robots, current_states)
                    mr_act = get_action_reactive_policy(current_states, n_opr)
                    robots_actions = [in(x, mr_act) ? :active : :passive for x = 1:n_robots]
                    ## Simulate for 1 step with selected actions and obtain next states
                    for k = 1:n_robots
                        sim(robots[k], max_steps = 2, initialstate = current_states[k]) do s
                            current_states[k] = s
                            a = robots_actions[k]
                            return a
                        end
                        robots_rewards += reward(robots[k], current_states[k], robots_actions[k])
                    end
                end
                temp_reward_Reactive[inner_itr] = robots_rewards / n_robots
            end
            total_reward_Reactive[itr] = mean(temp_reward_Reactive)
            temp_reward_Reactive = Nothing

        end

        Base.GC.gc() # Runs garbage collector
        all_averageCosts[n_robots, :] = [-mean(total_reward_Reactive), -mean(total_reward_WI), -mean(total_reward_Benefit), -mean(total_reward_MR_Greedy), -mean(total_reward_Myopic_1), -mean(total_reward_Myopic_2)]
        all_StdDevs[n_robots, :] = [std(total_reward_Reactive), std(total_reward_WI), std(total_reward_Benefit), std(total_reward_MR_Greedy), std(total_reward_Myopic_1), std(total_reward_Myopic_2)]


        writedlm("all_averageCosts_n_opr=$n_opr.csv", all_averageCosts, ',') # Even if the simulations are not complete, we save the results after each test condition.
        writedlm("all_StdDev_n_opr=$n_opr.csv", all_StdDevs, ',')
    end

    @show all_averageCosts # Each row shows costs like [Reactive policy, Index policy, Benefit Max. policy, Greedy policy, 1-step Myopic policy, 2-step Myopic policy].
    @show all_StdDevs

    x = 1:max_robots
    p = plot(x, all_averageCosts, yerror = all_StdDevs, title = "Average")
    display(p)

    writedlm("all_averageCosts_n_opr=$n_opr.csv", all_averageCosts, ',')
    writedlm("all_StdDev_n_opr=$n_opr.csv", all_StdDevs, ',')

    tock()
end
