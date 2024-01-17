# include("utilsPOMDP.jl") It has the function install_all_packages() that installs all required packages. Only required once after a fresh julia install.

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
solver = SparseValueIterationSolver(max_iterations = 1000000, belres = 1e-18, verbose = false)

## Checking terminal condition
function all_goals_reached(robots, robots_states)
    all_done = true
    for k=1:size(robots,1)
        all_done = all_done && (robots_states.robots[k].x == robots[k].size_x || robots_states.robots[k].x == -1) # TODO: Implement this as inline function
    end
    return all_done
end

## Outer loop # Number of problem instances to simulate
outer_itr = 100
pool_size = 100

max_robots = 5
for n_opr = [1]
    tick()
    # Initialize variables for storing results for each condition (number of robots) under each policy
    all_averageCosts = zeros(max_robots, 4)
    all_StdDevs = zeros(max_robots, 4)
    all_averageCosts_normalized = zeros(max_robots, 4)
    all_StdDevs_normalized = zeros(max_robots, 4)
    all_averageCosts_count = zeros(max_robots*outer_itr, 4)
    all_StdDevs_count = zeros(max_robots*outer_itr, 4)

for n_robots = [1,2,3]
    # Initialize variables for storing results for each iteration for each policy (which will be averaged later)
    total_reward_D_lambda = Float64[]
    total_reward_WI = Float64[]
    total_reward_Greedy = Float64[]
    total_reward_MR_Greedy = Float64[]
    total_reward_Optimal = Float64[]
    total_reward_Active = Float64[]
    percentile_costs = Float64[]
    println("At $n_opr operators and $n_robots robots.")

    # Initilize some system parameters
    map_size = [6,2]
    if n_robots >= 4
        map_size = [5,2]
    end
    discount_factor = 0.99
    n_inner_itr = 1000000 # Number of iterations for each problem instance
    lambda = -0.75
    # Obtain all possible states and transitions (in the format of state indices)
    temp_robot = TeleopMDP(size_x=map_size[1])
    ss = states(temp_robot)
    perm = Iterators.product(fill(ss,n_robots)...) # This gives us all permutations of states possible, given the individual state space is identical. If they are different just use product(ss1,ss2,ss3) etc.
    all_states = vec(map(x -> MultiRobot_State(collect(x)), perm))
    perm = Nothing
    Base.GC.gc()
    all_actions = unique(collect(combinations([fill(0, n_opr); 1:n_robots],n_opr)))
    next_states_dict = get_next_states_dict_cheap(all_states, map_size[1], temp_robot, length(ss), n_robots)
    uneq_count = 0
    
    for itr = 1:outer_itr
        @show itr
        ## Define parameters
        rho = -2.0
        rho_e = -4.0

        ## Initialize k robots and obtain individual optimal policies
        robots = Vector{TeleopMDP}(undef, n_robots) # A vector of single-robot MDPs
        policies = Vector{ValueIterationPolicy}(undef, n_robots) # A vector of policies for those single-robot problems
        all_probabilities = Any[]
        for k = 1:n_robots # Initialize transition probabilities for all waypoints
            allowed_transition_types = [2, 3] # These type-2 and 3 correspond to type-1 and 2 from the paper, respectively.
            push!(all_probabilities, get_transition_probabilities(map_size[1], allowed_transition_types))
        end

        for k=1:n_robots
            probabilities = sample(all_probabilities)
            robot = TeleopMDP(size_x=map_size[1], r_operate=lambda, r_not_goal=rho, r_error_state=rho_e, p0_ii = probabilities[1], p0_ie = probabilities[2], p1_ii = probabilities[3], p1_ie = probabilities[4], p1_ei = probabilities[5], p1_ee = probabilities[6], discount_factor = discount_factor)
            policy = solve(solver2, robot)
            robots[k] = robot
            policies[k] = policy
        end

        ## Now create a dictionary for transition probabilities for each state
        state_transition_table = Matrix{Array{Int64}}(undef, length(all_states),length(all_actions))
        state_transition_probs = Matrix{Array{Float64}}(undef, length(all_states),length(all_actions))
        for (s_ind,mr_state) in enumerate(all_states)

            for (a_ind,mr_action) in enumerate(all_actions)
                individual_next_probs  = Matrix{Array{Float64}}(undef, n_robots, 1) # Stores probabilities for next states for individual robots
                for k=1:n_robots
                    x = mr_state.robots[k].x
                    y = mr_state.robots[k].y
                    if (x >= robots[k].size_x) || (x<0)
                        # Return terminal state
                        individual_next_probs[k] = [1.0]
                    else
                        if !in(k,mr_action) # autonomous
                            if mr_state.robots[k].y == 1
                                individual_next_probs[k] = [robots[k].p0_ii[x], robots[k].p0_ie[x], 1-robots[k].p0_ii[x]-robots[k].p0_ie[x]]
                            else
                                individual_next_probs[k] = [0.0, 1.0, 0.0]
                            end
                        elseif in(k, mr_action) # teleoperate
                            if mr_state.robots[k].y == 1
                                individual_next_probs[k] = [robots[k].p1_ii[x], robots[k].p1_ie[x], 1-robots[k].p1_ii[x]-robots[k].p1_ie[x]]
                            else
                                individual_next_probs[k] = [robots[k].p1_ei[x], robots[k].p1_ee[x], 1-robots[k].p1_ei[x]-robots[k].p1_ee[x]]
                            end
                        end
                    end
                end
                next_probs = Iterators.product(individual_next_probs...)
                next_probs = vec(map(x -> prod(x), next_probs))
                state_transition_table[s_ind,a_ind] = next_states_dict[s_ind]
                state_transition_probs[s_ind,a_ind] = next_probs
            end
        end

        mr_mdp = MultiRobot_TeleopMDP(size_x=map_size[1], n_robots=n_robots, n_opr=n_opr, robots=robots, all_states=all_states,
                    r_operate=lambda, r_not_goal=rho, r_error_state=rho_e, discount_factor = discount_factor,
                    state_transition_table=state_transition_table, state_transition_probs=state_transition_probs)

        T_Mat,R_Mat,Ter,df = get_tabularMDP_data(mr_mdp)
        tabular_mr_mdp = SparseTabularMDP(T_Mat, R_Mat, Ter, df);
        tabular_mr_policy = solve(solver, tabular_mr_mdp)

## Simulate with optimal policy
    temp_reward_Optimal = Float64[]
    for inner_itr = 1:n_inner_itr
        robots_states = initialstate(mr_mdp)
        robots_rewards = 0.0

        sim(tabular_mr_mdp, max_steps=400, initialstate=1) do s
            temp_state = mr_mdp.all_states[s]
            a = action(tabular_mr_policy, s)
            rr = R_Mat[s,a]
            robots_rewards += rr
            return a
        end

        push!(temp_reward_Optimal, robots_rewards/n_robots)
    end
    push!(total_reward_Optimal, mean(temp_reward_Optimal))

## Simulate with Index policy
        temp_reward_WI = Float64[]
        whittleindex_matrix = getWhittleIndices_Offline(mr_mdp.robots, solver, -250.0, 10.0)
        for inner_itr = 1:n_inner_itr
            robots_states = initialstate(mr_mdp)
            robots_rewards = 0.0
            sim(tabular_mr_mdp, max_steps=400, initialstate=1) do s
                temp_state = mr_mdp.all_states[s]
                a = get_action_Index_policy(:WI, policies, temp_state, mr_mdp, whittleindex_matrix)
                rr = R_Mat[s,a]
                robots_rewards += rr
                return a
            end
            push!(temp_reward_WI, robots_rewards/n_robots)
        end
        push!(total_reward_WI, mean(temp_reward_WI))
end
all_states = Nothing
Base.GC.gc() # Runs garbage collector

percentile_cost = (total_reward_WI)./(total_reward_Optimal)
percentile_cost2 = (total_reward_D_lambda)./(total_reward_Optimal)
writedlm( "total_reward_OP_for_opr,robots=$n_opr,$n_robots.csv",  total_reward_Optimal, ',')
writedlm( "total_reward_WI_for_opr,robots=$n_opr,$n_robots.csv",  total_reward_WI, ',')
p = histogram([0.7:0.9], percentile_cost)
display(p)

end
tock()
end
