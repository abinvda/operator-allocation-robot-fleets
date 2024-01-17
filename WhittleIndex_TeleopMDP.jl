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
using Colors
using Plots
using FIB
using Printf
using Distributions # For Normal , norm functions
using LinearAlgebra
using BasicPOMCP

## Calculate Whittle index for a given state of a given MDP
function getWhittleIndex(mdp::TeleopMDP, solver::SparseValueIterationSolver, state::MDP_State, lambda_min::Float64, lambda_max::Float64)
    mdp_temp = deepcopy(mdp) # NOTE: deepcopy prevents changes in mdp_temp to affect the mdp in the original teleoperation file.
    lambda_high = mdp_temp.r_operate # lambda_max
    lambda_low = lambda_min

    while lambda_high-lambda_low > 0.0001
        lambda_current = (lambda_high+lambda_low)/2
        mdp_temp.r_operate = lambda_current

        optimal_action = POMDPs.action(solve(solver, mdp_temp), state)
        if optimal_action == :active # If teleoperate, further increase the cost
            lambda_high = lambda_current
        else
            lambda_low = lambda_current
        end
    end
    lambda_current = (lambda_high+lambda_low)/2
    wi = lambda_current - mdp.r_operate
    return wi
end


function getWhittleIndices_Offline(all_robots::Array{TeleopMDP}, solver::SparseValueIterationSolver, lambda_min::Float64, lambda_max::Float64)
    # Do binary search between mdp.lambda and lambda_max
    # since lambda here is negative, lambda_max is the lowest lambda
    n_robots = length(all_robots)
    all_WI = zeros(length(states(all_robots[1])), n_robots) # If robots have different number of waypoints, pick the one with most number of waypoints.

    for k=1:n_robots
        mdp_temp = deepcopy(all_robots[k]) # NOTE: deepcopy prevents changes in mdp_temp to affect the mdp in the original teleoperation file.

        for (si, state) in enumerate(states(mdp_temp))
            lambda_high = lambda_max
            lambda_low = lambda_min
            while lambda_high-lambda_low > 0.01
                lambda_current = (lambda_high+lambda_low)/2
                mdp_temp.r_operate = lambda_current
                optimal_action = POMDPs.action(solve(solver, mdp_temp), state)
                if optimal_action == :active || optimal_action == 2 # If teleoperate, further increase the cost
                    lambda_high = lambda_current
                else
                    lambda_low = lambda_current
                end
            end
            lambda_current = (lambda_high+lambda_low)/2
            wi = lambda_current - all_robots[k].r_operate
            all_WI[stateindex(mdp_temp, state),k] = wi
        end
    end
    return all_WI
end


function adaptive_greedy(all_robots::Array{TeleopMDP})
    n_robots = length(all_robots)
    all_WI = zeros(length(states(all_robots[1])),n_robots)

    for k=1:n_robots
        P = [] # Initialize passive set
        mdp = all_robots[k]
        all_states = states(mdp) # setdiff(states(mdp), [MDP_State(-1,-1)])
        WhittleIndexVector = zeros(length(all_states),1)
        while Set(P) != Set(all_states)
            # Compute mu_star for all y not in P
            set_y = setdiff(all_states,P)
            mu_star = [get_mu_star(mdp, P, y, all_states) for y in set_y]
            # Find min and arg min of mu_star
            min_mu = minimum(mu_star)
            argmin_mu = findall(==(minimum(mu_star)), mu_star)
            # Assign whittle indices and update P
            for y_ind in argmin_mu
                st = stateindex(mdp, set_y[y_ind])
                WhittleIndexVector[st] = min_mu
                push!(P, all_states[st])
            end
        end
        all_WI[:,k] = WhittleIndexVector
    end
    return all_WI
end
function get_mu_star(mdp, P, y, all_states)
    # First compute required parameters for set P
    PI = [!in(s,P) for s in all_states] # Policy vector
    Tpi = zeros(length(all_states), length(all_states)) # Transition matrix under policy PI
    for (si,s) in enumerate(all_states)
        t = transition(mdp, s, PI[si]==1 ? :active : :passive)
        for temp_ind in 1:length(t.vals)
            spi = stateindex(mdp, t.vals[temp_ind], all_states)
            Tpi[si,spi] = t.probs[temp_ind] # The transition prob of s to sp under policy pi
        end
    end
    Cpi = zeros(length(all_states), 1) # Cost vector under policy PI
    for (si,s) in enumerate(all_states)
        Cpi[si] = -reward(mdp,s, PI[si]==1 ? :active : :passive)
    end
    mt = pinv(I - mdp.discount_factor*Tpi)
    DP = mt*Cpi
    NP = mt*PI

    # Then compute required parameters for set P \cup {y}
    PIy = [!in(s,P) && s!=y for s in all_states]
    Tpiy = zeros(length(all_states), length(all_states))
    for (si,s) in enumerate(all_states)
        t = transition(mdp, s, PIy[si]==1 ? :active : :passive)
        for temp_ind in 1:length(t.vals)
            spi = stateindex(mdp, t.vals[temp_ind], all_states)
            Tpiy[si,spi] = t.probs[temp_ind] # The transition prob of s to sp under policy pi
        end
    end
    Cpiy = zeros(length(all_states), 1)
    for (si,s) in enumerate(all_states)
        Cpiy[si] = -reward(mdp,s, PIy[si]==1 ? :active : :passive)
    end

    mty = pinv(I - mdp.discount_factor*Tpiy)
    DPy = mty*Cpiy
    NPy = mty*PIy

    bad_s = Int64[] # These are the states that do not lie in the set \LAMBDA_y
    for i in 1:length(NP)
        if abs(NP[i] - NPy[i]) <= 10^-5
            push!(bad_s, i)
        end
    end
    LAMBDA_y = setdiff(1:length(all_states), bad_s)
    # Now calculate mu and mu_star
    mu = [(DPy[s] - DP[s])/(NP[s] - NPy[s]) for s in LAMBDA_y]
    mu_star = minimum(mu)
    return mu_star
end