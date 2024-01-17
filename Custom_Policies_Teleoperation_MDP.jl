using SparseArrays

## Define custom policies
function get_action_Benefit_policy(policies::Array{ValueIterationPolicy}, current_states::Array{MDP_State}, mr_mdp::MultiRobot_TeleopMDP)
        ## Calculate D(s)
        ac = Int64[]
        Benefits = Float64[]
        for k=1:length(current_states)
            # Calculate Q-values
            Q_1 = value(policies[k], current_states[k], :active)
            Q_0 = value(policies[k], current_states[k], :passive)
            push!(Benefits, Q_1-Q_0)
        end
        for opr = 1:mr_mdp.n_opr
            best_robot = findmax(Benefits)
            if best_robot[1] > 0.0
                push!(ac, best_robot[2])
                Benefits[best_robot[2]] = -100.0
            else
                push!(ac, 0)
            end
        end
        return actionindex(mr_mdp, ac)
end

function get_action_Benefit_policy_individual(policies::Array{ValueIterationPolicy}, current_states::Array{MDP_State}, n_opr::Int64)
        ## Calculate D(s)
        Benefits = [value(policies[k], current_states[k], :active) - value(policies[k], current_states[k], :passive) for k in 1:length(current_states)]
        b = sortperm(-Benefits)
        ac = [Benefits[b[k]]>0.0 ? b[k] : 0 for k in 1:min(n_opr, length(current_states))]
        return ac
end

function get_action_reactive_policy(current_states::Array{MDP_State}, n_opr::Int64)
        ac = Int64[]
        error_robots = []
        for k=1:length(current_states)
            if current_states[k].y == 2
                push!(error_robots, k)
            end
        end
        shuffle!(error_robots)
        for opr = 1:min(n_opr, length(error_robots))
            push!(ac, error_robots[opr])
        end
        if isempty(ac)
            push!(ac, 0)
        end
        return ac
end

function get_action_Greedy5_policy_individual(current_states::Array{MDP_State}, all_robots::Array{TeleopMDP}, n_opr::Int64)
    # This is the multi-robot one-step cost minimization policy.
        ac = Int64[]
        GreedyQ = Float64[]
        n_robots = length(all_robots)
        all_actions = unique(collect(combinations([fill(0, n_opr);1:n_robots],n_opr)))
        for act in all_actions
            robots_actions = [in(x, act) ? :active : :passive for x in 1:n_robots]
            greedyReward = 0.0
            for k = 1:length(current_states)
                # Current state reward
                greedyReward += reward(all_robots[k], current_states[k], robots_actions[k])

                # Incentivize to assist in error state
                if in(k,act) && (current_states[k].y == 2 && current_states[k].x < all_robots[k].size_x)
                    greedyReward += 100*abs(all_robots[k].r_operate) # This forces to assist the robots in error states
                end

                next_states_dist = transition(all_robots[k], current_states[k], robots_actions[k])
                next_states = next_states_dist.vals
                next_probs = next_states_dist.probs
                next_rewards = vec(map(x -> reward(all_robots[k], x, :passive), next_states))

                for (ind,sp) in enumerate(next_states)
                    greedyReward += all_robots[k].discount_factor * next_probs[ind] * next_rewards[ind]
                end
            end
            push!(GreedyQ, greedyReward)
        end
        ## Select actions for all robots (Greedy Policy) and store rewards
        best_robot = findmax(GreedyQ)
        a = best_robot[2]
        return all_actions[a]
end


function get_action_Greedy6_policy_individual(current_states::Array{MDP_State}, all_robots::Array{TeleopMDP}, n_opr::Int64, passive_value_function::Array{ValueIterationPolicy}, L::Int64, return_value::Bool=false)
    # This is the multi-robot one-step lookahead cost minimization policy. This is same as the Myopic policy function below but it's a faster implementation given that it's one-step lookahead.
        ac = Int64[]
        GreedyQ = Float64[]
        n_robots = length(all_robots)
        all_actions = unique(collect(combinations([fill(0, n_opr);1:n_robots],n_opr)))
        for act in all_actions
            robots_actions = [in(x, act) ? :active : :passive for x in 1:n_robots]
            greedyReward = 0.0
            for k = 1:n_robots
                # Current state reward
                greedyReward += reward(all_robots[k], current_states[k], robots_actions[k])

                next_states_dist = transition(all_robots[k], current_states[k], robots_actions[k])
                next_states = next_states_dist.vals
                next_probs = next_states_dist.probs
                next_rewards = vec(map(x -> value(passive_value_function[k], x), next_states))

                for (ind,sp) in enumerate(next_states)
                    greedyReward += all_robots[k].discount_factor * next_probs[ind] * next_rewards[ind]
                end
            end
            push!(GreedyQ, greedyReward)
        end
        ## Select actions for all robots (Greedy Policy) and store rewards
        best_robot = findmax(GreedyQ)
        a = best_robot[2]
        if return_value == false
            return all_actions[a]
        else
            return best_robot[1]
        end
end


function get_action_Index_policy(policy_name::Symbol, policies::Array{ValueIterationPolicy}, current_state::MultiRobot_State, mr_mdp::MultiRobot_TeleopMDP, whittleindex_matrix::Array{Float64, 2})
    ac = Int64[]
    if policy_name == :WI
        WhittleIndices = [whittleindex_matrix[stateindex(mr_mdp.robots[k], current_state.robots[k]),k] for k in 1:mr_mdp.n_robots]
        b = sortperm(WhittleIndices)
        ac = [WhittleIndices[b[k]]<0.0 ? b[k] : 0 for k in 1:min(mr_mdp.n_opr, mr_mdp.n_robots)]
        return actionindex(mr_mdp, ac)
    else
        println("Incorrect policy name. Expected :WI, got $policy_name.")
    end
end

function get_action_Index_policy_individual(policies::Array{ValueIterationPolicy}, all_robots::Array{TeleopMDP}, current_states::Array{MDP_State}, whittleindex_matrix::Array{Float64, 2}, n_opr::Int64)
    n_robots = size(all_robots, 1)
    WhittleIndices = [whittleindex_matrix[stateindex(all_robots[k], current_states[k]),k] for k in 1:n_robots]
    b = sortperm(WhittleIndices)
    ac = [WhittleIndices[b[k]]<0.0 ? b[k] : 0 for k in 1:min(n_opr, n_robots)]
    return ac
end

function get_action_AG_policy_individual(policies::Array{ValueIterationPolicy}, all_robots::Array{TeleopMDP}, current_states::Array{MDP_State}, whittleindex_matrix::Array{Float64, 2}, n_opr::Int64)
    n_robots = size(all_robots, 1)
    WhittleIndices = [whittleindex_matrix[stateindex(all_robots[k], current_states[k]),k] for k in 1:n_robots]
    b = sortperm(WhittleIndices)
    ac = [WhittleIndices[b[k]]<0.0 ? b[k] : 0 for k in 1:min(n_opr, n_robots)]
    return ac
end


function get_action_L_step_myopic(current_states::Array{MDP_State}, all_robots::Array{TeleopMDP}, n_opr::Int64, L::Int64, passive_value_function::Array{ValueIterationPolicy}, return_value::Int64 = 0)
    # This is the k-step myopic policy from Rosenfeld 2017 paper.
    # NOTE: To speed up the computation, we can pre-compute the passive values of all states, so that we don't need to re-calculate it in every iteration. This should be easy as we only need to add single robot passive values.
        n_robots = length(all_robots)
        if L >= 1
            next_states = Vector{Vector{MDP_State}}(undef, n_robots) # A vector of next states of all robots
            for k = 1:n_robots
                next_states_dist = transition(all_robots[k], current_states[k], :active)
                next_states[k] = next_states_dist.vals
            end
            all_next_states = map(x -> collect(x), Iterators.product(next_states...))

            if L-1 == 1
                all_next_myopicRewards = zeros(length(all_next_states),1)
                for (ind,sp) in enumerate(all_next_states)
                    all_next_myopicRewards[ind] = get_action_Greedy6_policy_individual(sp, all_robots, n_opr, passive_value_function, 1, true)
                end
            else
                all_next_myopicRewards = zeros(length(all_next_states),1)
                for (ind,sp) in enumerate(all_next_states) 
                    all_next_myopicRewards[ind] = get_action_L_step_myopic(sp, all_robots, n_opr, L-1, passive_value_function, 1)
                end
            end

            MyopicQ = Float64[]
            all_actions = unique(collect(combinations([fill(0, n_opr);1:n_robots],n_opr)))
            for act in all_actions
                robots_actions = [in(x, act) ? :active : :passive for x in 1:n_robots]
                myopicReward = 0.0
                next_states = Vector{Vector{MDP_State}}(undef, n_robots) # A vector of next states of all robots
                next_probs = Vector{Vector{Float64}}(undef, n_robots)
                for k = 1:n_robots
                    # Current state reward
                    myopicReward += reward(all_robots[k], current_states[k], robots_actions[k])

                    # Add expected reward from next states
                    next_states_dist = transition(all_robots[k], current_states[k], robots_actions[k])
                    next_probs[k] = next_states_dist.probs
                end
                all_next_probs = Iterators.product(next_probs...)
                all_next_probs = map(x -> prod(x), all_next_probs)

                for (ind,sp) in enumerate(all_next_states)
                    myopicReward += all_robots[1].discount_factor * all_next_probs[ind] * all_next_myopicRewards[ind]
                end
                push!(MyopicQ, myopicReward)
            end
            best_action = findmax(MyopicQ)
            a = best_action[2]
            if return_value == 0
                return all_actions[a]
            else
                return best_action[1]
            end
        else
            myopicReward = 0.0
            for k in 1:n_robots
                myopicReward += value(passive_value_function[k], current_states[k])
            end
            return myopicReward
        end
end


function get_L_step_value_individual(current_state::MDP_State, one_robot::TeleopMDP, passive_value_function::ValueIterationPolicy, L::Int64, only_value::Bool = false)
    ind_MyopicQ = Float64[]
    if L >= 1
        for act in [:active, :passive]
            myopicReward = reward(one_robot, current_state, act)

            # Add expected reward from next states
            next_states_dist = transition(one_robot, current_state, act)
            next_states = next_states_dist.vals
            next_probs = next_states_dist.probs

            for (ind,sp) in enumerate(next_states)
                myopicReward += one_robot.discount_factor * next_probs[ind] * get_L_step_value_individual(sp, one_robot, passive_value_function, L-1, true)
            end
            push!(ind_MyopicQ, myopicReward)
        end
        best_action = findmax(ind_MyopicQ)
        a = best_action[2]
        act = a == 1 ? :active : :passive
        if only_value == true
            return best_action[1]
        else
            return act, best_action[1]
        end
    else
        myopicReward = value(passive_value_function, current_state)
        return myopicReward
    end
end