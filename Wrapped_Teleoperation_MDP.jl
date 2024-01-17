using SparseArrays

## Define a function that returns 4 things: T=transition matrix as a SparseMatrixCSC type. R=Reward matrix, Ter=Set of terminal states, discount factor
function get_tabularMDP_data(mdp::MultiRobot_TeleopMDP)
    T_Mat = SparseMatrixCSC{Float64,Int64}[]
    # Sparse representation of transition martix needs three lists:
    # 1. list_s = [list of current state indices],
    # 2. list_sp = [list of next state indices],
    # 3. list_p = [list of corresponding probabilities (list_p[i] is the prob. of transition from list_s[i] to list_sp[i])]
    # And this is repeated for each action
    for (a_ind,mr_action) in enumerate(actions(mdp))
        list_s = Int64[]
        list_sp = Int64[]
        list_pr = Float64[]
        for si in 1:size(mdp.state_transition_table,1)
            push!(list_s, fill(si, length(mdp.state_transition_table[si,a_ind]))...)
            push!(list_sp, mdp.state_transition_table[si,a_ind]...)
            push!(list_pr, mdp.state_transition_probs[si,a_ind]...)
        end
        push!(T_Mat, sparse(list_s, list_sp, list_pr))
    end

    # Reward matrix is simply |S|x|A| state-action pair rewards
    R_Mat = zeros(length(mdp.all_states), length(actions(mdp)))
    for (si, mr_state) in enumerate(mdp.all_states), (a_ind,mr_action) in enumerate(actions(mdp))
        for k = 1:mdp.n_robots
            if mr_state.robots[k].x == mdp.robots[k].size_x # when goal reached
                R_Mat[si, a_ind] += mdp.robots[k].r_goal
            elseif mr_state.robots[k].x == -1
                R_Mat[si, a_ind] += 0 # No cost for being in terminal state.
            else
                if mr_state.robots[k].y == 2 # Error state
                    R_Mat[si, a_ind] += mdp.robots[k].r_error_state
                else
                    R_Mat[si, a_ind] += mdp.robots[k].r_not_goal
                end
            end
        end
        R_Mat[si, a_ind] += sum(mr_action.>0)*mdp.r_operate
    end
    # Now create a set of terminal states
    # Ter = Int64[] # Set of terminal states indices. Since there is only one terminal state in our problem, we dont need to create a list.
    # The way states are ordered, last state is our terminal state.
    term_state = Set([length(mdp.all_states)])

    discount_factor = mdp.discount_factor

return T_Mat, R_Mat, term_state, discount_factor
end

function get_individual_next_states(current_x::Int64, size_x::Int64)
    if (current_x >= size_x) || (current_x<0)
        # Return terminal state
        return [MDP_State(-1, -1)]
    else
        return [MDP_State(current_x, 1), MDP_State(current_x, 2), MDP_State(current_x+1, 1)]
    end
end

function get_next_states_single(all_states::Array{MultiRobot_State,1}, mr_state::MultiRobot_State, max_x::Int64)
    individual_next_states = map(rob -> get_individual_next_states(rob.x, max_x), mr_state.robots)
    next_states = Iterators.product(individual_next_states...) # This gets all possible combinations of next states for individual robots
    next_states = vec(map(x -> MultiRobot_State(collect(x)), next_states))
    next_states = vec(map(x -> findfirst(i->i.robots==x.robots, all_states), next_states)) # This changes next states to state indices
    return next_states
end

function get_next_states_dict_cheap(all_states::Array{MultiRobot_State,1}, max_x::Int64, single_mdp::TeleopMDP, single_SS_size::Int64, num_robots::Int64)
    next_states_dict = Array{Array{Int64}}(undef, length(all_states),1) # This means we are saving the desired data as array of arrays not as a dict.
    for (s_ind,mr_state) in enumerate(all_states)
        individual_next_states = map(rob -> get_individual_next_states(rob.x, max_x), mr_state.robots)
        next_states = Iterators.product(individual_next_states...) # This gets all possible combinations of next states for individual robots
        # The following is just a faster way of computing indices of next states in multi-robot systems when the number of robots is small. For larger number of robots we run the function get_next_state_singles which is quite slow.
        if num_robots == 1
            next_states = vec(map(s -> stateindex(single_mdp, s[1]), next_states))
        elseif num_robots == 2
            next_states = vec(map(s -> stateindex(single_mdp, s[1]) + (stateindex(single_mdp, s[2]) - 1)*single_SS_size, next_states))
        elseif num_robots == 3
            next_states = vec(map(s -> stateindex(single_mdp, s[1]) + (stateindex(single_mdp, s[2]) - 1)*single_SS_size + (stateindex(single_mdp, s[3]) - 1)*(single_SS_size^2), next_states))
        elseif num_robots == 4
            next_states = vec(map(s -> stateindex(single_mdp, s[1]) + (stateindex(single_mdp, s[2]) - 1)*single_SS_size + (stateindex(single_mdp, s[3]) - 1)*(single_SS_size^2) + (stateindex(single_mdp, s[4]) - 1)*(single_SS_size^3), next_states))
        elseif num_robots == 5
            next_states = vec(map(s -> stateindex(single_mdp, s[1]) + (stateindex(single_mdp, s[2]) - 1)*single_SS_size + (stateindex(single_mdp, s[3]) - 1)*(single_SS_size^2) + (stateindex(single_mdp, s[4]) - 1)*(single_SS_size^3) + (stateindex(single_mdp, s[5]) - 1)*(single_SS_size^4), next_states))
        else
            next_states = get_next_states_single(all_states, mr_state, max_x, single_mdp)
        end
        next_states_dict[s_ind] = next_states
    end
    return next_states_dict
end

function get_next_states_dict(all_states::Array{MultiRobot_State,1}, max_x::Int64)
    next_states_dict = Array{Array{Int64}}(undef, length(all_states),1) # This means we are saving the desired data as array of arrays not as a dict.
    for (s_ind,mr_state) in enumerate(all_states)
        next_states_dict[s_ind] = get_next_states_single(all_states, mr_state, max_x)
    end
    return next_states_dict
end

function get_individual_next_probs(current_x::Int64, size_x::Int64) # See if this should be re-named to ..._next_states
    if (current_x >= size_x) || (current_x<0)
        # Return terminal state
        return [MDP_State(-1, -1)]
    else
        return [MDP_State(current_x, 1), MDP_State(current_x, 2), MDP_State(current_x+1, 1)]
    end
end

function get_state_index(current_state, all_states)
    return findfirst(i->i.robots==current_state.robots, all_states)
end

function get_transition_probabilities(max_x::Int64, allowed_types::Array{Int64}, IsRandom::Symbol=:No, return_prob_types::Symbol=:No, gamma::Float64=1.0)
    state_type = rand(allowed_types, max_x, 1) # This randomly assigns one of the allowed types to each state index (waypoint)

    # Initialize probabilities lists
    r0_n0, q0_n0, r1_n0, q1_n0, q1_n1, r1_n1 = [zeros(max_x, 1) for i = 1:6]
    if IsRandom == :No # Default
        for i = 1:max_x
            r0_n0[i], q0_n0[i], r1_n0[i], q1_n0[i], q1_n1[i], r1_n1[i] = sample_individual_probabilities(state_type[i], 1)
        end
        if return_prob_types == :No # Default
            return [r0_n0, q0_n0, r1_n0, q1_n0, q1_n1, r1_n1]
        else
            return [[r0_n0, q0_n0, r1_n0, q1_n0, q1_n1, r1_n1], state_type]
        end
    else
        for i = 1:max_x
            r0_n0[i], q0_n0[i], r1_n0[i], q1_n0[i], q1_n1[i], r1_n1[i] = sample_individual_probabilities(state_type[i], 2)
        end
        if return_prob_types != :No
            return [[r0_n0, q0_n0, r1_n0, q1_n0, q1_n1, r1_n1], state_type]
        else
            return [r0_n0, q0_n0, r1_n0, q1_n0, q1_n1, r1_n1]
        end
    end
end

function sample_individual_probabilities(state_type::Int64, prob_set::Int64=1, gamma::Float64=1.0)
    if prob_set == 1 # Conditions on teleoperation probabilities for type-3
        if state_type == 1
            # Sample type-1 probabilities
            ind_r0_n0 = 0.4 + rand()*0.4
            ind_q0_n0 = 0.0

            ind_r1_n0 = 0.2 + rand()*0.3
            ind_q1_n0 = 0.0

            ind_q1_n1 = 0.0
            ind_r1_n1 = 0.3
        elseif state_type == 2
            # Sample type-2 probabilities
            ind_r0_n0 = 0.2 + rand()*0.3
            ind_q0_n0 = 0.2 + rand()*0.3

            ind_r1_n0 = 0.1 + rand()*0.3
            ind_q1_n0 = 0.0

            ind_q1_n1 = 0.0
            ind_r1_n1 = ind_r1_n0
        else
            # Sample type-3 probabilities
            ind_r0_n0 = 0.2 + rand()*0.3
            ind_r1_n0 = 0.1 + rand()*0.3

            x = clamp((1 - gamma*ind_r0_n0)/(gamma*(1+gamma*(1 - ind_r1_n0))), 0.1, 1.0-ind_r0_n0)
            ind_q0_n0 = x*rand()
            ind_q1_n0 = 0.0

            x = clamp((1/gamma) - ((gamma*ind_q0_n0*(1 - ind_r1_n0))/(1- gamma*ind_r0_n0 - gamma*ind_q0_n0)), 0.1, 0.9)
            ind_r1_n1 = x*rand() # p1_ee \in [0, x]
            ind_q1_n1 = 1.0 - ind_r1_n1

        end
    elseif prob_set == 2 # All random prababilities
        ind_r0_n0 = rand()
        ind_q0_n0 = rand()
        ind_p0_n0 = rand()

        sum_p0 = ind_r0_n0 + ind_q0_n0 + ind_p0_n0
        ind_r0_n0 = ind_r0_n0./sum_p0
        ind_q0_n0 = ind_q0_n0./sum_p0
        ind_p0_n0 = ind_p0_n0./sum_p0

        ind_r1_n0 = rand()
        ind_q1_n0 = rand()
        ind_p1_n0 = rand()

        sum_p1 = ind_r1_n0 + ind_q1_n0 + ind_p1_n0
        ind_r1_n0 = ind_r1_n0./sum_p1
        ind_q1_n0 = ind_q1_n0./sum_p1
        ind_p1_n0 = ind_p1_n0./sum_p1

        ind_r1_n1 = rand()
        ind_q1_n1 = rand()
        ind_p1_n1 = rand()

        sum_p1 = ind_r1_n1 + ind_q1_n1 + ind_p1_n1
        ind_r1_n1 = ind_r1_n1./sum_p1
        ind_q1_n1 = ind_q1_n1./sum_p1
        ind_p1_n1 = ind_p1_n1./sum_p1
    end

    return [ind_r0_n0, ind_q0_n0, ind_r1_n0, ind_q1_n0, ind_q1_n1, ind_r1_n1]
end

function check_conditions(probabilities, gamma)
    for ind in length(probabilities[1])
        r0_n0 = probabilities[1][ind]
        q0_n0 = probabilities[2][ind]
        r1_n0 = probabilities[3][ind]
        q1_n0 = probabilities[4][ind]
        q1_n1 = probabilities[5][ind]
        r1_n1 = probabilities[6][ind]

        p0_n0 = 1.0 - r0_n0 - q0_n0
        p1_n0 = 1.0 - r1_n0 - q1_n0
        p1_n1 = 1.0 - r1_n1 - q1_n1
        q0_n0_upper_bound = (1 - gamma*r0_n0)/(gamma*(1+gamma*(1 - r1_n0)))
        q1_n1_lower_bound = 1 - (1/gamma) + ((gamma*q0_n0*(1 - r1_n0))/(1- gamma*r0_n0 - gamma*q0_n0))

        alpha_1 = 1 + (gamma*q1_n0)/(1 - gamma *r1_n1) + (gamma *q0_n0 *(gamma *r1_n0 + (gamma^2 * q1_n0*q1_n1)/(1 - gamma *r1_n1) - 1))/(1 - gamma *r1_n1 - gamma*r0_n0 + gamma^2 *r1_n1* r0_n0 - gamma^2 *q0_n0* q1_n1)

        beta_0 = (gamma*(p1_n0 - p0_n0) + gamma^2 *(p0_n0 * r1_n0 - p1_n0 * r0_n0))/(1 - gamma *r0_n0)

        if (alpha_1 >= 0) && (beta_0/(1-gamma) >= -1)
            return 1
        end
    end
    return 0
end
