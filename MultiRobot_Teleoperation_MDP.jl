# Model of POMDP version of the toy example (UPDATED: Version 2.0)

include("SingleRobot_Teleoperation_MDP.jl")

struct MultiRobot_State
    robots::Array{MDP_State}
end

# For the multi-robot MDP, state is a vector of all single-robot states and action is a vector of indices of the robots to teleoperate. 
@with_kw mutable struct MultiRobot_TeleopMDP <: MDP{MultiRobot_State, Vector{Int64}} # POMDP{State, Action, Observation}
    size_x::Int64               = 5 # Number of waypoints
    size_y::Int64               = 2 # y=1 for normal states, y=2 for error states
    n_robots::Int64             = 2
    n_opr::Int64                = 1
    goal::Vector{Int64}         = [5,1]
    start::Vector{Int64}        = [1,1]
    robots                      = [TeleopMDP(), TeleopMDP()]#, TeleopMDP()]#, TeleopMDP()]
    r_operate::Float64          = -0.1  # reward for teleoperating
    r_goal::Float64             = 0.0 # reward for reaching goal
    r_error_state::Float64      = -2.0 # when at an error state
    r_not_goal::Float64         = -1.0 # when at any other state
    discount_factor::Float64    = 0.95 # This is for calculating discounted rewards by the POMDP solver
    all_states::Vector{MultiRobot_State} = []
    state_transition_table::Array{Array{Int64,1},2} = [] # ::Array{Array{SparseCat{Array{MultiRobot_State,1},Array{Float64,1}},1}}
    state_transition_probs::Array{Array{Float64,1},2} = []
end

## Default states
MultiRobot_State(mdp::MultiRobot_TeleopMDP) = MultiRobot_State(fill(MDP_State(),mdp.n_robots)) # means all robots are initialized at the default state
MultiRobot_State(n_robots::Int64) = MultiRobot_State(fill(MDP_State(),n_robots))
## Actions
function POMDPs.actions(mdp::MultiRobot_TeleopMDP)
    return unique(collect(combinations([fill(0, mdp.n_opr);1:mdp.n_robots],mdp.n_opr))) # This provides us with all possible actions. At most mdp.n_opr robots can be active at a time.
end

Base.rand(rng::AbstractRNG, t::NTuple{L,Symbol}) where L = t[rand(rng, 1:length(t))]

POMDPs.actionindex(mdp::MultiRobot_TeleopMDP, a::Vector{Int64}) = findfirst(x->Set(x)==Set(a), unique(collect(combinations([fill(0, mdp.n_opr);1:mdp.n_robots],mdp.n_opr))))

## States
POMDPs.initialstate(mdp::MultiRobot_TeleopMDP) = MultiRobot_State(fill(MDP_State(1,1), mdp.n_robots)) # Not used here but some simulators use this.

POMDPs.initialstate(mdp::MultiRobot_TeleopMDP, ::AbstractRNG) = POMDPs.initialstate(mdp::MultiRobot_TeleopMDP) # gets called at random initialization of state. Here, it is set to start at a fixed state index 1.

POMDPs.states(mdp::MultiRobot_TeleopMDP) = mdp.all_states # This defines state space. Here, we compute the state space separately and provide as an argument when initializing the MDP.

POMDPs.stateindex(mdp::MultiRobot_TeleopMDP, s::MultiRobot_State) = findfirst(x->x.robots==s.robots, mdp.all_states)

POMDPs.discount(mdp::MultiRobot_TeleopMDP) = mdp.discount_factor
POMDPs.isterminal(mdp::MultiRobot_TeleopMDP, s::MultiRobot_State) = all(st->st.x<0, s.robots)


## Transitions
function POMDPs.transition(mdp::MultiRobot_TeleopMDP, mr_state::MultiRobot_State, mr_action::Vector{Int64}, just_next_states::Bool = false)
    ## Based on state-action pair, return SparseCat(list of new possible states, list of their respective probabilities)

    # NOTE: The following block pre-defines the size of individual_next_states vector:
    individual_next_states =  Vector{Vector{MDP_State}}(undef, mdp.n_robots) # Stores next states for individual robots as separate elements
    for k = 1:mdp.n_robots
        if (mr_state.robots[k].x >= mdp.robots[k].size_x) || (mr_state.robots[k].x<0)
            individual_next_states[k] = [MDP_State(-1, -1)] # Deterministic(MDP_State(-1, -1))
        else
            individual_next_states[k] = [MDP_State(mr_state.robots[k].x, 1), MDP_State(mr_state.robots[k].x, 2), MDP_State(mr_state.robots[k].x+1, 1)]
        end
    end
    next_states = vec(map(x -> MultiRobot_State(collect(x)), Iterators.product(individual_next_states...))) # This gets all possible combinations of next states for individual robots
    if just_next_states # Allowing for returing of just the list of states instead of a distribution.
        return next_states
    end
    # If state_transition_table is present, just uncomment the following line.
    # return SparseCat(next_states, mdp.state_transition_table[stateindex(mdp, mr_state),mr_action + 1])

    individual_next_probs  = Matrix{Array{Float64}}(undef, mdp.n_robots, 1) # Array{Float64,1}[] # Stores probabilities for next states for individual robots
    for k=1:mdp.n_robots
        x = mr_state.robots[k].x
        y = mr_state.robots[k].y
        if (x >= mdp.robots[k].size_x) || (x<0)
            # Return terminal state
            individual_next_probs[k] = [1.0]
        else
            if !in(k, mr_action) # autonomous
                if mr_state.robots[k].y == 1
                    individual_next_probs[k] = [mdp.robots[k].p0_ii[x], mdp.robots[k].p0_ie[x], 1-mdp.robots[k].p0_ii[x]-mdp.robots[k].p0_ie[x]]
                else
                    individual_next_probs[k] = [0.0, 1.0, 0.0]
                end
            elseif in(k, mr_action) # teleoperate
                if mr_state.robots[k].y == 1
                    individual_next_probs[k] = [mdp.robots[k].p1_ii[x], mdp.robots[k].p1_ie[x], 1-mdp.robots[k].p1_ii[x]-mdp.robots[k].p1_ie[x]]
                else
                    individual_next_probs[k] = [mdp.robots[k].p1_ei[x], mdp.robots[k].p1_ee[x], 1-mdp.robots[k].p1_ei[x]-mdp.robots[k].p1_ee[x]]
                end
            end
        end
    end
        next_probs = Iterators.product(individual_next_probs...)
        next_probs = map(x -> prod(x), next_probs)
        return SparseCat(next_states, next_probs)
end

# Rewards, depending on state and action taken.
function POMDPs.reward(mdp::MultiRobot_TeleopMDP, state::MultiRobot_State, action::Vector{Int64})
    r = 0.0
    for k = 1:mdp.n_robots
        if state.robots[k].x == mdp.robots[k].size_x # when goal reached
            r += mdp.robots[k].r_goal
        elseif state.robots[k].x == -1 # No cost for being in terminal state.
            r += 0 
        elseif state.robots[k].y == 2 # Error state
            r += mdp.robots[k].r_error_state
        else
            r += mdp.robots[k].r_not_goal
        end
    end
    r += sum(action.>0)*mdp.r_operate # Add teleoperation cost for each robot being teleoperated.
    return r
end

POMDPs.reward(mdp::MultiRobot_TeleopMDP, s::MultiRobot_State, a::Int64, sp::MultiRobot_State) = reward(mdp, s, a)

function POMDPs.initialstate_distribution(mdp::MultiRobot_TeleopMDP)
    possible_states = MDP_State[]
    for k = 1:mdp.n_robots
        push!(possible_states, MDP_State(1,1)) # Deterministically start at the start state
    end
    return Deterministic(MultiRobot_State(possible_states))
end
