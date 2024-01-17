## For a single-robot MDP

struct MDP_State
    x::Int64 # Waypoint index
    y::Int64 # Internal State
end

@with_kw mutable struct TeleopMDP <: MDP{MDP_State, Symbol} # POMDP{State, Action, Observation}
    size_x::Int64               = 10 # Number of waypoints
    size_y::Int64               = 2 # y=1 for normal states, y=2 for error states
    goal::Vector{Int64}         = [5,1] # Default goal
    start::Vector{Int64}        = [1,1]
    p0_ii                       = fill(0.4, 15, 1) 
    p0_ie                       = fill(0.4, 15, 1) 
    p1_ii                       = fill(0.3, 15, 1) 
    p1_ie                       = fill(0.3, 15, 1) 
    p1_ei                       = fill(0.3, 15, 1) 
    p1_ee                       = fill(0.3, 15, 1) 
    r_operate::Float64          = -0.5  # reward for teleoperating
    r_goal::Float64             = 0.0 # reward for reaching goal
    r_error_state::Float64      = -2.0 # when at an error state
    r_not_goal::Float64         = -1.0 # when at any other state
    discount_factor::Float64    = 0.95 # This is for calculating discounted rewards by the POMDP solver
    is_passive::Int64           = 0 # This flag is used to define action set, depending on what kind of policy we want to find, e.g., passive one
end

MDP_State() = MDP_State(1, 1) # Index starts at (1,1)
MDP_State(x::Int64) = MDP_State(x, 1)
## Actions
function POMDPs.actions(mdp::TeleopMDP)
    if mdp.is_passive == 0
        return [:passive, :active]
    else
        return [:passive]
    end
end

Base.rand(rng::AbstractRNG, t::NTuple{L,Symbol}) where L =
    t[rand(rng, 1:length(t))]

function POMDPs.actionindex(mdp::TeleopMDP, a::Symbol)
    return Int64(a == :active) + 1
end

## States
POMDPs.initialstate(mdp::TeleopMDP) = MDP_State(mdp.start[1], mdp.start[2]) # Not used here but some simulators use this.
POMDPs.initialstate(mdp::TeleopMDP, ::AbstractRNG) = MDP_State(mdp.start[1], mdp.start[2]) # gets called at random initialization of state. Here it's set to start at a fixed state.

function POMDPs.states(mdp::TeleopMDP)
    # This defines state space
    ss = MDP_State[]
    # loop over all possible states
    for x = 1:mdp.size_x, y = 1:mdp.size_y
        push!(ss, MDP_State(x, y))
    end
    push!(ss, MDP_State(-1, -1)) # Signifies terminal state
    return ss
end

# POMDPs.n_states(mdp::TeleopMDP) = length(states(mdp))  # each index can have any of the 4 suggestion flags, 2 values for pa and 2 values for beta

function POMDPs.stateindex(mdp::TeleopMDP, s::MDP_State)
    return findfirst(x->x==s, states(mdp))
end

# State index function when all states are already computed
function POMDPs.stateindex(mdp::TeleopMDP, s::MDP_State, all_states::Vector{MDP_State})
    return findfirst(x->x==s, all_states)
end

POMDPs.discount(mdp::TeleopMDP) = mdp.discount_factor
POMDPs.isterminal(mdp::TeleopMDP, s::MDP_State) = s.x < 0 # -1 is terminal state, and once you get to the goal state you terminate in the next step.


## Transitions
function POMDPs.transition(mdp::TeleopMDP, state::MDP_State, action::Symbol)
    ## Based on state-action pair, return SparseCat(list of new possible states, list of their respective probabilities)
    if (state.x == mdp.size_x) || (state.x < 0)
        return SparseCat([MDP_State(-1, -1)], [1.0]) # Deterministic(MDP_State(-1, -1))
    end

    if action == :passive # autonomous
        if state.y == 1
            p_stay = mdp.p0_ii[state.x]
            p_error = mdp.p0_ie[state.x]
            p_fwd = abs(1.0 - p_stay - p_error)
        else
            p_stay = 0.0
            p_error = 1.0
            p_fwd = 0.0 # 1 - p_stay - p_error
        end
    elseif action == :active # teleoperate
        if state.y == 1
            p_stay = mdp.p1_ii[state.x]
            p_error = mdp.p1_ie[state.x]
            p_fwd = abs(1.0 - p_stay - p_error)
        else
            p_stay = mdp.p1_ei[state.x]
            p_error = mdp.p1_ee[state.x]
            p_fwd = abs(1.0 - p_stay - p_error)
        end
    end
    possible_next_states = @MVector([MDP_State(state.x, 1), MDP_State(state.x, 2), MDP_State(state.x+1, 1)])
    next_states = SparseCat(possible_next_states, @MVector[p_stay, p_error, p_fwd])
    return next_states
end

# Rewards, depending on state and action taken.
function POMDPs.reward(mdp::TeleopMDP, state::MDP_State, action::Symbol)
    # if state is goal state, give high positive reward, if not, give a small negative reward for being in that state.
    r = 0.0
    if state.x == mdp.size_x # when goal reached
        r += mdp.r_goal
    elseif state.x == -1
        r += 0 # No cost for being in terminal state.
    else
        if state.y == 2 # Error state
            r += mdp.r_error_state
        else
            r += mdp.r_not_goal
        end
    end
    if action == :active # when teleoperating
        r += mdp.r_operate
    end
    return r
end

POMDPs.reward(mdp::TeleopMDP, s::MDP_State, a::Symbol, sp::MDP_State) = reward(mdp, s, a)

function POMDPs.initialstate_distribution(mdp::TeleopMDP)
    possible_states = MDP_State(mdp.start[1],mdp.start[2]) # Deterministically start
    probs = [1.0]
    return SparseCat(possible_states, probs)
end
