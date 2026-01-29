module GACore

using Random
using Statistics
using ProgressMeter

export GAParams, run_ga, entropy_bits

Base.@kwdef mutable struct GAParams
    popsize::Int = 200
    generations::Int = 10000 
    pc::Float64 = 0.90  #crossover prob
    pm::Float64 = 0.0  #mutation prob
    tour_k::Int = 3  #tournament size
    survivor_mode::Symbol = :elitist  #elitist / generational
    elite::Int = 2  #number of elites kept
    seed::Int = 42
    objective::Symbol = :max
    log_every::Int = 25

    # ----- generalized crowding -----
    crowding_alpha::Float64 = 0.0   # 0 = deterministic, 1 = probabilistic
    crowding_scale::Float64 = 1.0   # temperature / smoothness
end

# ----- utilities -----

random_individual(nbits::Int, rng::AbstractRNG) = BitVector(rand(rng, Bool, nbits))

function init_population(popsize::Int, nbits::Int, rng::AbstractRNG)
    [random_individual(nbits, rng) for _ in 1:popsize]
end

# Convert "raw fitness" to "score to maximize"
@inline function to_score(f::Float64, objective::Symbol)
    objective === :max && return f
    objective === :min && return -f
    error("objective must be :max or :min, got $objective")
end

function tournament_select(pop::Vector{BitVector}, scores::Vector{Float64},
                           k::Int, rng::AbstractRNG)
    best = rand(rng, eachindex(pop))
    bests = scores[best]
    for _ in 2:k
        i = rand(rng, eachindex(pop))
        if scores[i] > bests
            best, bests = i, scores[i]
        end
    end
    return pop[best]
end

function crossover(p1::BitVector, p2::BitVector, pc::Float64, rng::AbstractRNG)
    n = length(p1)
    @assert length(p2) == n
    if rand(rng) > pc
        return copy(p1), copy(p2)
    end
    cut = (n > 2) ? rand(rng, 2:(n-1)) : 1
    c1 = similar(p1); c2 = similar(p2)
    c1[1:cut] = p1[1:cut];      c1[(cut+1):n] = p2[(cut+1):n]
    c2[1:cut] = p2[1:cut];      c2[(cut+1):n] = p1[(cut+1):n]
    return c1, c2
end

function mutate!(x::BitVector, pm::Float64, rng::AbstractRNG)
    @inbounds for i in eachindex(x)
        if rand(rng) < pm
            x[i] = !x[i]
        end
    end
    return x
end

# Survivor selection A: generational
survivors_generational(_parents, offspring, _score_par, _score_off, popsize::Int) =
    offspring[1:popsize]

# Survivor selection B: elitist from parents+offspring (by score)
function survivors_elitist(parents::Vector{BitVector}, offspring::Vector{BitVector},
                           score_par::Vector{Float64}, score_off::Vector{Float64},
                           popsize::Int, elite::Int)
    elite = clamp(elite, 0, popsize)

    # 1) Keep the best `elite` parents
    parent_idx = sortperm(score_par, rev=true)
    elites = parents[parent_idx[1:elite]]

    # 2) Fill the remaining slots with best offspring
    remaining = popsize - elite
    off_idx = sortperm(score_off, rev=true)
    rest = offspring[off_idx[1:remaining]]

    return vcat(elites, rest)
end


# ----- entropy (for Task 2 plots) -----
"""
Population entropy over bit positions.
Returns a Float64. Higher means more diversity.
"""
function entropy_bits(pop::Vector{BitVector})
    popsize = length(pop)
    nbits = length(pop[1])
    H = 0.0
    @inbounds for i in 1:nbits
        ones_count = 0
        for ind in pop
            ones_count += ind[i] ? 1 : 0
        end
        p = ones_count / popsize
        if p > 0 && p < 1
            H -= p*log2(p) + (1-p)*log2(1-p)
        end
    end
    return H
end

# ----- core GA runner -----
"""
run_ga(nbits, fitness_fn; params)

fitness_fn(ind::BitVector)::Float64 returns raw fitness.
If objective=:min, GA will minimize raw fitness (by maximizing -fitness).
Returns:
- best_ind: BitVector
- best_raw: Float64  (best raw fitness according to objective)
- history: named tuple of arrays: max/mean/min (raw), entropy
"""
function run_ga(nbits::Int, fitness_fn::Function; params::GAParams=GAParams())
    rng = MersenneTwister(params.seed)

    pop = init_population(params.popsize, nbits, rng)

    max_hist = Float64[]
    mean_hist = Float64[]
    min_hist = Float64[]
    ent_hist = Float64[]

    best_ind = copy(pop[1])
    best_raw = fitness_fn(best_ind)  # raw fitness

    # helper to decide "better" in raw fitness space
    better(a, b) = params.objective === :max ? (a > b) : (a < b)
    
    @showprogress "Computing generations" for gen in 1:params.generations
        raw = Vector{Float64}(undef, length(pop))
        score = Vector{Float64}(undef, length(pop))

        for i in eachindex(pop)
            r = fitness_fn(pop[i])
            raw[i] = r
            score[i] = to_score(r, params.objective)
            if better(r, best_raw)
                best_raw = r
                best_ind = copy(pop[i])
            end
        end

        push!(max_hist, maximum(raw))
        push!(mean_hist, mean(raw))
        push!(min_hist, minimum(raw))
        push!(ent_hist, entropy_bits(pop))

        if gen == 1 || gen % params.log_every == 0 || gen == params.generations
            # For minimization, "best" is min_hist; for maximization, it's max_hist
            best_now = params.objective === :min ? min_hist[end] : max_hist[end]
            println("gen=$gen  best=$(round(best_now, digits=5))  mean=$(round(mean_hist[end], digits=5))  entropy=$(round(ent_hist[end], digits=3))")
            flush(stdout)
        end

        offspring = BitVector[]
        families  = Tuple{BitVector,BitVector,BitVector,BitVector}[]
        
        while length(offspring) < params.popsize
            p1 = tournament_select(pop, score, params.tour_k, rng)
            p2 = tournament_select(pop, score, params.tour_k, rng)
            
            c1, c2 = crossover(p1, p2, params.pc, rng)
            mutate!(c1, params.pm, rng)
            mutate!(c2, params.pm, rng)
            
            push!(offspring, c1)
            if length(offspring) < params.popsize
                push!(offspring, c2)
                push!(families, (p1, p2, c1, c2))
            end
        end

        raw_off = Vector{Float64}(undef, length(offspring))
        score_off = Vector{Float64}(undef, length(offspring))
        for i in eachindex(offspring)
            r = fitness_fn(offspring[i])
            raw_off[i] = r
            score_off[i] = to_score(r, params.objective)
        end

        if params.survivor_mode == :generational
            pop = survivors_generational(pop, offspring, score, score_off, params.popsize)
        elseif params.survivor_mode == :elitist
            pop = survivors_elitist(pop, offspring, score, score_off, params.popsize, params.elite)
        elseif params.survivor_mode == :crowding
            score_fn(ind) = to_score(fitness_fn(ind), params.objective)
            pop = survivors_generalized_crowding(
                families,
                score_fn,
                rng,
                params.crowding_alpha,
                params.crowding_scale
            )
        else
            error("Unknown survivor_mode: $(params.survivor_mode)")
        end
    end

    history = (; max_hist, mean_hist, min_hist, ent_hist)
    return best_ind, best_raw, history
end

# ----- local scripts -----
include("Crowding.jl")

end # module 
