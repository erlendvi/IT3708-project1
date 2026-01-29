# Crowding.jl

# --- distance ---
@inline function hamming(a::BitVector, b::BitVector)
    @assert length(a) == length(b)
    d = 0
    @inbounds for i in eachindex(a)
        d += (a[i] != b[i]) ? 1 : 0
    end
    return d
end

@inline function prefer_direct_match(p1::BitVector, p2::BitVector,
                                     c1::BitVector, c2::BitVector)
    d_direct = hamming(p1, c1) + hamming(p2, c2)
    d_cross  = hamming(p1, c2) + hamming(p2, c1)
    return d_direct <= d_cross
end

# --- crowding selection ---
@inline function crowding_select(
    p::BitVector, sp::Float64,
    c::BitVector, sc::Float64,
    rng::AbstractRNG,
    alpha::Float64,
    scale::Float64
)
    alpha = clamp(alpha, 0.0, 1.0)

    det = (sc >= sp) ? 1.0 : 0.0
    delta = (sc - sp) / max(scale, eps())
    prob = 1.0 / (1.0 + exp(-delta))

    p_replace = (1 - alpha) * det + alpha * prob
    return rand(rng) < p_replace ? c : p
end

# --- generalized crowding ---
function survivors_generalized_crowding(
    families,
    score_fn::Function,
    rng::AbstractRNG,
    alpha::Float64,
    scale::Float64
)
    newpop = BitVector[]

    for (p1, p2, c1, c2) in families
        sp1 = score_fn(p1); sp2 = score_fn(p2)
        sc1 = score_fn(c1); sc2 = score_fn(c2)

        if prefer_direct_match(p1, p2, c1, c2)
            push!(newpop, crowding_select(p1, sp1, c1, sc1, rng, alpha, scale))
            push!(newpop, crowding_select(p2, sp2, c2, sc2, rng, alpha, scale))
        else
            push!(newpop, crowding_select(p1, sp1, c2, sc2, rng, alpha, scale))
            push!(newpop, crowding_select(p2, sp2, c1, sc1, rng, alpha, scale))
        end
    end

    return newpop
end
