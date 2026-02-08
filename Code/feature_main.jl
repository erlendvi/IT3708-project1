using DelimitedFiles
using DataFrames
using MLJ
using MLJLinearModels
using Random
using Plots

include("general_ga.jl")
include("../feature_selection/LinReg.jl")

# Load data
base = @__DIR__
data_path = joinpath(base, "..", "feature_selection", "dataset.txt")
data = readdlm(data_path, ',', Float64)

@assert size(data, 2) == 102
@assert size(data, 1) == 1994

Xmat = data[:, 1:101]
y    = vec(data[:, 102])

# Convert to MLJ-friendly table
Xdf = DataFrame(Xmat, :auto)   # columns x1, x2, ...
nbits = ncol(Xdf)              # 101

model = MLJLinearModels.LinearRegressor()
rng_lr = MersenneTwister(123)

# Fitness wrapper (RMSE to minimize)
function feature_fitness(ind::BitVector)
    if count(ind) == 0
        return Inf
    end
    # LinReg.get_columns returns Matrix; convert to DataFrame for MLJ
    Xsub_mat = get_columns(Xmat, ind)
    Xsub_df  = DataFrame(Xsub_mat, names(Xdf)[findall(ind)])  # preserve names
    return get_fitness(model, Xsub_df, y; rng=rng_lr)
end

baseline_rmse = get_fitness(model, Xdf, y; rng=rng_lr)
println("Baseline RMSE: ", baseline_rmse)

# --- GA config fra kommandolinje/ENV ---
survivor_mode = Symbol(get(ENV, "GA_SURVIVOR",
    length(ARGS) >= 1 ? ARGS[1] : "elitist"
))


params = GACore.GAParams(
    popsize=100, generations=100, pc=0.95, pm=0.005, tour_k=4,
    survivor_mode=survivor_mode, seed=42, elite=4, objective=:min,
    crowding_alpha=1.0, crowding_scale=1.0
)


best_ind, best_rmse, worst_ind, worst_rmse, hist = GACore.run_ga(nbits, feature_fitness; params=params)


println("Best RMSE:  ", best_rmse)
println("Worst RMSE: ", worst_rmse)
println("Selected features (best): ", count(best_ind), " / ", nbits)
println("Selected features (worst): ", count(worst_ind), " / ", nbits)

p1 = plot(hist.min_hist, label="min RMSE", xlabel="Generations", ylabel="RMSE")
plot!(p1, hist.max_hist, label="max RMSE")
plot!(p1, hist.mean_hist, label="mean RMSE")


# Baseline (all features)
#hline!(p1, [baseline_rmse],
#    label="baseline (all features)",
#    linestyle=:dash,
#    color=:black
#)
plot!(p1, legend = :outertopright)

savefig(p1, "sweep4_rmse.png")

p2 = plot(hist.ent_hist, label="entropy", xlabel="Generations", ylabel="H")
savefig(p2, "sweep1_entropy.png")