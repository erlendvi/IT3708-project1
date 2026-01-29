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

params = GACore.GAParams(popsize=150, generations=200, pc=0.90, pm=0.01, tour_k=4,
                         survivor_mode=:elitist, elite=4, seed=42, objective=:min)

best_ind, best_rmse, hist = GACore.run_ga(nbits, feature_fitness; params=params)

println("Best RMSE: ", best_rmse)
println("Selected features: ", count(best_ind), " / ", nbits)

display(plot(hist.min_hist, label="min RMSE", xlabel="gen", ylabel="RMSE"))
display(plot(hist.ent_hist, label="entropy", xlabel="gen", ylabel="H"))