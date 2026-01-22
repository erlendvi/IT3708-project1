using CSV, DataFrames
using Plots

include("general_ga.jl")  # defines module GACore

const CAPACITY = 280_785

df = CSV.read("knapsack/knapPI_12_500_1000_82.csv", DataFrame)
profits = Float64.(df[!, :p])
weights = Float64.(df[!, :w])
nbits = length(profits)  # 500

function knapsack_fitness(ind::BitVector)
    total_profit = sum(@view profits[ind])
    total_weight = sum(@view weights[ind])

    if total_weight <= CAPACITY
        return total_profit
    else
        alpha = 15.0
        return total_profit - alpha * (total_weight - CAPACITY)
    end
end

params = GACore.GAParams(
    popsize=300,
    generations=10000,
    pc=0.90,
    pm=0.002,
    tour_k=3,
    survivor_mode=:elitist,
    elite=2,
    seed=42,
    objective=:max
)

best_ind, best_fit, hist = GACore.run_ga(nbits, knapsack_fitness; params=params)

best_profit = sum(@view profits[best_ind])
best_weight = sum(@view weights[best_ind])

println("Returned best individual:")
println("  penalized fitness = $(round(best_fit, digits=2))")
println("  true profit       = $(round(best_profit, digits=2))")
println("  total weight      = $(round(best_weight, digits=2)) / $(CAPACITY)")
println("  selected items    = $(count(best_ind))")

plt = plot(hist.max_hist, label="max fitness", xlabel="generation", ylabel="fitness")
plot!(plt, hist.mean_hist, label="mean fitness")
plot!(plt, hist.min_hist, label="min fitness")
display(plt)

display(plot(hist.ent_hist, label="entropy", xlabel="generation", ylabel="H"))
