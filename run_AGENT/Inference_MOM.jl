# Inference with MOM

using Optim, CSV, DataFrames,LinearAlgebra, Distributions, SparseArrays
using Statistics,DelimitedFiles,StatsBase,Distributed
include("../utils.jl")

function observed_moments(U_sample,S_sample)
    mean_U = mean(U_sample)
    mean_S = mean(S_sample)
    var_U = var(U_sample)
    var_S = var(S_sample)
    return [mean_U,mean_S,var_U,var_S]
end

# FSP 模拟函数
function simulate_moments(ps)
    σon,σoff,ρ,d = ps
    τ = 1
    
    # mean and variance
    mean_U=ρ*σon*τ/(σoff+σon)
    mean_S=ρ*σon/((σoff+σon)*d)

    var_U=ρ*σon*τ/(σoff+σon)+2*ρ^2*σon*σoff*(-1+exp(-(σoff+σon)*τ)+(σoff+σon)*τ)/((σoff+σon)^4)
    var_S=ρ*σon/(d*(σoff+σon))+ρ^2*σon*σoff/(d*(σoff+σon)^2*(d+σoff+σon))

    simulated_data=[mean_U,mean_S,var_U,var_S]
    return simulated_data
end

function objective_function(ps,SSA_counts)
    simulated_data = simulate_moments(ps)

    U_sample = Int.(SSA_counts[:,1])
    S_sample = Int.(SSA_counts[:,2])
    observed_data = observed_moments(U_sample,S_sample)

    mse = sum(((observed_data .- simulated_data)./observed_data).^2) / length(observed_data)
    return mse
end

SSA_counts = readdlm("dataset/synthetic_data/counts_example2d.txt")
init = [1,1,1,1]
init_ps = log.(init)
itera = 1000
results, time, _,_ = @timed Optim.optimize(ps -> objective_function(exp.(ps), SSA_counts),
                                            init_ps,Optim.Options(show_trace=true,g_tol=1e-11,iterations = itera)).minimizer

infer_params = exp.(results)

