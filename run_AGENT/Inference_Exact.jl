# For the parameter inference method, please refer to https://github.com/edwardcao3026/pgf-inf

using Optim, Statistics, Distributions, Plots,StatsBase,DelimitedFiles,Random
using FastGaussQuadrature,Flux,DataFrames,CSV,HypergeometricFunctions
include("../utils.jl")

function Gz1z2(σon,σoff,ρ,τ,dm,z1,z2)
    τ=1
    u1=z1-1
    u2=z2-1
    x1=ρ*u1/dm
    x2=ρ*u2/dm
    α=σon/(σoff+σon)
    r=1+(σoff+σon)/dm-x1
    θ=sqrt(Complex(((σoff+σon)/dm-x1)^2+4*σon*x1/dm))
    v1=dm*(r+θ-1)/2
    v2=dm*(r-θ-1)/2
    G1=(v1*exp(-v2*τ)-v2*exp(-v1*τ))/(dm*θ)*pFq((σon/dm, ), ((σon+σoff)/dm, ), x2)+ρ*u1*(exp(-v2*τ)-exp(-v1*τ))/(dm*θ)*σon/(σoff+σon)*pFq((1+σon/dm, ), (1+(σoff+σon)/dm, ), x2)
    return real(G1)
end

function get_exact_gf(ps)
    σon,σoff,ρ,dm = ps
    z_mat = Gz1z2.(σon,σoff,ρ,1,dm, Z1, Z2)
    return vec(z_mat)
end

function int_dist(ps, SSA_PGF, a, W)
    dist = get_exact_gf(ps).^(1+a) .- get_exact_gf(ps).^a .* SSA_PGF .* (1+1/a) .+ SSA_PGF / a
    return sum(W .* dist)
end

n = 7
a = 0
b = 1
interval_X, weights = gausslegendre(n)
x1 = ((b - a) .* interval_X .+ b .+ a) ./ 2
w1 = weights * (b - a) / 2

a = 0
b = 1
interval_X, weights = gausslegendre(n)
x2 = ((b - a) .* interval_X .+ b .+ a) ./ 2
w2 = weights * (b - a) / 2

X = [(x1[i],x2[j]) for i = 1 :length(x1) for j = 1:length(x2)]
W = vec(w1*w2')

z1 = x1
z2 = x2

Z1 = repeat(z1, 1, length(z2))
Z2 = repeat(z2', length(z1), 1)

SSA_counts = readdlm("dataset/synthetic_data/counts_example2d.txt")
U_sample = Int.(SSA_counts[:,1])
S_sample = Int.(SSA_counts[:,2])
Sample_size = length(U_sample)

US_sample = [[U_sample[i],S_sample[i]] for i=1:Sample_size]
U_max = Int(maximum([n for (n, m) in US_sample]))
S_max = Int(maximum([m for (n, m) in US_sample]))

joint_prob_matrix = zeros(Float64, U_max+1, S_max+1)
for (m, n) in US_sample
    joint_prob_matrix[m+1, n+1] += 1
end

joint_prob_matrix /= length(US_sample)
SSA_PGF = vec(hist_gf2d(joint_prob_matrix,z1,z2))

init = [1,1,1,1]
init_ps = log.(init)
itera = 1000
results, time, _,_ = @timed Optim.optimize(ps->int_dist(exp.(ps),SSA_PGF,1.0,W),init_ps,
                                            Optim.Options(show_trace=true,g_tol=1e-11,iterations = itera)).minimizer

infer_params = exp.(results)

