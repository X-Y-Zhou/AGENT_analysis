using AGENT, Plots, Statistics, LinearAlgebra, Flux, Optim, FastGaussQuadrature

# Match the numerical execution used to validate these examples.
BLAS.set_num_threads(1)

# True parameters: [0.7080, 1.2732, 4.0171, 0.5903, 0.8317, 1.3848]
# Order: [sigma_on, sigma_off, rho, dm, lambda, dp].

# 1. Gaussian grid and explicit released AGENT network.
n = 7
nodes, weights = gausslegendre(n)
z1 = z2 = (nodes .+ 1) ./ 2
z3 = copy(z1)
w = weights ./ 2
W = vcat([vec(w * w') * w[i] for i in 1:n]...)
core = Chain(Dense(10, 160, tanh), Dense(160, 343))
# Preserve Float32 reconstruction for the released 3D model.
_, reconstruct = Flux.destructure(core)
trained_params = load_trained_params(joinpath(@__DIR__, "..", "parameters_trained", "params_trained3d.txt"))
rebuilt = reconstruct(trained_params)
network = Chain(rebuilt.layers..., x -> Flux.sigmoid.(x))
# Standardization is already folded into the released first-layer weights.
agent = AGENT.AGENTModel(3, z1, z2, z3, W, network,
    Float32.(trained_params), 1.0, 3)
function full_pgf(ps)
    input = vcat(AGENT.G_tele_delay.(ps[1], ps[2], ps[3], 1.0, z1), ps[4:end])
    vec(network(input))
end

# 2. Read counts and evaluate their empirical PGF on the model grid.
counts_path = joinpath(@__DIR__, "..", "dataset", "synthetic_data", "counts_example3d.txt")
U, S, P = read_counts3d(counts_path)
SSA_PGF = counts_to_pgf3d(U, S, P, agent.z1, agent.z2, agent.z3)

# 3. Explicit divergence and positive-parameter optimization in log space.
a = 1.0
function objective(log_ps, target)
    pred = full_pgf(exp.(log_ps))
    sum(W .* (pred.^(1+a) .- pred.^a .* target .* (1+1/a) .+ target/a))
end
opt = Optim.optimize(q -> objective(q, SSA_PGF), zeros(6), Optim.Options(show_trace=false, g_tol=1e-20, iterations=2000))
inferred_params = exp.(Optim.minimizer(opt))

# 4. Compare physical parameters with the annotated truth.
true_params = [0.7080, 1.2732, 4.0171, 0.5903, 0.8317, 1.3848]
relative_errors = abs.(inferred_params .- true_params) ./ abs.(true_params)
mean_relative_error = mean(relative_errors)
println("True parameters:     ", true_params)
println("Inferred parameters: ", inferred_params)
println("Mean relative error: ", 100 * mean_relative_error, "%")

# 5. Plot the fitted PGF against its empirical target.
inferred_PGF = full_pgf(inferred_params)
pgf_plot = scatter(SSA_PGF, inferred_PGF; label="AGENT", xlabel="SSA PGF", ylabel="Inferred PGF")
plot!(pgf_plot, [0, 1], [0, 1]; label="y = x", linestyle=:dash)
display(pgf_plot)
