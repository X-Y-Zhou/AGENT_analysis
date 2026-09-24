using AGENT, Plots, Statistics, LinearAlgebra, Flux, Optim, FastGaussQuadrature

# Match the numerical execution used to validate these examples.
BLAS.set_num_threads(1)

# True parameters: [1.0946, 0.1216, 2.5215, 0.4321, 1.8173, 1.2662, 1.4658, 0.0469, 2.6785, 0.3962, 2.4908, 1.1572]
# Order: [sigma_on, sigma_off, rho, dm, lambda, dp] for gene 1, then gene 2.

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
counts_path = joinpath(@__DIR__, "..", "dataset", "synthetic_data", "counts_example_toggle.txt")
U1, S1, P1, U2, S2, P2 = read_counts_toggle(counts_path)
SSA_PGF1, SSA_PGF2 = counts_to_toggle_pgfs(agent, U1, S1, P1, U2, S2, P2)

# 3. Explicit divergence and positive-parameter optimization in log space.
a = 1.0
function objective(log_ps, target)
    pred = full_pgf(exp.(log_ps))
    sum(W .* (pred.^(1+a) .- pred.^a .* target .* (1+1/a) .+ target/a))
end
opt1 = Optim.optimize(q -> objective(q, SSA_PGF1), zeros(6), Optim.Options(show_trace=false, g_tol=1e-20, iterations=2000))
opt2 = Optim.optimize(q -> objective(q, SSA_PGF2), zeros(6), Optim.Options(show_trace=false, g_tol=1e-20, iterations=2000))
equal1 = exp.(Optim.minimizer(opt1))
equal2 = exp.(Optim.minimizer(opt2))
soff = convert_LMA_toggle(vcat(equal1, equal2))
inferred_params = [equal1[1]; soff[1]; equal1[3:end]; equal2[1]; soff[2]; equal2[3:end]]
result = (inferred_params_equal1=equal1, inferred_params_equal2=equal2,
          inferred_PGF1=full_pgf(equal1), inferred_PGF2=full_pgf(equal2))

# 4. Compare physical parameters with the annotated truth.
true_params = [1.0946, 0.1216, 2.5215, 0.4321, 1.8173, 1.2662, 1.4658, 0.0469, 2.6785, 0.3962, 2.4908, 1.1572]
relative_errors = abs.(inferred_params .- true_params) ./ abs.(true_params)
mean_relative_error = mean(relative_errors)
println("True parameters:     ", true_params)
println("Inferred parameters: ", inferred_params)
println("Mean relative error: ", 100 * mean_relative_error, "%")
# The physical feedback/off rates above have undergone the LMA conversion.
println("Equivalent gene 1 parameters: ", result.inferred_params_equal1)
println("Equivalent gene 2 parameters: ", result.inferred_params_equal2)

# 5. Plot each marginal PGF against its empirical target.
p1 = scatter(SSA_PGF1, result.inferred_PGF1; label="AGENT", xlabel="SSA PGF", ylabel="Inferred PGF", title="Gene 1")
plot!(p1, [0, 1], [0, 1]; label="y = x", linestyle=:dash)
p2 = scatter(SSA_PGF2, result.inferred_PGF2; label="AGENT", xlabel="SSA PGF", ylabel="Inferred PGF", title="Gene 2")
plot!(p2, [0, 1], [0, 1]; label="y = x", linestyle=:dash)
pgf_plot = plot(p1, p2; layout=(1, 2), size=(900, 400), margin=5 * Plots.PlotMeasures.mm)
display(pgf_plot)
