using Flux, Optim, Statistics, Random, LinearAlgebra
using DelimitedFiles, FastGaussQuadrature, CSV, DataFrames, TOML

# Random-initialization training; never loads or overwrites released weights.
# Parameter truth order: [sigma_on, sigma_off, rho, dm].
# n_z = 7; SSA targets were generated with n_t = 10^4 trajectories.
const NODES_PER_AXIS = 7
const ADAM_STEPS = 2000
const LBFGS_STEPS = 12000
const HIDDEN_WIDTH = 80
const INPUT_SIZE = 8
const OUTPUT_SIZE = 49
const TRAINING_GROUPS = 2807

function training_data()
    truth = readdlm(joinpath(@__DIR__, "ps_for_train.txt"), Float64)
    reduced = readdlm(joinpath(@__DIR__, "matrix_Gz1.csv"), Float64)
    target = readdlm(joinpath(@__DIR__, "matrix_Gz1z2.csv"), Float64)
    @assert size(truth) == (TRAINING_GROUPS, 4)
    @assert size(reduced) == (NODES_PER_AXIS, TRAINING_GROUPS)
    @assert size(target) == (OUTPUT_SIZE, TRAINING_GROUPS)
    input = vcat(reduced, permutedims(truth[:, 4:end]))
    @assert all(isfinite, input) && all(isfinite, target)
    @assert all(>(0), truth) && minimum(target) >= 0 && maximum(target) <= 1 + 1e-12
    _, weights = gausslegendre(NODES_PER_AXIS)
    w = weights ./ 2
    W = vec(w * w')
    return input, target, W
end

# Keep standardization in training, then fold it into exported first-layer weights.
# The inference scripts use the raw reduced PGF and kinetic parameters.
function initialize(input; seed=20260916)
    Random.seed!(seed)
    model = Flux.f64(Chain(Dense(INPUT_SIZE, HIDDEN_WIDTH, tanh),
                          Dense(HIDDEN_WIDTH, OUTPUT_SIZE)))
    parameters, reconstruct = Flux.destructure(model)
    mu = vec(mean(input; dims=2))
    scale = max.(vec(std(input; dims=2)), 1e-12)
    return parameters, reconstruct, mu, scale
end

learning_rate(t) = 0.03 * 0.1^((t - 1) / 1999)

function train(; output=joinpath(@__DIR__, "outputs"), seed=20260916,
                 adam_steps=ADAM_STEPS, lbfgs_steps=LBFGS_STEPS)
    1 <= adam_steps <= ADAM_STEPS || error("Invalid Adam update count")
    1 <= lbfgs_steps <= LBFGS_STEPS || error("Invalid L-BFGS update count")
    ispath(output) && error("Output path already exists: $output")
    input, target, W = training_data()
    parameters, reconstruct, mu, scale = initialize(input; seed=seed)
    X = (input .- mu) ./ scale
    loss(p) = sum(W .* (Flux.sigmoid.(reconstruct(p)(X)) .- target).^2) / size(X, 2)
    function fg!(F, G, p)
        if G === nothing
            return F === nothing ? nothing : loss(p)
        end
        value, back = Flux.pullback(loss, p)
        G .= back(one(value))[1]
        return F === nothing ? nothing : value
    end
    mkpath(output)
    trace = DataFrame(stage=String[], update=Int[], loss=Float64[], learning_rate=Float64[])
    # One persistent Adam state across the complete exponentially decaying stage.
    optimizer = Flux.Adam(learning_rate(1))
    gradient = similar(parameters)
    seconds = @elapsed begin
        for t in 1:adam_steps
            optimizer.eta = learning_rate(t)
            fg!(true, gradient, parameters)
            Flux.Optimise.update!(optimizer, parameters, gradient)
            value = loss(parameters)
            isfinite(value) || error("Nonfinite Adam loss at update $t")
            push!(trace, ("Adam", t, value, learning_rate(t)))
            t % 100 == 0 && println("Adam $t/$adam_steps: $value")
        end
        callback = state -> begin
            if state.iteration > 0
                push!(trace, ("L-BFGS", state.iteration, state.value, NaN))
                state.iteration % 100 == 0 && println("L-BFGS $(state.iteration)/$lbfgs_steps: $(state.value)")
            end
            false
        end
        # Zero tolerances avoid the library's ordinary early-convergence thresholds.
        # iterations is an upper bound: record actual updates if line search terminates.
        result = Optim.optimize(Optim.only_fg!(fg!), parameters, Optim.LBFGS(m=30),
            Optim.Options(iterations=lbfgs_steps, g_tol=0.0, x_tol=0.0, f_tol=0.0,
                          callback=callback, show_trace=false))
        parameters = copy(Optim.minimizer(result))
    end
    # Export weights in exactly the same Flux.destructure order as inference.
    core = reconstruct(parameters)
    folded_W = core[1].weight ./ permutedims(scale)
    folded_b = core[1].bias .- folded_W * mu
    exported = Chain(Dense(folded_W, folded_b, tanh), core[2])
    folded_parameters, _ = Flux.destructure(exported)
    @assert isapprox(exported(input[:, 1:5]), core(X[:, 1:5]); rtol=1e-10, atol=1e-10)
    CSV.write(joinpath(output, "params_trained2d.txt"), DataFrame(params=folded_parameters))
    CSV.write(joinpath(output, "training_trace.csv"), trace)
    actual_lbfgs = count(==("L-BFGS"), trace.stage)
    open(joinpath(output, "training_summary.toml"), "w") do io
        TOML.print(io, Dict("seed"=>seed, "training_groups"=>TRAINING_GROUPS,
            "adam_updates"=>adam_steps, "lbfgs_requested"=>lbfgs_steps,
            "lbfgs_updates"=>actual_lbfgs, "lbfgs_history"=>30,
            "training_seconds"=>seconds, "final_loss"=>loss(parameters),
            "standardization_folded"=>true, "precision"=>"Float64"))
    end
    actual_lbfgs == lbfgs_steps || @warn "L-BFGS stopped early" requested=lbfgs_steps actual=actual_lbfgs
    return (parameters=folded_parameters, trace=trace, final_loss=loss(parameters))
end

# Command-line entry point. In Julia/VSCode: include this file, then call train().
if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
