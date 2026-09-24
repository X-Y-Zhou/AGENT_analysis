# Inference with NNCME
# For methods of parameter inference, please refer to https://www.nature.com/articles/s41467-021-22919-1

using Flux,Zygote,NLsolve,LinearMaps,IterativeSolvers,LinearAlgebra,Statistics,Random
using Zygote: @adjoint
using DelimitedFiles

function resize_matrix(mat::AbstractMatrix, N1::Int, N2::Int)
    M1, M2 = size(mat)
    new_mat = zeros(eltype(mat), N1, N2)
    r1 = min(N1, M1)
    r2 = min(N2, M2)
    new_mat[1:r1, 1:r2] .= mat[1:r1, 1:r2]
    return new_mat
end

Nn = 60
Nm = 60 
NS = Nn * Nm
TOT = 2 * NS
τ = 1.0

counts = readdlm("dataset/synthetic_data/counts_example2d.txt")
U_sample = Int.(counts[:,1])
S_sample = Int.(counts[:,2])
Sample_size = length(U_sample)

# Convert counts data to joint distribution
US_sample = [[U_sample[i],S_sample[i]] for i=1:Sample_size]
U_max = Int(maximum([n for (n, m) in US_sample]))
S_max = Int(maximum([m for (n, m) in US_sample]))

joint_prob_matrix = zeros(Float64, U_max+1, S_max+1)
for (m, n) in US_sample
    joint_prob_matrix[m+1, n+1] += 1
end

joint_prob_matrix /= length(US_sample)
P_data = resize_matrix(joint_prob_matrix,Nn,Nm)
P_data ./= sum(P_data)

softplus_stable(x) = log1p(exp(-abs(x))) + max(x, 0.0)
inv_softplus(y) = y > 20 ? y : log(exp(y) - 1.0)

@adjoint nlsolve(f, x0; kwargs...) =
    let result = nlsolve(f, x0; kwargs...)
        result, function(vresult)
            dx = vresult[].zero
            x  = result.zero

            _, back_x = Zygote.pullback(f, x)
            JT(df) = back_x(df)[1]

            L = LinearMap(JT, length(x0))
            df = gmres(L, -dx)

            _, back_f = Zygote.pullback(f -> f(x), f)
            return (back_f(df)[1], nothing, nothing)
        end
    end

function unpack_prob(x::AbstractVector)
    P0 = reshape(view(x, 1:NS), Nn, Nm)
    P1 = reshape(view(x, NS+1:2NS), Nn, Nm)
    return P0, P1
end

function pack_prob(P0::AbstractMatrix, P1::AbstractMatrix)
    return vcat(vec(P0), vec(P1))
end

model = Chain(
    Dense(NS, 5, tanh),
    Dense(5, NS),
    x -> softplus_stable.(x) .+ 1e-8
)

p_net0, re = Flux.destructure(model)
σ_on0,σ_off0,ρ0,d_m0 = [1.,1.,1.,1.]

θ = vcat([
    inv_softplus(σ_on0),
    inv_softplus(σ_off0),
    inv_softplus(ρ0),
    inv_softplus(d_m0)
], p_net0)

ps = Flux.params(θ)

function unpack_all_params(θ)
    σ_on  = softplus_stable(θ[1]) + 1e-8
    σ_off = softplus_stable(θ[2]) + 1e-8
    ρ     = softplus_stable(θ[3]) + 1e-8
    d_m   = softplus_stable(θ[4]) + 1e-8
    p_net = θ[5:end]
    return σ_on, σ_off, ρ, d_m, p_net
end

function initial_guess(P_data, σ_on, σ_off)
    π_on = σ_on / (σ_on + σ_off)
    P1 = π_on .* P_data
    P0 = (1.0 - π_on) .* P_data
    x0 = pack_prob(P0, P1)
    return x0
end

σ_on_init, σ_off_init, ρ_init, d_m_init, _ = unpack_all_params(θ)
x_init = initial_guess(P_data, σ_on_init, σ_off_init)

function effective_rate(Ptot::AbstractMatrix, p_net)
    λ = re(p_net)(vec(Ptot))
    return reshape(λ, Nn, Nm)
end

function steady_residual(x::AbstractVector, θ)
    σ_on, σ_off, ρ, d_m, p_net = unpack_all_params(θ)

    P0, P1 = unpack_prob(x)
    Ptot = P0 .+ P1
    λ = effective_rate(Ptot, p_net)

    function one_residual(i, j, layer)
        n = i - 1
        m = j - 1

        r = 0.0
        P  = layer == 0 ? P0 : P1
        Po = layer == 0 ? P1 : P0

        if layer == 0
            r += σ_off * P1[i, j] - σ_on * P0[i, j]
        else
            r += σ_on * P0[i, j] - σ_off * P1[i, j]
        end

        if layer == 1
            if i >= 2
                r += ρ * P1[i-1, j]
            end
            if i <= Nn-1
                r -= ρ * P1[i, j]
            end
        end

        if j <= Nm-1
            r += d_m * (m + 1) * P[i, j+1]
        end
        if m >= 1
            r -= d_m * m * P[i, j]
        end

        if i <= Nn-1 && j >= 2
            r += λ[i+1, j-1] * P[i+1, j-1]
        end
        if i >= 2 && j <= Nm-1
            r -= λ[i, j] * P[i, j]
        end

        return r
    end

    R0 = [one_residual(i, j, 0) for i in 1:Nn, j in 1:Nm]
    R1 = [one_residual(i, j, 1) for i in 1:Nn, j in 1:Nm]

    v0 = vec(R0)
    v1 = vec(R1)

    return vcat([sum(Ptot) - 1.0], v0[2:end], v1)
end

function steady_solution(θ; x0=nothing)
    σ_on, σ_off, _, _, _ = unpack_all_params(θ)

    if x0 === nothing
        x0 = initial_guess(P_data, σ_on, σ_off)
    end

    sol = nlsolve(x -> steady_residual(x, θ),x0)
    return sol.zero
end

function loss_func(θ; x0=nothing, λreg=1e-8)
    xstar = steady_solution(θ; x0=x0)
    P0, P1 = unpack_prob(xstar)
    Ppred = P0 .+ P1

    fit_loss = Flux.mse(Ppred, P_data)

    _, _, _, _, p_net = unpack_all_params(θ)
    reg_loss = λreg * sum(abs2, p_net)

    return fit_loss + reg_loss
end

opt = ADAM(1e-3)
epochs = 10

# warm start for nlsolve
current_x0 = copy(x_init)

loss_history = Float64[]
σon_hist  = Float64[]
σoff_hist = Float64[]
ρ_hist    = Float64[]
dm_hist   = Float64[]

@time loss_func(θ)
@time gradient(() -> loss_func(θ), ps)

for epoch in 1:epochs
    grads = gradient(() -> loss_func(θ), ps)
    Flux.update!(opt, ps, grads)

    xstar = steady_solution(θ; x0=current_x0)
    current_x0 .= xstar

    P0_star, P1_star = unpack_prob(xstar)
    Ppred = P0_star .+ P1_star
    cur_loss = Flux.mse(Ppred, P_data)

    σ_on_est, σ_off_est, ρ_est, d_m_est, _ = unpack_all_params(θ)

    push!(loss_history, cur_loss)
    push!(σon_hist,  σ_on_est)
    push!(σoff_hist, σ_off_est)
    push!(ρ_hist,    ρ_est)
    push!(dm_hist,   d_m_est)

    println(
        "epoch = ", epoch,
        ", loss = ", cur_loss,
        ", σ_on = ", σ_on_est,
        ", σ_off = ", σ_off_est,
        ", ρ = ", ρ_est,
        ", d_m = ", d_m_est,
        ", τ = ", τ
    )
end

xstar = steady_solution(θ; x0=current_x0)
P0_star, P1_star = unpack_prob(xstar)
Ppred = P0_star .+ P1_star

σ_on_est, σ_off_est, ρ_est, d_m_est, p_net_est = unpack_all_params(θ)


