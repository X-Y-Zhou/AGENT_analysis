# Inference with Finite State Projection

using DelimitedFiles,Optim,Flux
using SparseArrays, StatsBase, LinearAlgebra
using Distributions, SparseArrays,DifferentialEquations
include("../utils.jl")

function CME_aux!(du,u,p,t)
    ρ,σon,σoff,N=p
    # Define transition matrix for \bar{P}'s CME
    B = zeros(2*N,2*N)
    B[1:N,1:N] = - spdiagm(0 => σon*ones(N)) 
    B[1:N,N+1:2*N] = spdiagm(0 => σoff*ones(N)) 
    B[N+1:2*N,1:N] = spdiagm(0 => σon*ones(N)) 
    B[N+1:2*N,N+1:2*N] = - spdiagm(0 => σoff*ones(N))  - spdiagm(0 => ρ*vcat(ones(N-1),0)) + spdiagm(-1 => ρ*ones(N-1))
    du[1:end] = B*u

    return u
end

function CME_maturemar!(p)
    ρ,σon,σoff,d,N=p
    # Define transition matrix for \bar{P}'s CME
    C = zeros(2*N,2*N)
    C[1:N,1:N] = - spdiagm(0 => σon*ones(N)) - spdiagm(0 => d*collect(0:N-1)) + spdiagm(1 => d*collect(1:N-1))
    C[1:N,N+1:2*N] = spdiagm(0 => σoff*ones(N)) 
    C[N+1:2*N,1:N] = spdiagm(0 => σon*ones(N)) 
    C[N+1:2*N,N+1:2*N] = - spdiagm(0 => σoff*ones(N))  - spdiagm(0 => ρ*vcat(ones(N-1),0)) + spdiagm(-1 => ρ*ones(N-1))- spdiagm(0 => d*collect(0:N-1)) + spdiagm(1 => d*collect(1:N-1))
    C[1,:].=1 
    pp=C\[1;zeros(2*N-1)]
    pm1=pp[N+1:2*N]

    return pm1
end

function CME_main!(p1)
    ρ,σon,σoff,d,N1,N2,pn0,pn1,pm1=p1
        # Define transition matrix for Q's CME
        M=zeros(2*N1*N2,2*N1*N2)
        for i =1:N2 
            M[(i-1)*N1+1:i*N1 ,(i-1)*N1+1:i*N1]=-spdiagm(0=> σon*ones(N1))-spdiagm(d*(i-1)*ones(N1))
            M[(i-1)*N1+1:i*N1 ,(i-1)*N1+1+N1*N2:i*N1+N1*N2]=spdiagm(0=>σoff*ones(N1))
            M[(i-1)*N1+1+N1*N2:i*N1+N1*N2 ,(i-1)*N1+1:i*N1]=spdiagm(0=>σon*ones(N1))
            M[(i-1)*N1+1+N1*N2:i*N1+N1*N2 ,(i-1)*N1+1+N1*N2:i*N1+N1*N2]=-spdiagm(0=>σoff*ones(N1))-spdiagm(d*(i-1)*ones(N1))-spdiagm(0=>ρ*vcat(ones(N1-1),0))+spdiagm(-1=>ρ*ones(N1-1))
        end
        
        for i =1:N2-1 
            M[(i-1)*N1+1:i*N1 , i*N1+1:(i+1)*N1]=spdiagm(d*i*ones(N1))
            M[(i-1)*N1+1+N1*N2:i*N1+N1*N2 , i*N1+1+N1*N2:(i+1)*N1+N1*N2]=spdiagm(d*i*ones(N1))
        end

    D=-ρ * vcat(reshape(pn0*vcat(0,pm1[1:end-1])'-vcat(0,pn0[1:end-1])*pm1',(N1*N2,1)),reshape(pn1*vcat(0,pm1[1:end-1])'-vcat(0,pn1[1:end-1])*pm1',(N1*N2,1))).+[1;zeros(2*N1*N2-1)]
    M[1,:].=1 
    solution=M\D

    p_0 = solution[1:N1*N2]
    p_1 = solution[N1*N2+1:2*N1*N2]
    u_intk = reshape(p_0+p_1,(N1,N2))

    U_intk = max.(0.0,u_intk)
    U_intk = U_intk/sum(U_intk)

    return U_intk
end

function delaysolG1(pt,N1,N2)
    σon,σoff,ρ,τ,d = pt
    # Define transition matrix without delay effect terms

    # Obtain the edge probability of Nuclear mRNA at τ 
    pn = (ρ,σon,σoff,N1)
    u_aux = zeros(2*N1)
    # Initial condition -- gene state ON (always true)
    u_aux[N1+1] = 1.
    tspan = (0.0, τ)
    prob2 = ODEProblem(CME_aux!, u_aux, tspan, pn)
    sol2 = solve(prob2, Tsit5(), saveat=0.2)
    pn0 = sol2.u[end][1:N1]
    pn1 = sol2.u[end][N1+1:end]
  
    # Obtain the edge probability of Cytoplasmic mRNA at steady state
    pm = (ρ,σon,σoff,d,N2)
    pm1=CME_maturemar!(pm)

    # Obtain joint probability distribution
    p1=(ρ,σon,σoff,d,N1,N2,pn0,pn1,pm1)
    u_int=CME_main!(p1)
    return u_int
end

function resize_matrix(mat::AbstractMatrix, N1::Int, N2::Int)
    M1, M2 = size(mat)
    new_mat = zeros(eltype(mat), N1, N2)
    r1 = min(N1, M1)
    r2 = min(N2, M2)
    new_mat[1:r1, 1:r2] .= mat[1:r1, 1:r2]
    return new_mat
end

function int_dist(ps, hist_data)
    σon,σoff,ρ,d = ps
    τ = 1
    pt = [σon,σoff,ρ,τ,d]
    N_max,M_max = [60,60]
    hist_data = resize_matrix(hist_data,N_max,M_max)
    output = delaysolG1(pt,N_max, M_max)
    return -sum(hist_data.*log.(output.+1e-12))
end

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
hist_data = joint_prob_matrix

init = [1,1,1,1]
init_ps = log.(init)
itera = 1000
results, time, _,_ = @timed Optim.optimize(ps->int_dist(exp.(ps),hist_data),init_ps,
Optim.Options(show_trace=true,g_tol=1e-11,iterations = itera)).minimizer

infer_params = exp.(results)



