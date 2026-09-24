# The Tutorials can be found at https://github.com/marcjwilliams1/ApproxBayes.jl

using Statistics, Distributions, Plots,StatsBase,DelimitedFiles,Random
using Flux,DataFrames,CSV
using Catalyst.EnsembleAnalysis,Statistics
using DelaySSAToolkit,Catalyst,ApproxBayes
include("../utils.jl")

rn = @reaction_network begin
    σon,  Goff --> Gon
    σoff, Gon  --> Goff
    ρ, Gon --> Gon + U
    dm, S --> 0
end σon σoff ρ dm

jumpsys = convert(JumpSystem, rn; combinatoric_ratelaws=false)

u0 = [1, 0, 0, 0]
de_chan0 = [[]]
tf = 15.0
tspan = (0, tf)

function bayesdist(params, constant, SSA_counts)
    p = params
    τ = 1

    dprob = DiscreteProblem(u0, tspan, p)
    delay_trigger_affect! = function (integrator, rng)
        append!(integrator.de_chan[1], τ)
    end
    delay_trigger = Dict(3 => delay_trigger_affect!)
    delay_complete = Dict(1 => [3 => -1, 4 => 1])
    delay_interrupt = Dict()
    delayjumpset = DelayJumpSet(delay_trigger, delay_complete, delay_interrupt)

    djprob = DelayJumpProblem(
        jumpsys, dprob, DelayRejection(), delayjumpset, de_chan0; save_positions=(false, false)
    )

    interation = Int(1e4)
    ensprob = EnsembleProblem(djprob)
    ens = solve(ensprob, SSAStepper(), EnsembleThreads(); trajectories=interation,saveat=1)
    U_sample = componentwise_vectors_timepoint(ens, tf)[3]
    S_sample = componentwise_vectors_timepoint(ens, tf)[4]
    simdata = [U_sample;S_sample]
    ApproxBayes.ksdist(simdata, SSA_counts), 1
end

counts = readdlm("dataset/synthetic_data/counts_example2d.txt")
U_sample = Int.(counts[:,1])
S_sample = Int.(counts[:,2])
SSA_counts = [U_sample;S_sample]
ϵ = 0.2

setup = ABCRejection(bayesdist, # simulation function
    4, # number of parameters
    ϵ, # target 
    ApproxBayes.Prior([Uniform(0, 4.0), Uniform(0, 4.0),Uniform(0.0, 50.0),Uniform(0, 2),]); # Prior for each of the parameters
    maxiterations = 10^6 # Maximum number of iterations before the algorithm terminates
)

results, time, _,_ = @timed runabc(setup, SSA_counts,progress=false)




