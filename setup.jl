using Pkg
Pkg.activate(@__DIR__)
package=normpath(joinpath(@__DIR__,"..","AGENT.jl"))
isdir(package) || error("Place AGENT.jl beside AGENT_analysis, then run setup.jl again")
cd(@__DIR__) do
    Pkg.develop(path="../AGENT.jl")
    Pkg.instantiate()
end
