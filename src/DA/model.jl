export Model, SyntheticData


struct Model
    # Dimension of the state variable
    Nx::Int64

    # Dimension of the observation variable
    Ny::Int64

    Δtdyn::Float64

    Δtobs::Float64

    ϵx::InflationType

    ϵy::AdditiveInflation

    # Mean of the initial distribution
    m0::Array{Float64,1}

    # Covariance of the initial distribution
    C0::Array{Float64,2}

    # Number of steps to burn from the end of the spin up
    # to compute the metrics
    Tburn::Int64
    Tstep::Int64
    Tspinup::Int64

    # State-Space Model
    F::StateSpace
end


struct SyntheticData
	tt::Array{Float64,1}
	x0::Array{Float64,1}
	xt::Array{Float64,2}
	yt::Array{Float64,2}
end
