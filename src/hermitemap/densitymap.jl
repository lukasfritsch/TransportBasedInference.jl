export
    HermiteMap,
    totalordermap,
    optimize!,
    setcoeff!,
    ncoeff,
    grad_xd,
    grad_coeff,
    grad_coeff_grad_xd,
    objective_KL!,
    objective_KL,
    logdeterminant

import Base: @propagate_inbounds


"""
    HermiteMap(m::Int64, d::Int64; α::Float64 = αreg, b::String="CstProHermiteBasis")

Constructs a Hermite map for density estimation using an identity map for transport maps.

# Arguments
- `m::Int64`: The order of the Hermite polynomial basis.
- `d::Int64`: The number of dimensions for the input.
- `α::Float64`: Regularization parameter (default is `αreg=1.0e-6`).
- `b::String`: The type of basis to use. Options include:
    - `"ProHermiteBasis"`
    - `"PhyHermiteBasis"`
    - `"CstProHermiteBasis"`
    - `"CstPhyHermiteBasis"`
    - `"CstLinProHermiteBasis"`
    - `"CstLinPhyHermiteBasis"`
  Default is `"CstProHermiteBasis"`.

# Returns
- A `HermiteMap` object initialized with the specified parameters.

"""
function HermiteMap(m::Int64, d::Int64; α::Float64 = αreg, b::String="CstProHermiteBasis")

    # Initialize the identity transformation
    L = deepcopy(LinearTransform(d))

    # Select polynomial basis
    if b ∈ ["ProHermiteBasis"; "PhyHermiteBasis";
        "CstProHermiteBasis"; "CstPhyHermiteBasis";
        "CstLinProHermiteBasis"; "CstLinPhyHermiteBasis"]
        B = eval(Symbol(b))(m)
    else
        error("The basis "*b*" is not defined.")
    end

    # Initialize the map components
    Nψ = 1
    coeff = zeros(Nψ)
    C = HermiteMapComponent[]
    idx = zeros(Int, Nψ, d)

    @inbounds for i=1:d
        MultiB = MultiBasis(B, i)
        vidx = idx[:,1:i]
        push!(C, HermiteMapComponent(IntegratedFunction(ExpandedFunction(MultiB, vidx, coeff)); α = α))
    end

    return HermiteMap(m, d, L, C)
end

"""
    totalordermap(order::Int64, d::Int64; withconstant::Bool = false, b::String = "CstProHermiteBasis") -> HermiteMap

Construct a total order Hermite map of a given order and dimension.

# Arguments
- `order::Int64`: The order of the Hermite map.
- `d::Int64`: The dimension of the Hermite map.
- `withconstant::Bool`: Whether to include a constant term in the map (default: `false`).
- `b::String`: The basis type to use. Options are `"CstProHermiteBasis"`, `"CstPhyHermiteBasis"`, `"CstLinProHermiteBasis"`, and `"CstLinPhyHermiteBasis"` (default: `"CstProHermiteBasis"`).

# Returns
- `HermiteMap`: The constructed Hermite map.

"""
function totalordermap(order::Int64, d::Int64; withconstant::Bool = false, b::String="CstProHermiteBasis")
    L = deepcopy(LinearTransform(d))

    C = HermiteMapComponent[]
    @inbounds for i=1:d
        push!(C, totalordermapcomponent(i, order; withconstant = withconstant, b = b))
    end

    if b ∈ ["ProHermiteBasis", "PhyHermiteBasis"]
        m = order+1
    elseif b ∈ ["CstProHermiteBasis", "CstPhyHermiteBasis"]
        m = order+2
    elseif b ∈ ["CstLinProHermiteBasis", "CstLinPhyHermiteBasis"]
        m = order+3
    else
        error("Undefined basis")
    end

    return HermiteMap(m, d, L, C)
end

"""
    ncoeff(M::HermiteMap) -> Int64
"""
function ncoeff(M::HermiteMap)
    return sum([ncoeff(component) for component in M.C])
end

# Set the coefficients in all map components.
function setcoeff!(M::HermiteMap, a::Vector{Float64})
    counter = 1
    for component in M.C
        setcoeff!(component, a[counter:counter+component.Nψ-1])
        counter += component.Nψ
    end
end

"""
    grad_xd(M::HermiteMap, X::Matrix{Float64}) -> Matrix{Float64}

Compute the gradient of the Hermite map `M` with respect to the coordiantes x evaluated at `X`.

# Arguments
- `M::HermiteMap`: The Hermite map object containing components.
- `X::Matrix{Float64}`: The input matrix.

# Returns
- `Matrix{Float64}`: An array of gradients for each component in the Hermite map.

"""
function grad_xd(M::HermiteMap, X::Matrix{Float64})
    ∇M = zeros(size(X))

    for (i, component) in enumerate(M.C)
        ∇M[i,:] += grad_xd(component.I, X[1:i,:])
    end

    return ∇M
end

"""
    grad_coeff(M::HermiteMap, X::Matrix{Float64}) -> Array{Float64, 3}

Compute the gradient of the Hermite map `M` with respect to the coefficients evaluated at `X`.

# Arguments
- `M::HermiteMap`: The Hermite map object containing components.
- `X::Matrix{Float64}`: The input matrix.

# Returns
- `Array{Float64, 3}`: An array of gradients for each component in the Hermite map.

"""
function grad_coeff(M::HermiteMap, X::Matrix{Float64})
    # initialize array for ∇M: size: N*Nx*Ncoeff
    ∇M = zeros(size(X,2), M.Nx, ncoeff(M))

    counter = 1

    for (i, component) in enumerate(M.C)
        ∇M[:,i,counter:counter+component.Nψ-1] += grad_coeff(component.I, X[1:i,:])
        counter += component.Nψ
    end

    return ∇M
end

"""
    grad_coeff_grad_xd(M::HermiteMap, X::Matrix{Float64}) -> Array{Float64, 3}

"""
function grad_coeff_grad_xd(M::HermiteMap, X::Matrix{Float64})
    # initialize array for ∇M: size: N*Nx*Ncoeff
    ∇M = zeros(size(X,2), M.Nx, ncoeff(M))

    counter = 1

    for (i, component) in enumerate(M.C)
        ∇M[:,i,counter:counter+component.Nψ-1] += grad_coeff_grad_xd(component.I, X[1:i,:])
        counter += component.Nψ
    end

    return ∇M
end

"""
    logdeterminant(M::HermiteMap, X::Matrix{Float64}) -> Vector{Float64}
"""
function logdeterminant(M::HermiteMap, X::Matrix{Float64})
    return vec(sum(log.(grad_xd(M, X)), dims=1))
end

"""
    objective_KL(M::HermiteMap, ReferenceDist::MultidimensionalDistribution, quadrature::QuadraturePoints,
                 a::Vector{Float64}, α::Float64; δ::Float64=1e-9)

Compute the Kullback-Leibler (KL) divergence objective function for a given Hermite map `M` and reference distribution `ReferenceDist`.

# Arguments
- `M::HermiteMap`: The Hermite map object.
- `ReferenceDist::MultidimensionalDistribution`: The reference distribution object.
- `quadrature::QuadraturePoints`: The quadrature points used for numerical integration.
- `a::Vector{Float64}`: The coefficients of the Hermite map.
- `α::Float64=.0`: The regularization parameter for L₂ regularization.
- `δ::Float64=1e-9`: A small regularization parameter added to the points to avoid numerical issues (default is 1e-9).

# Returns
- `kl_divergence::Float64`: The computed KL divergence value.

# Notes
- The function evaluates the Hermite map and its gradient at the given quadrature points.
- The KL divergence is computed using numerical quadrature.
- L₂ regularization is added to the KL divergence, controlled by the parameter `α`.
"""
function objective_KL!(obj::Float64, ∇obj::Vector{Float64}, coeff::Vector{Float64},
    M::HermiteMap, Target::MultidimensionalDistribution, quadrature::QuadraturePoints,
    α::Float64, δ::Float64)

    # Extract quadrature points and weights
    points = quadrature.points
    weights = quadrature.weights

    # Set map components
    setcoeff!(M, coeff)

    # evaluate map and its gradient
    Sₓ = evaluate(M, points) .+ δ*points # δ is used for regularization
    ∇Sₓ = grad_xd(M, points) .+ δ

    # Formatting for Optim.jl
    if ∇obj !== nothing
        # Gradient computation (only if required, i.e., when input ∇obj is not nothing)
        fill!(∇obj, 0.0)

        # Gradient of map w.r.t. coefficients
        grad_coefficients = grad_coeff(M, points)

        # Gradient of gradient of map w.r.t. coefficients w.r.t. x
        ∇grad_coefficients = grad_coeff_grad_xd(M, points)

        integrand_grad = sum(Target.grad_logpdf(Sₓ)' .* grad_coefficients + ∇grad_coefficients ./ ∇Sₓ', dims=2)
        integrand_grad = reshape(integrand_grad, size(points, 2), ncoeff(M))

        ∇obj .= -vec(sum(weights .* integrand_grad, dims=1))
    end

    if obj !== nothing
        # compute integrand and numerical quadrature
        integrand = Target.logpdf(Sₓ) + vec(sum(log.(∇Sₓ), dims=1))
        obj = -dot(weights, integrand)

        # Add L₂ regularization specified by α
        D = α*I(length(coeff))
        obj += coeff'*D*coeff
        return obj
    end

end

# Wrapper function for optimization
objective_KL(M, Target, Quadrature, α, δ) =
    (obj, ∇obj, coeff) -> objective_KL!(obj, ∇obj, coeff, M, Target, Quadrature, α, δ)

# Optimize the map using the KL divergence
function nonadaptive(M::HermiteMap, Target::MultidimensionalDistribution, Quadrature::QuadraturePoints,
    α::Float64, δ::Float64)

    # Perform optimization using LBFGS and automatic differentiation
    @time result = Optim.optimize(
        Optim.only_fg!(objective_KL(M, Target, Quadrature, α, δ)),
        zeros(ncoeff(M)), # initialize with zeros
        Optim.LBFGS(),
        Optim.Options(show_trace=false, iterations=1000)
    )

    if !Optim.converged(result)
        println("Optimization hasn't converged")
    end

    return Optim.minimizer(result)

end

# Optimize the map using the KL divergence
function optimize!(M::HermiteMap, Target::MultidimensionalDistribution,
    Quadrature::QuadraturePoints, optimkind::Union{Nothing, Int64, String};
    apply_rescaling::Bool=true, α::Float64=.0, δ::Float64=1e-9)

    # Perform greedy optimization with max terms
    if isnothing(optimkind) || optimkind ∈ ["nonadaptive", "non_adaptive", "non-adaptive"]
        a_opt = nonadaptive(M, Target, Quadrature, α, δ)
    end

    # Set map components
    setcoeff!(M, a_opt)

    if apply_rescaling == true
    #     itransform!(M.L, X)
    end
end

# Wrapper function for optimization for greedy optimization
optimize!(M::HermiteMap, Target::MultidimensionalDistribution, Quadrature::QuadraturePoints;
apply_rescaling::Bool=true, α::Float64=.0, δ::Float64=1e-9) =
    optimize!(M, Target, Quadrature, nothing; apply_rescaling=apply_rescaling, α=α, δ=δ)
