export
    HermiteMap,
    totalordermap,
    optimize,
    setcoeff!,
    grad_xd,
    objective_KL

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

    if b ∈ ["CstProHermiteBasis", "CstPhyHermiteBasis"]
        m = order+2
    elseif b ∈ ["CstLinProHermiteBasis", "CstLinPhyHermiteBasis"]
        m = order+3
    else
        error("Undefined basis")
    end

    return HermiteMap(m, d, L, C)
end

# Set the coefficients in all map components.
function setcoeff!(M::HermiteMap, a::Vector{Float64})
    counter = 1
    for component in M.C
        setcoeff!(component, a[counter:counter+component.Nψ-1])
        counter += component.Nψ
    end
end

# Optimize the map using the KL divergence
function optimize(M::HermiteMap, Target::MultidimensionalDistribution,
    quadrature::QuadraturePoints, optimkind::Union{Nothing, Int64, String};
    apply_rescaling::Bool=true, α::Float64=.0)

    # Perform greedy optimization with max terms
    if optimkind ∈ ["nonadaptive", "non_adaptive", "non-adaptive"]
        a_opt = nonadaptive(M, Target, quadrature; α)
    end

    # Set map components
    setcoeff!(M, a_opt)

    # if apply_rescaling == true
    #     itransform!(M.L, X)
    # end

    return M

end

function nonadaptive(M::HermiteMap, Target::MultidimensionalDistribution,
    quadrature::QuadraturePoints; α::Float64=.0)

    # Initialize coefficients
    a₀ = zeros(sum([ncoeff(component) for component in M.C]))

    # Optimize map
    objective(a) = objective_KL(M, Target, quadrature, a, α)

    # TODO Test optimization using this function and autodiff with a greedy fit

    # Perform optimization using LBFGS and automatic differentiation
    result = Optim.optimize(
        a -> objective_function(a, M, X),
        a -> gradient_function(a, M, X),
        a₀,
        Optim.LBFGS()
    )

    # TODO Check if the optimization converged


    # Extract the optimized coefficients
    return Optim.minimizer(result)

end

"""
    grad_xd(M::HermiteMap, X::Matrix{Float64}) -> Array{Any,1}

Compute the gradient of the Hermite map `M` with respect to the input matrix `X`.

# Arguments
- `M::HermiteMap`: The Hermite map object containing components.
- `X::Matrix{Float64}`: The input matrix for which the gradient is computed.

# Returns
- `Vector{Float64}`: An array of gradients for each component in the Hermite map.

"""
function grad_xd(M::HermiteMap, X::Matrix{Float64})
    ∇M = zeros(size(X))

    for (i, component) in enumerate(M.C)
        ∇M[i,:] += grad_xd(component.I, X[1:i,:])
    end

    return ∇M
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
function objective_KL(M::HermiteMap, Target::MultidimensionalDistribution, quadrature::QuadraturePoints,
    a::Vector{Float64}, α::Float64=.0; δ::Float64=1e-9)

    # Extract quadrature points and weights
    points = quadrature.points
    weights = quadrature.weights

    # Set map components
    setcoeff!(M, a)

    # evaluate map and its gradient
    Sₓ = evaluate(M, points) .+ δ*points # δ is used for regularization
    ∇Sₓ = grad_xd(M, points) .+ δ

    # compute integrand and numerical quadrature
    integrand = -Target.logpdf(Sₓ) - vec(sum(log.(∇Sₓ), dims=1))
    kl_divergence = dot(weights, integrand)

    # Add L₂ regularization specified by α
    D = α*I(length(a))
    kl_divergence += a'*D*a

    return kl_divergence
end

# TODO: compute gradient of map w.r.t. the coefficients a
# TODO: make compatible with `Optim.only_fg!` as seen here:
# * https://julianlsolvers.github.io/Optim.jl/stable/user/tipsandtricks/
# * and in the `negative_log_likelihood` function in `src/hermitemap/hermitemapcomponent.jl`
# For the gradient of the objective function we need to take care of the gradient of the
# target pdf given in Target.gradient()
function objective_gradient(f, ∇f, coeff)
    # do common computations here
    # ...
    if ∇f !== nothing
        # code to compute gradient here
        # writing the result to the vector ∇f
        # ∇f .= ...

        # ! This depends on the gradient of the target pdf!
    end
    if f !== nothing
        # value = ... code to compute objective function
        return value
    end
end
