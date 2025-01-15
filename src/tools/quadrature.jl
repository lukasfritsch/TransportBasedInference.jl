export
    QuadraturePoints,
    gausshermite_weights,
    montecarlo_weights,
    latinhypercube_weights

struct QuadraturePoints
    number_points::Int64
    dimension::Int64
    points::Matrix{Float64}
    weights::Vector{Float64}
    type::String
end

"""
    gausshermite_weights(n::Int64, d::Int64; sparse::Bool=false) -> QuadraturePoints

Generate (sparse) Gauss-Hermite quadrature points and weights.

# Arguments
- `n::Int64`: The number of points.
- `d::Int64`: The dimension of the points.
- `sparse::Bool`: Whether to generate a sparse grid. Defaults to `false`.

# Returns
- `QuadraturePoints`: A `QuadraturePoints` object containing the generated points and weights.

"""
function gausshermite_weights(n::Int64, d::Int64; sparse::Bool=false)

    if sparse
        # Generate sparse grid
        points, weights = sparsegrid(d, n, normalize_gausshermite)
        type = "SparseGaussHermite"
    else
        # Generate full grid
        points, weights = tensorgrid(d, n, normalize_gausshermite)
        type = "GaussHermite"
    end

    return QuadraturePoints(length(points), d, hcat(points...), weights, type)
end

"""
    montecarlo_weights(n::Int64, d::Int64) -> QuadraturePoints

Generate Monte Carlo quadrature points and weights.

# Arguments
- `n::Int64`: The number of points.
- `d::Int64`: The dimension of the points.

# Returns
- `QuadraturePoints`: A `QuadraturePoints` object containing the generated points and weights.

"""
function montecarlo_weights(n::Int64, d::Int64)
    points = randn(d, n)
    weights = 1/n*ones(n)
    return QuadraturePoints(n, d, points, weights, "MonteCarlo")
end


"""
    latinhypercube_weights(n::Int64, d::Int64) -> QuadraturePoints

Generate Latin Hypercube quadrature points and weights.

# Arguments
- `n::Int64`: The number of points.
- `d::Int64`: The dimension of the points.

# Returns
- `QuadraturePoints`: A `QuadraturePoints` object containing the generated points and weights.

"""
function latinhypercube_weights(n::Int64, d::Int64)
    points = [quantile(Normal(), u) for u in QuasiMonteCarlo.sample(n, d, LatinHypercubeSample())]
    weights = 1/n*ones(n)
    return QuadraturePoints(n, d, points, weights, "LatinHypercube")
end

# Noramilized Gauss-Hermite quadrature for integration w.r.t. standard normal density
function normalize_gausshermite(n::Int64)
    x, w = gausshermite(n)
    return ( sqrt(2.0) * x, w / sqrt(π) )
end


# TODO add Smolyak grid
