using Distributions
using TransportBasedInference

function grad_logpdf_banana(x::Matrix{Float64})
    @assert size(x, 2) == 2

    X1 = Normal(0,0.5)
    X2_noise = Normal(0,0.1)

    x2_noise = x[:, 2] .- x[:, 1].^2
    grad_logpi = [
        grad_logpdf(X1, x[:, 1]) .+ grad_logpdf(X2_noise, x2_noise) .* (-2 .* x[:, 1]),
        grad_logpdf(X2_noise, x2_noise)
    ]
    return grad_logpi
end

function grad_logpdf(dist::Normal, x::Vector{Float64})
    return 1/dist.σ^2 .* (dist.μ .- x)
end

function hess_logpdf(dist::Normal, x::Vector{Float64})
    return -1/dist.σ^2 .* ones(length(x))
end


Banana = MultidimensionalDistribution(log_pdf_banana, grad_logpdf_banana)
