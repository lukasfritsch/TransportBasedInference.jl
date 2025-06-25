# Here, we show how to create a map from a density function.

using LinearAlgebra
using TransportBasedInference
using SpecialFunctions
using Statistics
using Distributions
using Plots
using LaTeXStrings

default(
    framestyle=:box,
    fontfamily = "Computer Modern",
    tickfont = font("Computer Modern", 9),
    titlefont = font("Computer Modern", 14),
    guidefont = font("Computer Modern", 12),
    legendfont = font("Computer Modern", 10),
    grid = true,
    margin = 5*Plots.mm
)

# Define the model
function forward_model(t, θ1, θ2)
    A = 0.4 + 0.4 * (1 + erf(θ1 / sqrt(2)))
    B = 0.01 + 0.15 * (1 + erf(θ2 / sqrt(2)))
    return A * (1 - exp(-B * t))
end

# data
t = [1, 2, 3, 4, 5]
D = [0.18, 0.32, 0.42, 0.49, 0.54]
σ = sqrt(1e-3)

# Define the posterior
function log_posterior(x)
    p1, p2 = x
    # Calculate the log likelihood
    log_likelihood = sum([-log(sqrt(2π) * σ) - 0.5 * ((forward_model(t[k], p1, p2) - D[k]) / σ)^2 for k in 1:5])
    # Calculate the log prior
    return logpdf(Normal(0,1), p1) + logpdf(Normal(0,1), p2) + log_likelihood
end

# Gradient of unnomalized target density required for gradient objective
#! TBD
function grad_x_log_pdf(x::Matrix{Float64})
    grad1 = 1
    grad2 = 1
    return mapreduce(permutedims, vcat, [grad1, grad2])
end
Target = MultidimensionalDistribution(x-> log_posterior.(eachcol(x)), grad_x_log_pdf)

# Transport Map
Map = totalordermap(3,2)
Quadrature = latinhypercube_weights(500,2)
# Quadrature = gausshermite_weights(9, 2; sparse=false)
coefficients = zeros(sum([ncoeff(component) for component in Map.C]))

# Optimize the map
using Optim

result = Optim.optimize(coeff -> objective_KL(Map, Target, Quadrature, coeff), coefficients, LBFGS(), Optim.Options(show_trace=true))
a_opt = Optim.minimizer(result)
setcoeff!(Map, a_opt)

# Define the log push forward density
function log_push_forward_pdf(M, X)
    inverse!(zeros(size(X)), X, M)
    ∇Sₓ = grad_xd(M, X)
    logdet = -vec(sum(log.(∇Sₓ), dims=1))
    return logpdf(MvNormal(I(M.Nx)), X) + logdet
end

# Plot comparison
Nlog = 100

xrange = range(-1; stop = 2, length = Nlog)
yrange = range(-1; stop = 3, length = Nlog)

logposterior = zeros(Nlog, Nlog)
logapprox = zeros(Nlog, Nlog)

for (i,x) in enumerate(xrange)
    for (j,y) in enumerate(yrange)
        logposterior[i,j] = log_posterior(reshape([x; y], (2,1)))[1]
        logapprox[i,j] = log_push_forward_pdf(Map, reshape([x; y], (2,1)))[1]
    end
end

# Push samples through the map
reference_samples = randn(2, 1000)
target_samples = TransportBasedInference.evaluate(Map, reference_samples)
log_pdf(Map, target_samples)


# Plot
plt = plot(layout = grid(1, 2), colorbar = false, size=(1200,800))
contour!(plt, xrange, yrange, exp.(logposterior)', subplot=1,
         title = "True density",
         color = cgrad([:dodgerblue4, :deepskyblue3, :skyblue, :olivedrab1, :yellow, :orange, :red, :firebrick]),
         xlim = (-1, 2), ylim = (-1, 3), linewidth = 3)

contour!(plt, xrange, yrange, exp.(logapprox)', subplot=2,
         title = "TM approximate",
         color = cgrad([:dodgerblue4, :deepskyblue3, :skyblue, :olivedrab1, :yellow, :orange, :red, :firebrick]),
         xlim = (-1, 2), ylim = (-1, 3), linewidth = 3)

scatter!(plt, target_samples[1,:], target_samples[2,:], subplot=2,
        label = "Mapped samples", color = :red, markersize = 2)

display(plt)

# Variance diagnostic
function variance_diagnostic(M, Target, X)
    ref_logpdf = logpdf(MvNormal(I(M.Nx)), X)
    Sₓ = evaluate(M, X)
    ∇Sₓ = grad_xd(M, X)
    logdet = vec(sum(log.(∇Sₓ), dims=1))
    pullback_logpdf = Target.logpdf(Sₓ) + logdet
    diff = ref_logpdf - pullback_logpdf
    expect = mean(diff)
    return 0.5*mean((diff.-expect) .^2)
end

# Compute variance diagnostic
var_diag = variance_diagnostic(Map, Target, randn(2, 1000))

# Print final coeffs and objective
println("==================")
println("Variance diagnostic: $var_diag")
println("==================")
