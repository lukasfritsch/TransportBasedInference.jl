# Here, we show how to create a map from a density function.

using LinearAlgebra
using TransportBasedInference
using SpecialFunctions
using Statistics
using Distributions
using Plots
using LaTeXStrings

default(fontfamily = "Computer Modern",
        tickfont = font("Computer Modern", 9),
        titlefont = font("Computer Modern", 14),
        guidefont = font("Computer Modern", 12),
        legendfont = font("Computer Modern", 10),
        grid = false)

# Define the banana density
function log_bananadensity(x::Matrix{Float64})
    return logpdf(Normal(0,1), x[1,:]) + logpdf(Normal(0,1), x[2,:] - x[1,:].^2)
end

# Gradient of unnomalized target density required for gradient objective
function grad_x_log_pdf(x::Matrix{Float64})
    grad1 = -x[1,:] + (2*x[1,:].*(x[2,:] - x[1,:].^2))
    grad2 = (x[1,:].^2 - x[2,:])
    return mapreduce(permutedims, vcat, [grad1, grad2])
end
Target = MultidimensionalDistribution(log_bananadensity, grad_x_log_pdf)

# Transport Map
Map = totalordermap(3,2)
#Quadrature = latinhypercube_weights(100,2)
Quadrature = gausshermite_weights(7, 2; sparse=false)
coefficients = zeros(sum([ncoeff(component) for component in Map.C]))


using Optim

result = Optim.optimize(coeff -> objective_KL(Map, Target, Quadrature, coeff), coefficients, LBFGS(), Optim.Options(show_trace=true))

a_opt = Optim.minimizer(result)

setcoeff!(Map, a_opt)

# Plot comparison
Nlog = 100

xrange = range(-3; stop = 3, length = Nlog)
yrange = range(-3; stop = 6, length = Nlog)

logposterior = zeros(Nlog, Nlog)
logapprox = zeros(Nlog, Nlog)

for (i,x) in enumerate(xrange)
    for (j,y) in enumerate(yrange)
        logposterior[i,j] = log_bananadensity(reshape([x; y], (2,1)))[1]
        logapprox[i,j] = log_push_forward_pdf(Map, deepcopy(reshape([x; y], (2,1))))[1]
    end
end

# Push samples through the map
reference_samples = randn(2, 1000)
target_samples = TransportBasedInference.evaluate(Map, reference_samples)
log_pdf(Map, target_samples)

function log_push_forward_pdf(M, X)
    inverse = inverse!(zeros(size(X)), X, M)
    ∇Sₓ = grad_xd(M, inverse)
    logdet = -vec(sum(log.(∇Sₓ), dims=1))
    return logpdf(MvNormal(I(M.Nx)), inverse) + logdet
end

# Plot
plt = plot(layout = grid(1, 2), colorbar = false, grid = false, size=(1200,800))
contour!(plt, xrange, yrange, exp.(logposterior)', subplot=1,
         title = "True density",
         color = cgrad([:dodgerblue4, :deepskyblue3, :skyblue, :olivedrab1, :yellow, :orange, :red, :firebrick]),
         xlim = (-3, 3), ylim = (-3, 6), linewidth = 3)

contour!(plt, xrange, yrange, exp.(logapprox)', subplot=2,
         title = "TM approximate",
         color = cgrad([:dodgerblue4, :deepskyblue3, :skyblue, :olivedrab1, :yellow, :orange, :red, :firebrick]),
         xlim = (-3, 3), ylim = (-3, 6), linewidth = 3)

scatter!(plt, target_samples[1,:], target_samples[2,:], subplot=2,
        label = "Mapped samples", color = :red, markersize = 2)

display(plt)
