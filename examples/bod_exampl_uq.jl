# Bayesian Inference of the Biochemical Oxygen Demand (BOD) Model

using LinearAlgebra
using SpecialFunctions
using UncertaintyQuantification
using Revise
using TransportBasedInference
using Statistics
using Distributions

using Plots
default(fontfamily = "Computer Modern",
        tickfont = font("Computer Modern", 9),
        titlefont = font("Computer Modern", 14),
        guidefont = font("Computer Modern", 12),
        legendfont = font("Computer Modern", 10),
        grid = false)
using LaTeXStrings

# Define the model
function forward_model(t, θ1, θ2)
    A = 0.4 + 0.4 * (1 + erf(θ1 / sqrt(2)))
    B = 0.01 + 0.15 * (1 + erf(θ2 / sqrt(2)))
    return A * (1 - exp(-B * t))
end

# Define the posterior: this is only needed to plot the true density for comparison
function log_posterior(p1, p2)
    # Calculate the log likelihood
    log_likelihood = sum([-log(sqrt(2π) * σ) - 0.5 * ((forward_model(t[k], p1, p2) - D[k]) / σ)^2 for k in 1:5])
    # Calculate the log prior
    return logpdf(Normal(0,1), p1) + logpdf(Normal(0,1), p2) + log_likelihood
end

# Define the prior
prior = RandomVariable.(Normal(0, 1), [:θ1, :θ2])

# data
t = [1, 2, 3, 4, 5]
D = [0.18, 0.32, 0.42, 0.49, 0.54]
σ = sqrt(1e-3)

# Define the likelihood as model
L = ParallelModel(
    df -> sum([-log(sqrt(2π) * σ) - 0.5 * ((forward_model(t[k], df.θ1, df.θ2) - D[k]) / σ)^2 for k in 1:5]), :L
)

likelihood = df -> df.L

# Perform TMCMC
tmcmc = TransitionalMarkovChainMonteCarlo(prior, 1000, 5)
tmcmc_samples, S = bayesianupdating(likelihood, [L], tmcmc)

# Plot the samples
scatter(tmcmc_samples.θ1, tmcmc_samples.θ2, label = "TMCMC samples")

# Get samples as matrix and define the dimension
X = reduce(vcat, [tmcmc_samples.θ1', tmcmc_samples.θ2'])
m = 15

# Transport map construct
S = HermiteMap(m, X; diag = true, b = "CstLinProHermiteBasis");

# Optimization
@time optimize(S, X, 15; withqr = true, verbose = true, P=Thread())

# Plot comparison
Nlog = 100

xrange = range(-1.0; stop = 4.0, length = Nlog)
yrange = range(-1.0; stop = 4.0, length = Nlog)

logposterior = zeros(Nlog, Nlog)
logapprox = zeros(Nlog, Nlog)
lognormal = zeros(Nlog, Nlog)

for (i,x) in enumerate(xrange)
    for (j,y) in enumerate(yrange)
        logposterior[i,j] = log_posterior(x, y)
        logapprox[i,j] = log_pdf(S, reshape([x; y], (2,1)))[1]
        lognormal[i,j] = logpdf(MvNormal(zeros(2), 1.0), [x;y])
    end
end

plt = plot(layout = grid(1, 2), colorbar = false, grid = false)
contour!(plt[1,1], xrange, yrange, exp.(logposterior)', ratio = 1,
         title = "True density",
         color = cgrad([:dodgerblue4, :deepskyblue3, :skyblue, :olivedrab1, :yellow, :orange, :red, :firebrick]),
         xlim = (xrange[1], xrange[end]), ylim = (-Inf, Inf), linewidth = 3)

contour!(plt[1,2], xrange, yrange, exp.(logapprox)', ratio = 1,
         title = "ATM approximate",
         color = cgrad([:dodgerblue4, :deepskyblue3, :skyblue, :olivedrab1, :yellow, :orange, :red, :firebrick]),
         xlim = (xrange[1], xrange[end]), ylim = (-Inf, Inf), linewidth = 3)
