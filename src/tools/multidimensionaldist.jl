export MultidimensionalDistribution

struct MultidimensionalDistribution
    logpdf::Function
    grad_logpdf::Function
end
