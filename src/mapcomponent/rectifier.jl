

export  Rectifier,
        square, dsquare, d2square,
        softplus, dsoftplus, d2softplus, invsoftplus,
        explinearunit, dexplinearunit, d2explinearunit, invexplinearunit,
        inverse!, inverse,
        grad_x!, grad_x,
        hess_x!, hess_x,
        evaluate!

# Structure for continuous rectifier
# x->rec(x)
struct Rectifier
    T::String
end

square(x) = x^2
dsquare(x) = 2.0*x
d2square(x) = 2.0

# Softplus tools
softplus(x) = (log(1.0 + exp(-abs(log(2.0)*x))) + max(log(2.0)*x, 0.0))/log(2.0)
dsoftplus(x) = 1/(1 + exp(-log(2.0)*x))
d2softplus(x) = log(2.0)/(2.0*(1.0 + cosh(log(2.0)*x)))
invsoftplus(x) = min(log(exp(log(2.0)*x) - 1.0)/log(2.0), x)



explinearunit(x) = x < 0.0 ? exp(x) : x + 1.0
dexplinearunit(x) = x < 0.0 ? exp(x) : 1.0
d2explinearunit(x) = x < 0.0 ? exp(x) : 0.0
invexplinearunit(x) = x < 1.0 ? log(x) : x - 1.0

# Type of the rectifier should be in the following list:
# "squared", "exponential", "softplus", "explinearunit"


Rectifier() = Rectifier("softplus")

function (g::Rectifier)(x)
    if g.T=="squared"
        return square(x)
    elseif g.T=="exponential"
        return exp(x)
    elseif g.T=="softplus"
        return softplus(x)
    elseif g.T=="explinearunit"
        return explinearunit(x)
    end
end

function evaluate!(result, g::Rectifier, x)
    @assert size(result,1) == size(x,1) "Dimension of result and x don't match"
    if g.T=="squared"
        vmap!(square, result, x)
        return result
    elseif g.T=="exponential"
        vmap!(exp, result, x)
        return result
    elseif g.T=="softplus"
        vmap!(softplus, result, x)
        return result
    elseif g.T=="explinearunit"
        vmap!(explinearunit, result, x)
        return result
    end
end

evaluate(g::Rectifier, x) = evaluate!(zero(x), g, x)

function inverse(g::Rectifier, x)
    @assert x>=0 "Input to rectifier is negative"
    if g.T=="squared"
        error("squared rectifier is not invertible")
    elseif g.T=="exponential"
        return log(x)
    elseif g.T=="softplus"
        return invsoftplus(x)
    elseif g.T=="explinearunit"
        return invexplinearunit(x)
    end
end

function inverse!(result, g::Rectifier, x)
    @assert all(x .> 0) "Input to rectifier is negative"
    @assert size(result,1) == size(x,1) "Dimension of result and x don't match"
    if g.T=="squared"
        error("squared rectifier is not invertible")
    elseif g.T=="exponential"
        vmap!(log, result, x)
        return result
    elseif g.T=="softplus"
        vmap!(invsoftplus, result, x)
        return result
    elseif g.T=="explinearunit"
        vmap!(invexplinearunit, result, x)
        return result
    end
end

# inverse(g::Rectifier, x)  = inverse!(zero(x), g, x)


function grad_x(g::Rectifier, x)
    if g.T=="squared"
        return dsquare(x)
    elseif g.T=="exponential"
        return exp(x)
    elseif g.T=="softplus"
        return dsoftplus(x)
    elseif g.T=="explinearunit"
        return dexplinearunit(x)
    end
end

function grad_x!(result, g::Rectifier, x)
    @assert size(result,1) == size(x,1) "Dimension of result and x don't match"
    if g.T=="squared"
        vmap!(dsquare, result, x)
        return result
    elseif g.T=="exponential"
        vmap!(exp, result, x)
        return result
    elseif g.T=="softplus"
        vmap!(dsoftplus, result, x)
        return result
    elseif g.T=="explinearunit"
        vmap!(dexplinearunit, result, x)
        return result
    end
end

# grad_x(g::Rectifier, x) = grad_x!(zero(x), g, x)


function hess_x(g::Rectifier, x::T) where {T <: Real}
    if g.T=="squared"
        return d2square(x)
    elseif g.T=="exponential"
        return exp(x)
    elseif g.T=="softplus"
        return d2softplus(x)
    elseif g.T=="explinearunit"
        return d2explinearunit(x)
    end
end

function hess_x!(result, g::Rectifier, x)
    @assert size(result,1) == size(x,1) "Dimension of result and x don't match"
    if g.T=="squared"
        vmap!(d2square, result, x)
        return result
    elseif g.T=="exponential"
        vmap!(exp, result, x)
        return result
    elseif g.T=="softplus"
        vmap!(d2softplus, result, x)
        return result
    elseif g.T=="explinearunit"
        vmap!(d2explinearunit, result, x)
        return result
    end
end

# hess_x(g::Rectifier, x) = hess_x!(zero(x), g, x)
