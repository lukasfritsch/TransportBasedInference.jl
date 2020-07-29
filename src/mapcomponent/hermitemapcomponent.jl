export  MapComponent,
        EmptyMapComponent,
        ncoeff,
        getcoeff,
        setcoeff!,
        getidx,
        evaluate!,
        evaluate,
        negative_log_likelihood!,
        precond!,
        diagprecond!,
        hess_negative_log_likelihood!


struct MapComponent
    m::Int64
    Nψ::Int64
    Nx::Int64
    # IntegratedFunction
    I::IntegratedFunction
    # Regularization parameter
    α::Float64
end

function MapComponent(I::IntegratedFunction; α::Float64=1e-6)
    return MapComponent(I.m, I.Nψ, I.Nx, I, α)
end

function MapComponent(m::Int64, Nx::Int64, idx::Array{Int64,2}, coeff::Array{Float64,1}; α::Float64 = 1e-6)
    Nψ = size(coeff,1)
    @assert size(coeff,1) == size(idx,1) "Wrong dimension"
    B = MultiBasis(CstProHermite(m-2; scaled =true), Nx)

    return MapComponent(m, Nψ, Nx, IntegratedFunction(ExpandedFunction(B, idx, coeff)), α)
end

function MapComponent(f::ExpandedFunction; α::Float64 = 1e-6)
    return MapComponent(f.m, f.Nψ, f.Nx, IntegratedFunction(f), α)
end


function MapComponent(m::Int64, Nx::Int64; α::Float64 = 1e-6)
    Nψ = 1

    # m is the dimension of the basis
    B = MultiBasis(CstProHermite(m-2; scaled =true), Nx)
    idx = zeros(Int64, Nψ, Nx)
    coeff = zeros(Nψ)

    f = ExpandedFunction(B, idx, coeff)
    I = IntegratedFunction(f)
    return MapComponent(I; α = α)
end

# function EmptyMapComponent(m::Int64, Nx::Int64; α::Float64 = 1e-6)
#
#
#     # m is the dimension of the basis
#     B = MultiBasis(CstProHermite(m-2; scaled =true), Nx)
#     idx = zeros(Int64, Nψ,Nx)
#     coeff = zeros(Nψ)
#
#     f = ExpandedFunction(B, idx, coeff)
#     I = IntegratedFunction(f)
#     return MapComponent(I; α = α)
# end



ncoeff(C::MapComponent) = C.Nψ
getcoeff(C::MapComponent)= C.I.f.f.coeff

function setcoeff!(C::MapComponent, coeff::Array{Float64,1})
        @assert size(coeff,1) == C.Nψ "Wrong dimension of coeff"
        C.I.f.f.coeff .= coeff
end

getidx(C::MapComponent) = C.I.f.f.idx


## Evaluate
function evaluate!(out, C::MapComponent, X)
    @assert C.Nx==size(X,1) "Wrong dimension of the sample"
    @assert size(out,1) == size(X,2) "Dimensions of the output and the samples don't match"
    return evaluate!(out, C.I, X)
end

evaluate(C::MapComponent, X::Array{Float64,2}) =
    evaluate!(zeros(size(X,2)), C, X)

## negative_log_likelihood

# function negative_log_likelihood!(J, dJ, coeff, S::Storage{m, Nψ, k}, C::MapComponent{m, Nψ, k}, X::Array{Float64,2}) where {m, Nψ, k}

function negative_log_likelihood!(J, dJ, coeff, S::Storage, C::MapComponent, X)
    NxX, Ne = size(X)
    m = C.m
    Nx = C.Nx
    Nψ = C.Nψ
    @assert NxX == Nx "Wrong dimension of the sample X"
    @assert size(S.ψoff, 1) == Ne
    @assert size(S.ψoff, 2) == Nψ

    # Output objective, gradient
    xlast = view(X,NxX,:)

    fill!(S.cache_integral, 0.0)

    # Integrate at the same time for the objective, gradient
    function integrand!(v::Vector{Float64}, t::Float64)
        repeated_grad_xk_basis!(S.cache_dcψxdt, S.cache_gradxd, C.I.f.f, t*xlast)

        @avx @. S.cache_dψxd = (S.cache_dcψxdt .* S.ψoff) *ˡ coeff
        # S.cache_dcψxdt .*= S.ψoff
        # mul!(S.cache_dψxd, S.cache_dcψxdt, coeff)

        # Integration for J
        vJ = view(v,1:Ne)
        evaluate!(vJ, C.I.g, S.cache_dψxd)

        # Integration for dcJ

        grad_x!(S.cache_dψxd, C.I.g, S.cache_dψxd)

        @avx for j=2:Nψ+1
            for i=1:Ne
                v[(j-1)*Ne+i] = S.cache_dψxd[i]*S.cache_dcψxdt[i,j-1]
            end
        end

        # v[Ne+1:Ne+Ne*Nψ] .= reshape(S.cache_dψxd .* S.cache_dcψxdt , (Ne*Nψ))
        # v[Ne+1:Ne+Ne*Nψ] .= reshape(grad_x(C.I.g, S.cache_dψxd) .* S.cache_dcψxdt , (Ne*Nψ))
    end

    quadgk!(integrand!, S.cache_integral, 0.0, 1.0; rtol = 1e-3)#; order = 9, rtol = 1e-10)

    # Multiply integral by xlast (change of variable in the integration)
    @avx for j=1:Nψ+1
        for i=1:Ne
            S.cache_integral[(j-1)*Ne+i] *= xlast[i]
        end
    end

    # Multiply integral by xlast (change of variable in the integration)
    # @inbounds for j=1:Nψ+1
    #     @. S.cache_integral[(j-1)*Ne+1:j*Ne] *= xlast
    # end

    # Add f(x_{1:d-1},0) i.e. (S.ψoff .* S.ψd0)*coeff to S.cache_integral
    @avx for i=1:Ne
        f0i = zero(Float64)
        for j=1:Nψ
            f0i += (S.ψoff[i,j] * S.ψd0[i,j])*coeff[j]
        end
        S.cache_integral[i] += f0i
    end

    # Store g(∂_{xk}f(x_{1:k})) in S.cache_g
    @avx for i=1:Ne
        prelogJi = zero(Float64)
        for j=1:Nψ
            prelogJi += (S.ψoff[i,j] * S.dψxd[i,j])*coeff[j]
        end
        S.cache_g[i] = prelogJi
    end

    # Formatting to use with Optim.jl
    if dJ != nothing
        reshape_cacheintegral = reshape(S.cache_integral[Ne+1:end], (Ne, Nψ))
        fill!(dJ, 0.0)
        @inbounds for i=1:Ne
            for j=1:Nψ
            dJ[j] += gradlog_pdf(S.cache_integral[i])*(reshape_cacheintegral[i,j] + S.ψoff[i,j]*S.ψd0[i,j]) + # dsoftplus(S.cache_g[i])*S.ψoff[i,j]*S.dψxd[i,j]*(1/softplus(S.cache_g[i]))
                     grad_x(C.I.g, S.cache_g[i])*S.ψoff[i,j]*S.dψxd[i,j]/C.I.g(S.cache_g[i])
            end
        end
        rmul!(dJ, -1/Ne)
        # Add derivative of the L2 penalty term ∂_c α ||c||^2 = 2 *α c
        dJ .+= 2*C.α*coeff
    end

    if J != nothing
        J = 0.0
        @avx for i=1:Ne
            J += log_pdf(S.cache_integral[i]) + log(C.I.g(S.cache_g[i]))
        end
        J *=(-1/Ne)
        J += C.α*norm(coeff)^2
        return J
    end
end

negative_log_likelihood(S::Storage, C::MapComponent, X) = (J, dJ, coeff) -> negative_log_likelihood!(J, dJ, coeff, S, C, X)


function precond!(P, coeff, S::Storage, C::MapComponent, X)
    Nψ = C.Nψ
    NxX, Ne = size(X)
    @assert NxX == C.Nx "Wrong dimension of the sample X"
    @assert size(S.ψoff, 1) == Ne
    @assert size(S.ψoff, 2) == Nψ

    # Output objective, gradient
    xlast = view(X,NxX,:)#)

    fill!(S.cache_integral, 0)

    # Integrate at the same time for the objective, gradient
    function integrand!(v::Vector{Float64}, t::Float64)
        repeated_grad_xk_basis!(S.cache_dcψxdt, S.cache_gradxd, C.I.f.f, t*xlast)

        @avx @. S.cache_dψxd = (S.cache_dcψxdt .* S.ψoff) *ˡ coeff
        # S.cache_dcψxdt .*= S.ψoff
        # mul!(S.cache_dψxd, S.cache_dcψxdt, coeff)

        # Integration for J
        vJ = view(v,1:Ne)
        evaluate!(vJ, C.I.g, S.cache_dψxd)

        # Integration for dcJ

        grad_x!(S.cache_dψxd, C.I.g, S.cache_dψxd)

        # v[Ne+1:Ne+Ne*Nψ] .= reshape(S.cache_dψxd .* S.cache_dcψxdt , (Ne*Nψ))
        @avx for j=2:Nψ+1
            for i=1:Ne
                v[(j-1)*Ne+i] = S.cache_dcψxdt[i,j-1]*S.cache_dψxd[i]
            end
        end
        # v[Ne+1:Ne+Ne*Nψ] .= reshape(grad_x(C.I.g, S.cache_dψxd) .* S.cache_dcψxdt , (Ne*Nψ))
    end

    quadgk!(integrand!, S.cache_integral, 0.0, 1.0; rtol = 1e-3)#; order = 9, rtol = 1e-10)

    # Multiply integral by xk (change of variable in the integration)
    # @inbounds for j=1:Nψ+1
    #     @. S.cache_integral[(j-1)*Ne+1:j*Ne] *= xlast
    # end
    @avx for j=1:Nψ+1
        for i=1:Ne
            S.cache_integral[(j-1)*Ne+i] *= xlast[i]
        end
    end

    # Add f(x_{1:d-1},0) i.e. (S.ψoff .* S.ψd0)*coeff to S.cache_integral
    @avx for i=1:Ne
        f0i = zero(Float64)
        for j=1:Nψ
            f0i += (S.ψoff[i,j] * S.ψd0[i,j])*coeff[j]
        end
        S.cache_integral[i] += f0i
    end

    # Store g(∂_{xk}f(x_{1:k})) in S.cache_g
    @avx for i=1:Ne
        prelogJi = zero(Float64)
        for j=1:Nψ
            prelogJi += (S.ψoff[i,j] * S.dψxd[i,j])*coeff[j]
        end
        S.cache_g[i] = prelogJi
    end


    reshape_cacheintegral = reshape(S.cache_integral[Ne+1:Ne+Ne*Nψ], (Ne, Nψ))
    # reshape2_cacheintegral = reshape(S.cache_integral[Ne + Ne*Nψ + 1: Ne + Ne*Nψ + Ne*Nψ*Nψ], (Ne, Nψ, Nψ))
    # @show reshape2_cacheintegral
    fill!(P, 0.0)
    @inbounds for l=1:Ne
        # Exploit symmetry of the Hessian
        for i=1:Nψ
            for j=i:Nψ
            # P[i,j] +=  reshape2_cacheintegral[l,i,j]*S.cache_integral[l]
            P[i,j] +=  (reshape_cacheintegral[l,i] + S.ψoff[l,i]*S.ψd0[l,i]) * (reshape_cacheintegral[l,j] + S.ψoff[l,j]*S.ψd0[l,j])
            P[i,j] -=  ( (S.ψoff[l,i]*S.dψxd[l,i]) * (S.ψoff[l,j]*S.dψxd[l,j])*(
                            hess_x(C.I.g, S.cache_g[l]) * C.I.g(S.cache_g[l]) -
                            grad_x(C.I.g, S.cache_g[l])^2))/C.I.g(S.cache_g[l])^2

            P[j,i] = P[i,j]
            end
        end
    end
    rmul!(P, 1/Ne)
    # Add derivative of the L2 penalty term ∂^2_c α ||c||^2 = 2 *α *I
    @inbounds for i=1:Nψ
        P[i,i] += 2*C.α*I
    end
    return P
end

precond!(S::Storage, C::MapComponent, X) = (P, coeff) -> precond!(P, coeff, S, C, X)

function diagprecond!(P, coeff, S::Storage, C::MapComponent, X::Array{Float64,2})
    Nψ = C.Nψ
    Nx = C.Nx
    NxX, Ne = size(X)
    @assert NxX == Nx "Wrong dimension of the sample X"
    @assert size(S.ψoff, 1) == Ne
    @assert size(S.ψoff, 2) == Nψ

    # Output objective, gradient
    xlast = view(X, Nx,:)#)

    fill!(S.cache_integral, 0)

    # Integrate at the same time for the objective, gradient
    function integrand!(v::Vector{Float64}, t::Float64)
        S.cache_dcψxdt .= repeated_grad_xk_basis(C.I.f.f, t*xlast)

        # @avx @. S.cache_dψxd = (S.cache_dcψxdt .* S.ψoff) *ˡ coeff
        S.cache_dcψxdt .*= S.ψoff
        mul!(S.cache_dψxd, S.cache_dcψxdt, coeff)

        # Integration for J
        vJ = view(v,1:Ne)
        evaluate!(vJ, C.I.g, S.cache_dψxd)
        # v[1:Ne] .= C.I.g(S.cache_dψxd)

        # Integration for dcJ
        v[Ne+1:Ne+Ne*Nψ] .= reshape(grad_x(C.I.g, S.cache_dψxd) .* S.cache_dcψxdt , (Ne*Nψ))
    end

    quadgk!(integrand!, S.cache_integral, 0.0, 1.0; rtol = 1e-3)#; order = 9, rtol = 1e-10)

    # Multiply integral by xk (change of variable in the integration)
    @inbounds for j=1:Nψ+1
        @. S.cache_integral[(j-1)*Ne+1:j*Ne] *= xlast
    end

    # Add f(x_{1:d-1},0) i.e. (S.ψoff .* S.ψd0)*coeff to S.cache_integral
    @avx for i=1:Ne
        f0i = zero(Float64)
        for j=1:Nψ
            f0i += (S.ψoff[i,j] * S.ψd0[i,j])*coeff[j]
        end
        S.cache_integral[i] += f0i
    end

    # Store g(∂_{xk}f(x_{1:k})) in S.cache_g
    @avx for i=1:Ne
        prelogJi = zero(Float64)
        for j=1:Nψ
            prelogJi += (S.ψoff[i,j] * S.dψxd[i,j])*coeff[j]
        end
        S.cache_g[i] = prelogJi
    end


    reshape_cacheintegral = reshape(S.cache_integral[Ne+1:Ne+Ne*Nψ], (Ne, Nψ))
    # reshape2_cacheintegral = reshape(S.cache_integral[Ne + Ne*Nψ + 1: Ne + Ne*Nψ + Ne*Nψ*Nψ], (Ne, Nψ, Nψ))
    # @show reshape2_cacheintegral
    fill!(P, 0.0)
    @inbounds for l=1:Ne
        # Exploit symmetry of the Hessian
        for i=1:Nψ
            # P[i,j] +=  reshape2_cacheintegral[l,i,j]*S.cache_integral[l]
            P[i] +=  (reshape_cacheintegral[l,i] + S.ψoff[l,i]*S.ψd0[l,i])^2# * (reshape_cacheintegral[l,j] + S.ψoff[l,j]*S.ψd0[l,j])
            P[i] -=  ( (S.ψoff[l,i]*S.dψxd[l,i])^2*(
                            hess_x(C.I.g, S.cache_g[l]) * C.I.g(S.cache_g[l]) -
                            grad_x(C.I.g, S.cache_g[l])^2))/C.I.g(S.cache_g[l])^2
        end
    end
    rmul!(P, 1/Ne)
    # Add derivative of the L2 penalty term ∂^2_c α ||c||^2 = 2 *α *I
    @inbounds for i=1:Nψ
        P[i] += 2*C.α
    end
    return P
end

diagprecond!(S::Storage, C::MapComponent, X::Array{Float64,2}) = (P, coeff) -> diagprecond!(P, coeff, S, C, X)


function hess_negative_log_likelihood!(J, dJ, d2J, coeff, S::Storage, C::MapComponent, X::Array{Float64,2})
    Nψ = C.Nψ
    Nx = C.Nx

    NxX, Ne = size(X)
    @assert NxX == Nx "Wrong dimension of the sample X"
    @assert size(S.ψoff, 1) == Ne
    @assert size(S.ψoff, 2) == Nψ

    # Output objective, gradient
    xlast = view(X, NxX, :)#)

    fill!(S.cache_integral, 0)

    dcdψouter = zeros(Ne, Nψ, Nψ)

    # Integrate at the same time for the objective, gradient
    function integrand!(v::Vector{Float64}, t::Float64)
        S.cache_dcψxdt .= repeated_grad_xk_basis(C.I.f.f, t*xlast)

        # @avx @. S.cache_dψxd = (S.cache_dcψxdt .* S.ψoff) *ˡ coeff
        S.cache_dcψxdt .*= S.ψoff
        mul!(S.cache_dψxd, S.cache_dcψxdt, coeff)

        @inbounds for i=1:Nψ
            for j=1:Nψ
                dcdψouter[:,i,j] = S.cache_dcψxdt[:,i] .* S.cache_dcψxdt[:, j]
            end
        end

        # Integration for J
        vJ = view(v,1:Ne)
        evaluate!(vJ, C.I.g, S.cache_dψxd)
        # v[1:Ne] .= C.I.g(S.cache_dψxd)

        # Integration for dcJ
        v[Ne+1:Ne+Ne*Nψ] .= reshape(grad_x(C.I.g, S.cache_dψxd) .* S.cache_dcψxdt , (Ne*Nψ))

        # Integration for d2cJ
        v[Ne + Ne*Nψ + 1: Ne + Ne*Nψ + Ne*Nψ*Nψ] .= reshape(hess_x(C.I.g, S.cache_dψxd) .* dcdψouter, (Ne*Nψ*Nψ))

    end

    quadgk!(integrand!, S.cache_integral, 0.0, 1.0; rtol = 1e-3)#; order = 9, rtol = 1e-10)

    # Multiply integral by xlast (change of variable in the integration)
    @inbounds for j=1:1 + Nψ# + Nψ*Nψ
        @. S.cache_integral[(j-1)*Ne+1:j*Ne] *= xlast
    end

    # Add f(x_{1:d-1},0) i.e. (S.ψoff .* S.ψd0)*coeff to S.cache_integral
    @avx for i=1:Ne
        f0i = zero(Float64)
        for j=1:Nψ
            f0i += (S.ψoff[i,j] * S.ψd0[i,j])*coeff[j]
        end
        S.cache_integral[i] += f0i
    end

    # Store g(∂_{xk}f(x_{1:k})) in S.cache_g
    @avx for i=1:Ne
        prelogJi = zero(Float64)
        for j=1:Nψ
            prelogJi += (S.ψoff[i,j] * S.dψxd[i,j])*coeff[j]
        end
        S.cache_g[i] = prelogJi
    end


    # Formatting to use with Optim.jl
    if dJ != nothing
        reshape_cacheintegral = reshape(S.cache_integral[Ne+1:Ne+Ne*Nψ], (Ne, Nψ))
        fil!(dJ, 0.0)#dJ .= zeros(Nψ)
        @inbounds for i=1:Ne
            # dJ .= zeros(Nψ)
            for j=1:Nψ
            dJ[j] += gradlog_pdf(S.cache_integral[i])*(reshape_cacheintegral[i,j] + S.ψoff[i,j]*S.ψd0[i,j])
            dJ[j] += grad_x(C.I.g, S.cache_g[i])*S.ψoff[i,j]*S.dψxd[i,j]/C.I.g(S.cache_g[i])
            end
            # @show i, dJ
        end
        rmul!(dJ, -1/Ne)
        # Add derivative of the L2 penalty term ∂_c α ||c||^2 = 2 *α c
        dJ .+= 2*C.α*coeff
    end

    if d2J != nothing
        reshape_cacheintegral = reshape(S.cache_integral[Ne+1:Ne+Ne*Nψ], (Ne, Nψ))
        reshape2_cacheintegral = reshape(S.cache_integral[Ne + Ne*Nψ + 1: Ne + Ne*Nψ + Ne*Nψ*Nψ], (Ne, Nψ, Nψ))
        # @show reshape2_cacheintegral
        fill!(d2J, 0.0)
        # d2J .= zeros(Nψ, Nψ)
        @inbounds for l=1:Ne
            # Exploit symmetry of the Hessian
            for j=1:Nψ
                for i=j:Nψ
                d2J[i,j] +=  reshape2_cacheintegral[l,i,j]*S.cache_integral[l]
                d2J[i,j] +=  (reshape_cacheintegral[l,i] + S.ψoff[l,i]*S.ψd0[l,i]) * (reshape_cacheintegral[l,j] + S.ψoff[l,j]*S.ψd0[l,j])
                d2J[i,j] -=  ( (S.ψoff[l,i]*S.dψxd[l,i]) * (S.ψoff[l,j]*S.dψxd[l,j])*(
                                hess_x(C.I.g, S.cache_g[l]) * C.I.g(S.cache_g[l]) -
                                grad_x(C.I.g, S.cache_g[l])^2))/C.I.g(S.cache_g[l])^2

                d2J[j,i] = d2J[i,j]
                end
            end
        end
        rmul!(d2J, 1/Ne)
        # Add derivative of the L2 penalty term ∂^2_c α ||c||^2 = 2 *α *I
        @inbounds for i=1:Nψ
            d2J[i,i] += 2*C.α*I
        end
        # d2J = Symmetric(d2J)
        # return d2J
    end

    if J != nothing
        J = 0.0
        @inbounds for i=1:Ne
            J += log_pdf(S.cache_integral[i]) + log(C.I.g(S.cache_g[i]))
        end
        J *=(-1/Ne)
        J += C.α*norm(coeff)^2
        return J
    end
    # return J, dJ, d2J
end


hess_negative_log_likelihood!(S::Storage, C::MapComponent, X::Array{Float64,2}) =
    (J, dJ, d2J, coeff) -> hess_negative_log_likelihood!(J, dJ, d2J, coeff, S, C, X)
