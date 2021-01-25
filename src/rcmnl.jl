module rcmnl

using StatsFuns: logsumexp
using Base.Iterators: product
using FastGaussQuadrature: gausshermite
using StaticArrays
using LinearAlgebra: mul!, checksquare
using StatsFuns: sqrt2, sqrtπ, invsqrt2π, invπ

export simlogL

const AV = AbstractVector
const AM = AbstractMatrix

include("integration.jl")

"log lik of choices yᵢ conditional on mean inside good utilities uᵢ and shock eₖ"
function logLi!(tmp::Vector{T}, y::AV, u::AM, e::AV) where {T}
    length(y) == size(u,2) || throw(DimensionMismatch())
    logL = zero(T)
    tmp[1] = 0
    for j = 1:size(u,2)
        tmp[2:end] .= view(u,:,j) .+ e
        yit = y[j]
        logL += tmp[yit] - logsumexp(tmp)
    end
    return logL
end


"simulated logL of data using `npts` quadrature points"
function simlogL(y, X, θ::AbstractVector{T}; npts=15) where {T}
    
    # unpack parameters
    β = reshape(view(θ, 1:8), :, 2)
    Σchol = SMatrix{2,2}(θ[end-2], θ[end-1], 0, θ[end])

    # zip together vectors of nodes + weights for Multidim quad
    # and take cartesian product
    # https://gist.github.com/markvdw/f9ca12c99484cf2a881e84cb515b86c8
    vw = zip(gausshermite(npts)...)
    quadprod = product(vw, vw)

    # preallocate temp vars
    u = zeros(T, 2, size(y,1))
    tmp = zeros(T, 3)
    lse = zeros(T, size(quadprod))

    # loop over individuals
    # would be good to parallelize!
    sll = zero(T)
    for j in 1:size(y,2)
        
        # mean utilities of inside good
        @views mul!(u, β', X[:,j,:]')

        # loop over cartesian product of quadrature wts/nodes
        # (don't have to expand these)
        for (i, ((v1,w1), (v2,w2))) in enumerate(quadprod)

            # transform iid vₖ normal to correlatd eₖ
            # using StaticArrays b/c this allocates less
            # probably unecessary
            v = SVector(v1,v2)
            e = sqrt2 .* Σchol * v
            
            # quadrature weight wₖ * π*N/2
            w = w1*w2*π

            # log( wₖ * likᵢ conditional on eₖ)
            lse[i] = log(w) + logLi!(tmp, view(y,:,j), u, e)
        end

        # compute ∑ wtₖ * likₖ
        sll += logsumexp(lse)
    end
    return - invπ * sll
end



end # module
