function integrate_wrt_normal_pdf(f::Function, μ::Real, σ::Real; npts::Int=15)
    nodes, wts = gausshermite(npts)
    return 1/sqrtπ * sum( f.(sqrt2 .* σ .* nodes .+ μ) .* wts)
end

function gh_chg_of_vars(f, ndswts::NTuple{N}, μ, L) where {N}
    nodes = SVector{N}(first.(ndswts)...)
    wt = prod(last.(ndswts))
    feval = f(sqrt2 .* L*nodes .+ μ)
    return wt*feval
end

function integrate_wrt_normal_pdf(f::Function, μ::AbstractVector, L::AbstractMatrix; npts::Int=15)
    k = length(μ)
    k == checksquare(L) || throw(DomainError())

    ndswts = zip(gausshermite(npts)...)

    # https://stackoverflow.com/questions/56120583/n-dimensional-cartesian-product-of-a-set-in-julia
    quadprod = product( ntuple(i->ndswts, k)...)

    #closure
    g(node_wt_ntup) = gh_chg_of_vars(f, node_wt_ntup, μ, L)

    return (1/sqrtπ)^k * sum( g(vw) for vw in quadprod)
end
