# using Revise # uncomment for development

using Test
using Random
using StatsFuns
using Optim
using StatsBase: countmap
using LinearAlgebra
using ForwardDiff
using DataFrames
using Distributions
using BenchmarkTools: @btime

using rcmnl

@testset "Gauss Hermite integration" begin
    @test rcmnl.integrate_wrt_normal_pdf(x -> x, 1.23, 2.34; npts=15) ≈ 1.23
    @test sqrt(rcmnl.integrate_wrt_normal_pdf(x -> (x-1.23)^2, 1.23, 2.34; npts=15)) ≈ 2.34

    μ = [1.0, 0.5]
    L = [1.0  0.0; 0.7 0.3]

    @test rcmnl.integrate_wrt_normal_pdf(x -> (x-μ)*(x-μ)', μ, L; npts=7) ≈ L*L'
    @test rcmnl.integrate_wrt_normal_pdf(x -> x,            μ, L; npts=7) ≈ μ
end

Random.seed!(1234)

# dim of data
nindv    = 1_000
nt       = 20
nquadpts = 7

# coefs
β = [ 1.0  0.3;
     -2.0  0.0;
      0.0  1.0;
      0.2  0.5]
k, nchoice = size(β)
@assert nchoice == 2

# cholesky decomposition of Σ
Σchol = [1.0 0.0;
         0.5 0.5]
θtrue = vcat(vec(β), Σchol[:,1], Σchol[end])

# generate X and V
X = randn(nt, nindv, k)
V = randn(nindv,2)*Σchol' # need Σchol b/c usu. assume U is a 2-dim vector

# utilities
U = zeros(3, nt, nindv)
mul!(reshape(view(U,2:3,:,:), 2,:), β', reshape(X, :, k)')
U[2,:,:] .+= V[:,1]'
U[3,:,:] .+= V[:,2]'

# probabilities
PrU = mapslices(softmax, U; dims=1)
cumPrU = cumsum(PrU; dims=1)

# instead of simulating random type-1 extreme values, we just
# use a uniform variable and the CDF
y = Array{Int64}(undef, nt, nindv)
for idx in CartesianIndices(y)
    vw = view(cumPrU, :, idx)
    y[idx] = searchsortedfirst(vw, rand())
end

# how many of each choice?
countmap(vec(y))

# run function once
simlogL(y, X, θtrue; npts=nquadpts)
println("It runs!")

# wrapper for optimizer
f(θ) = simlogL(y, X, θ; npts=nquadpts)

# starting value
theta0 = vcat(vec(β), Σchol[:,1], Σchol[end]) .* 2

# time function
@btime f(theta0)

# profile function in IDE
# @profileview f(theta0)

# optimize
res = optimize(f, theta0, BFGS(), Optim.Options(;show_trace=true), autodiff=:forward)

# get results
thetahat = res.minimizer
H = ForwardDiff.hessian(f, thetahat)
se = sqrt.(diag(inv(H)))
pval = 2*normcdf.(-abs.(thetahat./se))

# nice table
results_to_print = DataFrame(
    θtrue = θtrue,
    θhat = thetahat,
    se = se,
    pval = pval
)
@show results_to_print

# Wald test
alpha_reject = 0.001
wald_test_stat = (thetahat - θtrue)' * (H \ (thetahat - θtrue))
wald_test_p = ccdf(Chisq(length(θtrue)), wald_test_stat)
if wald_test_p < alpha_reject
    println("Reject that we got the right answer")
else
    println("We fail to reject our estimate!!")
end

@test wald_test_p > alpha_reject
