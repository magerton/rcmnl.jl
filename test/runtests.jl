using Revise
using Random
using StatsFuns
using Optim
using StatsBase
using LinearAlgebra

using rcmnl

Random.seed!(1234)

nindv = 1_000
nt    = 20

β1 = [1.0, -2.0,]
β2 = [1.0,  0.5]
Σchol = [1.0 0.0;
         0.5 0.5]

k = length(β1)
X1 = randn(nt, nindv, k)
X2 = randn(nt, nindv, k)
V = randn(nindv,2)*Σchol' # need Σchol b/c usu. assume U is a 2-dim vector

# utilities
U = zeros(3, nt, nindv)
U0 = view(U,1, :,:)
U1 = view(U,2, :,:)
U2 = view(U,3, :,:)
mul!(vec(U1), reshape(X1, :, k), β1)
mul!(vec(U2), reshape(X2, :, k), β2)
U1 .+= V[:,1]'
U2 .+= V[:,2]'

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

countmap(vec(y))

simlogL(y, X1, X2, vcat(β1, β2, Σchol[:,1], Σchol[end]))

f(θ) = simlogL(y, X1, X2, θ)
theta0 = vcat(β1, β2, Σchol[:,1], Σchol[end]) .* 2
@btime f(theta0)
@profview f(theta0)

res = optimize(f, theta0, BFGS(), Optim.Options(;show_trace=true), autodiff=:forward)



