### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 8595f030-5e9b-11eb-3bd7-ef8250cd1fab
begin
	using Random: seed!
	using StatsFuns: logsumexp, softmax
	using LinearAlgebra: diag
	using Optim
	using Distributions
	using DataFrames
	using StatsBase: countmap
	using ForwardDiff # autodiff instead of finite diff?
end

# ╔═╡ 0ece0030-5e9d-11eb-0092-254fc6e72e83
md"
# Logit

Suppose that agents can choose an action $y \in \{0,1\}$. The payoffs to each are
```math
\begin{align*}
u_0(X,\epsilon) &= \epsilon_{i0} \\
u_1(X,\epsilon) &= x_i^\top \beta + \epsilon_{i1}
\end{align*}
```
The shocks $\epsilon_{i0},\epsilon_{i1}$ are distributed as iid Type-I extreme value with mean 0 and scale parameter 1. Agents choose $y_i=1$ if $u_1 \geq u_0$ and $y_0$ if $u_1 < u_0$

I have simulated $X$ and $y$ below. 

1. Write down the log likelihood function for each individual $i$, $\log L_i(y_i|X_i)$ and the score $\nabla_\beta \log L_i(y_i|X_i)$. Feel free to consult Greene
2. Write a function to compute the log likelihood of the data: $\sum_i \log L_i(y_i|X_i)$ and the gradient of this function.
3. Use the finite difference capabilities of `Calculus.jl` to check your gradient and make sure that it's correct
4. Maximize the likelihood and compute standard errors using the Information Matrix. These should be

```math
\left[\sum_i \nabla \log L_i \nabla \log L_i^\top\right]^{-1} \to Var(\beta)
```
    
Greene ch 21 (5th ed) has the appropriate formulas
    
You might be able to make good use of the following functions from [`StatsFuns.jl`](https://github.com/JuliaStats/StatsFuns.jl)

```julia
logsumexp      # log(exp(x) + exp(y)) or log(sum(exp(x)))
softmax        # exp(x_i) / sum(exp(x)), for i
```
"

# ╔═╡ 75c48830-5e9e-11eb-1ec1-f3a083780a1b
begin
	const AV = AbstractVector
	const AM = AbstractMatrix;
end

# ╔═╡ b9c02010-5e9b-11eb-1e5d-9358c778cbba
seed!(1234)

# ╔═╡ b9a85250-5e9b-11eb-1787-6b671b235984
begin
	nobs = 10_000
	β = [1.0, -2.0, 1.0, 0.5]
	k = length(β)
	X = randn(nobs, k);

	# choice utilities
	u0 = zeros(nobs)
	u1 = X*β
	u = hcat(u0, u1)
end

# ╔═╡ b976e210-5e9b-11eb-0b18-e95537e1d64f
# multinomial logit probabilities
prob_actions = mapslices(softmax, u; dims=2)

# ╔═╡ dc8270d0-5e9b-11eb-02cb-1faa58ce2656
cum_prob = cumsum(prob_actions; dims=2)

# ╔═╡ dc82bef0-5e9b-11eb-20f5-0d1ed55e9bb5
@assert all(cum_prob[:,2] .≈ 1)

# ╔═╡ dc87c800-5e9b-11eb-194c-4f7f0e83a5c5
# instead of simulating random type-1 extreme values, we just
# use a uniform variable and the CDF
y = [searchsortedfirst(row, rand()) for row in eachrow(cum_prob) ] .-1 

# ╔═╡ 694cc4c0-5e9c-11eb-3d00-458ca8c8fa28
countmap(y)

# ╔═╡ 73a5d8d0-5e9c-11eb-31ee-5322c7f27f1e
function checksizes(y,X,theta)
    n,k = size(X)
    n == length(y) || throw(DimensionMismatch())
    k == length(theta) || throw(DimensionMismatch())
    return n, k
end

# ╔═╡ 90b963b0-5e9c-11eb-3e39-bdf160b8655e
function loglik(y::AV{Int}, X::AM, theta::AV)
    n,k = checksizes(y,X,theta)
    ff(z) = logcdf(Logistic(), z)

    # see footnote 6 on p. 778 in Greene 6th ed for this shortcut
    q = 2 .* y .- 1
    u1 = X*theta
    LL = sum(ff.(q.*u1))    

    return -LL  # I *think* you'll need to flip sign to maximize
end

# ╔═╡ 90b963b0-5e9c-11eb-1d75-25729dfb57ee
function dloglik!(grad::AV, y::AV{Int}, X::AM, theta::AV)
    
	n,k = checksizes(y,X,theta)    
    k == length(grad) || throw(DimensionMismatch())
    
    u1 = X*theta
    ff(z) = cdf(Logistic(), z)    
    g = y .- ff.(u1)   # as per Greene 6th ed p. 779

    grad .= -vec(sum(g .* X; dims=1))
    return grad
end

# ╔═╡ 90b963b0-5e9c-11eb-0dea-81d683f42601
dloglik(y, X, theta) = dloglik!(similar(theta), y, X, theta)

# ╔═╡ 640b3fa2-5e9c-11eb-19d3-97ebb410851a
# closures wrap likelihood & gradient
begin
	f(thet) = loglik(y,X,thet)
	g!(grad,thet) = dloglik!(grad,y,X,thet)
end

# ╔═╡ 90c65c02-5e9c-11eb-032a-7d600b60b9c4
function informationmatrix(y::AV{Int}, X::AM, theta::AV)

	n,k = checksizes(y,X,theta)    
    infomatrix = zeros(k,k)

    u1 = X*theta
    f(z) = cdf(Logistic(), z)    
    g = y .- f.(u1)   # as per Greene 6th ed p. 779
    
    infomatrix = (g .* X)' * (g .* X)
    
    return infomatrix # maybe flip signs?
end

# ╔═╡ ba5f02b0-5e9c-11eb-30cc-fbb3e2d4b6e2
# initial guess
theta0 = zeros(k)

# ╔═╡ ba5f02b0-5e9c-11eb-2221-4b0d5e525673
# Check gradient against autodiff
@assert ForwardDiff.gradient(f, theta0) ≈ dloglik(y,X,theta0)

# ╔═╡ ba614ca0-5e9c-11eb-3619-0507168a18d6
res = optimize(f, g!, theta0, BFGS(), Optim.Options(;show_trace=true))

# ╔═╡ 61327d20-5e9c-11eb-3704-bf5055d67bb0
begin
	theta1 = res.minimizer  # should be about β
	vcov = informationmatrix(y, X, theta1)
	vcovinv = inv(vcov)
	stderr = sqrt.(diag(vcovinv))
	tstats = theta1 ./ stderr
	pvals = map(z -> 2 .* cdf(Normal(), -abs(z)), tstats)
	
	DataFrame(beta = β, betahat = theta1, tstat=tstats, se = stderr, pval=pvals)
end

# ╔═╡ Cell order:
# ╟─0ece0030-5e9d-11eb-0092-254fc6e72e83
# ╠═8595f030-5e9b-11eb-3bd7-ef8250cd1fab
# ╠═75c48830-5e9e-11eb-1ec1-f3a083780a1b
# ╠═b9c02010-5e9b-11eb-1e5d-9358c778cbba
# ╠═b9a85250-5e9b-11eb-1787-6b671b235984
# ╠═b976e210-5e9b-11eb-0b18-e95537e1d64f
# ╠═dc8270d0-5e9b-11eb-02cb-1faa58ce2656
# ╠═dc82bef0-5e9b-11eb-20f5-0d1ed55e9bb5
# ╠═dc87c800-5e9b-11eb-194c-4f7f0e83a5c5
# ╠═694cc4c0-5e9c-11eb-3d00-458ca8c8fa28
# ╠═73a5d8d0-5e9c-11eb-31ee-5322c7f27f1e
# ╠═90b963b0-5e9c-11eb-3e39-bdf160b8655e
# ╠═90b963b0-5e9c-11eb-1d75-25729dfb57ee
# ╠═90b963b0-5e9c-11eb-0dea-81d683f42601
# ╠═90c65c02-5e9c-11eb-032a-7d600b60b9c4
# ╠═640b3fa2-5e9c-11eb-19d3-97ebb410851a
# ╠═ba5f02b0-5e9c-11eb-30cc-fbb3e2d4b6e2
# ╠═ba5f02b0-5e9c-11eb-2221-4b0d5e525673
# ╠═ba614ca0-5e9c-11eb-3619-0507168a18d6
# ╠═61327d20-5e9c-11eb-3704-bf5055d67bb0
