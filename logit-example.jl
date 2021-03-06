### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 8595f030-5e9b-11eb-3bd7-ef8250cd1fab
begin
	# import the entire package
	using Optim
	using Distributions
	using DataFrames

	# import just a few functions
	using Random: seed!
	using StatsFuns: logsumexp, softmax
	using LinearAlgebra: diag
	using StatsBase: countmap
	
	# autodiff instead of finite diff?
	using FiniteDiff: finite_difference_gradient
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
"

# ╔═╡ b7910ac0-5ffe-11eb-3925-13fbd88d3f67
md" 
Denote $cdf(z) = F(z)$. Let choice $y_i \in \{0,1\}$. For a symmetric distribution and a binary discrete choice model, we can use this shortcut (trick is in Greene's Econometrics tome, Greene 6th ed p. 779):
```math
\log L(y|X) = \sum_i \log F\left( 2(y_i-1) x_i^\top \beta \right)
```
Score is a vector
```math
\nabla_\beta \log L(y_i|x_i) = \left[y_i - F\left(x_i^\top\beta\right)\right]x_i
```
Information matrix is
```math
\left[\sum_i \nabla \log L_i \nabla \log L_i^\top\right] \to Var(\beta)
```
"

# ╔═╡ e2248c2e-5ffe-11eb-189c-85dbf7703168
md"
We can vectorize stuff to make it simpler. The $.$ means element-by-element operations a la MATLAB/Julia. Define
```math
\boldsymbol q \equiv 2.\boldsymbol y .- 1
```
Then 
```math
\log L(y|X) = \sum_i \log. F.\left(\boldsymbol q .* X\beta \right)
```
Matrix of scores
```math
\frac{\partial \log L_i}{\partial \beta} = \left(y .- F.(X\beta)\right) .* X
```
Information matrix
```math
\left(\frac{\partial \log L_i}{\partial \beta}\right)^\top \frac{\partial \log L_i}{\partial \beta} \to Var(\beta)
```

"

# ╔═╡ 1508d920-6000-11eb-34bf-81e7d3ceb9bc
md"
In the version below, we use the `Distributions.jl` package, which means we could actually change to a binary probit just by swapping out the distribution from `Logistic` to `Normal`.

Alternatively, for lower-level control, we can use the following functions from [`StatsFuns.jl`](https://github.com/JuliaStats/StatsFuns.jl). This is useful for more computationally intensive work with multinomial discrete choice.

```julia
logsumexp      # log(exp(x) + exp(y)) or log(sum(exp(x)))
softmax        # exp(x_i) / sum(exp(x)), for i
```
"

# ╔═╡ b9c02010-5e9b-11eb-1e5d-9358c778cbba
# set seed for random # generator
seed!(1234)

# ╔═╡ b9a85250-5e9b-11eb-1787-6b671b235984
begin
	nobs = 1_000
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
# will throw an error if we goof
@assert all(cum_prob[:,2] .≈ 1)

# ╔═╡ dc87c800-5e9b-11eb-194c-4f7f0e83a5c5
# instead of simulating random type-1 extreme values, we just
# use a uniform variable and the CDF
y = [searchsortedfirst(row, rand()) for row in eachrow(cum_prob) ] .-1 

# ╔═╡ 694cc4c0-5e9c-11eb-3d00-458ca8c8fa28
countmap(y)

# ╔═╡ 90b963b0-5e9c-11eb-3e39-bdf160b8655e
function loglik(y, X, theta)
    n,k = size(X)
    ff(z) = logcdf(Logistic(), z)

    # see footnote 6 on p. 778 in Greene 6th ed for this shortcut
    q = 2 .* y .- 1
    u1 = X*theta
    LL = sum(ff.(q.*u1))    

    return -LL  # I *think* you'll need to flip sign to maximize
end

# ╔═╡ 90b963b0-5e9c-11eb-1d75-25729dfb57ee
# note that the `!` means we're updating the first argument(s)
function dloglik!(grad, y, X, theta)
    
	n,k = size(X)    
    u1 = X*theta
	
	# create function
	ff(z) = cdf(Logistic(), z)    

	# all the broadcasting fuses operations into a single
	# loop instead of allocating temp vectors
	# this can help w/ speed + memory
    grad .= -vec(sum( (y .- ff.(u1)) .* X; dims=1))
    return grad
end

# ╔═╡ 90b963b0-5e9c-11eb-0dea-81d683f42601
# wrapper to allocate gradient vector
dloglik(y, X, theta) = dloglik!(similar(theta), y, X, theta)

# ╔═╡ 90c65c02-5e9c-11eb-032a-7d600b60b9c4
function informationmatrix(y, X, theta)

	n,k = size(X)
    infomatrix = zeros(k,k)

    u1 = X*theta
    ff(z) = cdf(Logistic(), z)    
    g = y .- ff.(u1)   # as per Greene 6th ed p. 779
    
    infomatrix = (g .* X)' * (g .* X)
    
    return infomatrix # maybe flip signs?
end

# ╔═╡ 640b3fa2-5e9c-11eb-19d3-97ebb410851a
# closures wrap likelihood & gradient
begin
	f(thet) = loglik(y,X,thet)
	g!(grad,thet) = dloglik!(grad,y,X,thet)
end

# ╔═╡ ba5f02b0-5e9c-11eb-30cc-fbb3e2d4b6e2
# initial guess
theta0 = zeros(k)

# ╔═╡ ba5f02b0-5e9c-11eb-2221-4b0d5e525673
# Check gradient against finite difference
begin
	fdgrad = finite_difference_gradient(f, theta0, Val{:central})
	@assert  fdgrad ≈ dloglik(y,X,theta0)
	fdgrad .- dloglik(y,X,theta0)
end

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
# ╠═0ece0030-5e9d-11eb-0092-254fc6e72e83
# ╠═b7910ac0-5ffe-11eb-3925-13fbd88d3f67
# ╠═e2248c2e-5ffe-11eb-189c-85dbf7703168
# ╠═1508d920-6000-11eb-34bf-81e7d3ceb9bc
# ╠═8595f030-5e9b-11eb-3bd7-ef8250cd1fab
# ╠═b9c02010-5e9b-11eb-1e5d-9358c778cbba
# ╠═b9a85250-5e9b-11eb-1787-6b671b235984
# ╠═b976e210-5e9b-11eb-0b18-e95537e1d64f
# ╠═dc8270d0-5e9b-11eb-02cb-1faa58ce2656
# ╠═dc82bef0-5e9b-11eb-20f5-0d1ed55e9bb5
# ╠═dc87c800-5e9b-11eb-194c-4f7f0e83a5c5
# ╠═694cc4c0-5e9c-11eb-3d00-458ca8c8fa28
# ╠═90b963b0-5e9c-11eb-3e39-bdf160b8655e
# ╠═90b963b0-5e9c-11eb-1d75-25729dfb57ee
# ╠═90b963b0-5e9c-11eb-0dea-81d683f42601
# ╠═90c65c02-5e9c-11eb-032a-7d600b60b9c4
# ╠═640b3fa2-5e9c-11eb-19d3-97ebb410851a
# ╠═ba5f02b0-5e9c-11eb-30cc-fbb3e2d4b6e2
# ╠═ba5f02b0-5e9c-11eb-2221-4b0d5e525673
# ╠═ba614ca0-5e9c-11eb-3619-0507168a18d6
# ╠═61327d20-5e9c-11eb-3704-bf5055d67bb0
