### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ a9c917f0-37ec-11eb-39c1-8d9fec2fdaf1
# can take a while....
# might need to install DifferentialEquations & Plots with
# ]add Differential Equations Plots
begin
	using DifferentialEquations
	using Plots
	using Base.Iterators: product
end

# ╔═╡ 7866daf0-5eaa-11eb-2c25-afec8b81fb56
md"
## Investment problem

```math
\max_{i(t)} \int_0^T e^{-\rho t} \left\{p k(t) - c i(t) - \frac{d}{2} i(t)^2 \right\} dt
\qquad st \qquad
\dot k(t) = i(t) - \delta k(t)
```
given boundary conditions
```math
k(0) = k_0 \qquad k(T) = k_T \qquad T:\text{ given}
```

Necessary conditions
```math
\begin{align*}
\dot k &= i - \delta k \\
\dot \psi &= (\rho + \delta)\psi - p
\end{align*}
```
"

# ╔═╡ 6c39a320-5eaa-11eb-0729-0fdeb7f66e7f
# tell plots to use GR b/c it's fastest
gr();

# ╔═╡ bcf87142-5ea7-11eb-2359-115b8572fbe2
begin
	# parameters
	px=0.5
	c=3
	d=1
	ρ=0.07
	δ=0.05
	
	# steady state
	ssi = (px / (ρ+δ) - c) / d
	ssk = ssi/δ
end

# ╔═╡ 6b845912-37eb-11eb-3948-d513425270f2
begin
	doti(k,i) = (ρ+δ)*(i - ssi)
	dotk(k,i) = i - δ*k
	
	# solvers in DifferentialEquations.jl require functions that 
	# take the form   `f!(dy, y, params, t)`
	function dot_ki!(dy, y, parm, t)
		k,i = y
		dy[1] = dotk(k,i)
		dy[2] = doti(k,i)
		return dy
	end
end

# ╔═╡ 25be1680-37f2-11eb-2c06-99283062577c
# create a "mesh"
begin
	# grid of points in K, I space
	kspace = range(ssk/2, stop = 1.5*ssk, length=15)
	ispace = range(ssi/2, stop = 1.5*ssi, length=15)
	
	kispace = product(kspace, ispace)
	KK = [k for (k,i) in kispace]
	II = [i for (k,i) in kispace]

	SCALE = 2
	dotII = [doti(k,i) for (k,i) in kispace].*SCALE
	dotKK = [dotk(k,i) for (k,i) in kispace].*SCALE
end

# ╔═╡ 001a0000-5ea9-11eb-1a57-29544ffeb338
begin
	plt_quiv = plot(;legend=:bottomright, xlabel="\$k\$", ylabel="\$i\$")

	# motion vectors
	# note, use `vec` b/c need to input a vector
	quiver!(plt_quiv, vec(KK), vec(II); quiver=(vec(dotKK),vec(dotII)), alpha=0.2)

	# steady state
	# also note that we have to give vectors to scatter
	scatter!(plt_quiv, [ssk],[ssi], label="SS")

	# nullclines
	plot!(plt_quiv, kspace, k -> k*δ, label="\$\\dot k = 0\$")
	plot!(plt_quiv, kspace, k -> ssi, label="\$\\dot i = 0\$")
end

# ╔═╡ d412fe80-37f2-11eb-3c13-87530a5aa318
begin
	# initial conditions
	# (k₀ i₀)
	y0 = [12.0, 1.19]
	
	# time-span for problem
	# NOTE: this MUST be in Floats, not Ints!
	tmax = 25.0
	tspan = (0.0, tmax)
	
	# define a problem
	prob = ODEProblem(dot_ki!, y0, tspan);
end

# ╔═╡ e9758ae2-37f2-11eb-2299-c17952d5c072
# solve the problem using the `Tsit5()` method
# Note how much better our tolerances are
sol = solve(prob, Tsit5())

# ╔═╡ 587a34e0-5ea9-11eb-09ee-b9c836837819
begin
	TT = 0 : 0.1 : tmax
	kpath = first.(sol.(TT))
	ipath = last.(sol.(TT))
	
	# need to copy plot b/c of updating in Pluto
	pltq = deepcopy(plt_quiv)
	plot!(pltq, kpath, ipath; label="\$(k_t, i_t)\$")
end

# ╔═╡ faaf5290-5ea9-11eb-3761-7db74d6930e8
plot(
    plot(0:0.1:tmax, t -> sol(t)[1], title="\$k(t)\$", legend=:none),
    plot(0:0.1:tmax, t -> sol(t)[2], title="\$i(t)\$", legend=:none),
    plot(
        sol, 
        labels=["\$k(t)\$" "\$i(t)\$"], # NOTE: labels are row matrix!
        legend=:right, 
        title="plot(sol)"
    ),
    layout=(1,3)
)

# ╔═╡ Cell order:
# ╟─7866daf0-5eaa-11eb-2c25-afec8b81fb56
# ╠═a9c917f0-37ec-11eb-39c1-8d9fec2fdaf1
# ╠═6c39a320-5eaa-11eb-0729-0fdeb7f66e7f
# ╠═bcf87142-5ea7-11eb-2359-115b8572fbe2
# ╠═6b845912-37eb-11eb-3948-d513425270f2
# ╠═25be1680-37f2-11eb-2c06-99283062577c
# ╠═001a0000-5ea9-11eb-1a57-29544ffeb338
# ╠═d412fe80-37f2-11eb-3c13-87530a5aa318
# ╠═e9758ae2-37f2-11eb-2299-c17952d5c072
# ╠═587a34e0-5ea9-11eb-09ee-b9c836837819
# ╠═faaf5290-5ea9-11eb-3761-7db74d6930e8
