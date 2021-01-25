Contains 2 examples:

1. Standard Logit
    - Shows off [Pluto.jl](https://github.com/fonsp/Pluto.jl) workflow (like Jupyter, but harder to mess up)
    - To open
        + install Julia
        + Make sure relevant packages are installed

            ```julia
            ]add Pluto StatsFuns Optim Distributions DataFrames StatsBase ForwardDiff
            ```

        + Run Pluto
            
            ```julia
            using Pluto
            Pluto.run()
            ```
        + open [logit-example.jl](./logit-example.jl) in Pluto

2. Random Coef multinomial logit w/ 3 choices for balanced panel
    - Shows off package development / [Revise.jl](https://github.com/timholy/Revise.jl)-based workflow
    - Also shows off automagic differentiation in gradient-based optimization
    - Better to edit with an IDE like **[VS Code](https://www.julia-vscode.org/)** ~~or Atom~~
    - Might be cool to parallelize
    - To work on package

        + install Julia

            ```julia
            # clone package to your development directory
            # (intead of just `]add` ing the package)
            ]dev https://github.com/magerton/rcmnl.jl
            
            # activate package (like a virtual environment)
            ]activate rcmnl

            # install required packages (in `Project.toml`)
            ]instantiate

            # run `runtests.jl` which simulates & estimates the rcmnl model
            ]test rcmnl
            ```

3. Solving and Plotting an IVP

    - Shows VERY basics of
        + solving IVP using `DifferentialEquations.jl`
        + Plotting the IVP and the phase plane with `Plots.jl`
    - Will require you to install these two packages, which take a while to install AND precompile the first time you use them.

        ```julia
        ]add DifferentialEquations Plots
        ```
    - Open [ivp-plotting-example.jl](./ivp-plotting-example.jl)
