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

            # run unit tests for rcmnl package
            ]test rcmnl
            ```
