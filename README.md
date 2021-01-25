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

4. Using [`JuMP`](https://github.com/jump-dev/JuMP.jl) to write down a constrained optimization problem

    - Uses a Jupyter notebook (can't use JuMP in Pluto yet)
    - JuMP is great for constrained optimization, can hook in to a BUNCH of solvers (Knitro, Ipopt, Gurobi, etc)
    - Uses automatic differentiation
    - Can use for BLP/Rust models with MPEC approach (ask Mark if you need example)
    - To run, switch to this directory, install IJulia, open up a jupyterlab
    
        ```julia
        ]add JuMP Ipopt SparseArrays
        ]add IJulia
        
        # might need to build IJulia
        ]build IJulia
        using IJulia
        jupyterlab(;dir="~/.julia/dev/rcmnl")
        ```
    - Making a jupyterlab desktop shortcut in Windows
        - [see blog post](https://medium.com/@kostal91/create-a-desktop-shortcut-for-jupyterlab-on-windows-9fcabcfa0d3f)
        - Right-click on the desktop and choose New -> Shortcut
        - Target: `%windir%\System32\cmd.exe "/K" C:\Users\%username%\.julia\conda\3\Scripts\activate.bat C:\Users\%username%\.julia\conda\3 & jupyter lab && exit`
        - Icon: `C:\Users\%username%\.julia\conda\3\Menu\jupyter.ico`
        - Start in: The directory where you put your ARE 254 stuff
    - To start Jupyterlab in its own window, add the line below to `%USERPROFILE%/.jupyter/jupyter_notebook_config.py`. See [blog post](http://christopherroach.com/articles/jupyterlab-desktop-app/)

