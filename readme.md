This repository contains the code for the paper 
> *Chaotic Slow Slip Events in New Zealand from two coupled slip patches: a proof of concept*, Poulet, T., Truttmann, S., Boussange, V., Veveakis, M., (2024).

# Getting started
1. Install [`juliaup`](https://github.com/JuliaLang/juliaup).
2. Install Julia 1.8.5:
```
juliaup add 1.8.5
juliaup default 1.8.5
```
3. Within the root folder, execute the script:
```
julia --project=. runCoupledOscillators.jl
```