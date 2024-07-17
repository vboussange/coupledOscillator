#=
Run the coupled oscillator problem.

Authors:
- Victor Boussange, WSL Birmensdorf
- Sandro Truttmann
- Thomas Poulet
=#

# Import packages
using DiffEqOperators
using OrdinaryDiffEq
using LaTeXStrings
using ProgressMeter
using JLD2
using UnPack
using PyPlot
using Pkg; Pkg.instantiate()

function plot_PT(model_params, N, sol, main_dir)
    """
        plot_PT(model_params, N, sol, main_dir)
    Plot oscillators in PT space.

        Output: Figure
    """

    # Unpack sol values
    @unpack Le2, Gr2_M = model_params
    times1 = sol.t
    T2 = [sol.u[i][(N*3)+1:N*4] for i in range(1, length(sol.t))]
    T2_centre = [T2[k][Int(N/2)] for k in range(1, length(sol.t))]

    fig, ax = plt.subplots()

    ax.plot(times1, T2_centre, c="blue")

    # Set labels
    fig.suptitle("Le2=$Le2, Gr2_M=$Gr2_M", fontsize=12)
    ax.set_ylabel(L"T2_{centre}(t)")

    fig.tight_layout()

    # Save plots
    fig_filename = joinpath(main_dir, string("Plot_P-T_Le2=", Le2, "_Gr2M=", Gr2_M, ".png"))

    if ~isdir(main_dir)
        mkpath(main_dir)
    end
    fig.savefig(fig_filename)

end

function coupled_oscillators(;N, tspan, model_params)
    """
        coupled_oscillators(;N, tspan, model_params)

    Run the coupled oscillators.

    # Output 
    JLD2 file with the obtained solutions.
    """

    ## Pre-compute derivatives
    dx = 1.0 / (N + 1)
    x = range(0.0, 1.0, length=N + 2)

    # Differentiation stencil
    ord_deriv = 2
    ord_approx = 2
    D_x = CenteredDifference(1, ord_approx, dx, N) # First derivative
    Δ_x = CenteredDifference(ord_deriv, ord_approx, dx, N) # Second derivative

    ## Problem "constants"
    Ar = 40
    Tc = 360
    δ = 1e-3
    n = 3.8
    n2 = 10
    α = 0.65
    ΔE = 90e3
    μ1 = 8e-3
    μ2 = 8e-3
    Kc = 1e11
    ϕ_0 = 0.03

    # initial and boundary conditions
    T_0 = -100.0 # BC
    T_i = T_0 # initial value above BC
    Δp_0 = 0.0 # BC
    Δp_i = 0.0 # initial value above BC

    # physical constants
    R = 8.3144621 # J.mol^-1.K^-1
    M_cao = 0.056 # kg/mol
    M_caco3 = 0.1 # kg/mol
    M_co2 = 0.044 # kg/mol
    ρ_cao = 3.35e3 # kg/m3
    ρ_caco3 = 2.71e3 # kg/m3
    ρ_co2 = 1.e3 # kg/m3

    # convenience parameters
    η1 = (ρ_co2 / ρ_cao) * (M_cao / M_co2)
    η2 = (ρ_caco3 / ρ_cao) * (M_cao / M_caco3)
    ΔAr_c = ΔE / (R * Tc)

    # Boundary conditions
    bc_Δp = DirichletBC(Δp_0, Δp_0)
    bc_T = DirichletBC(T_0, T_0)

    # Preallocating variables used for preprocessing
    # ϕ (porosity) and s (partial solid ratio)
    ## Oscillator 1
    w_rel1 = zeros(N)
    s1 = zeros(N)
    s_pad1 = zeros(N+2)
    Δϕ_chem1 = zeros(N)
    ϕ1 = zeros(N)
    ϕ_pad1 = zeros(N+2)
    rhobar_s1 = zeros(N)
    ρ_m1 = zeros(N)
    q_z1 = zeros(N)
    
    # Oscillator 2
    w_rel2 = zeros(N)
    s2 = zeros(N)
    s_pad2 = zeros(N+2)
    Δϕ_chem2 = zeros(N)
    ϕ2 = zeros(N)
    ϕ_pad2 = zeros(N+2)
    rhobar_s2 = zeros(N)
    ρ_m2 = zeros(N)
    q_z2 = zeros(N)

    factor_s = 1.0 # 0.98


    ## Define model
    function f(du, u, p, t)
        dΔp1 = @view du[1:N]
        dT1 = @view du[N+1:2*N]
        dΔp2 = @view du[2*N+1:3*N]
        dT2 = @view du[3*N+1:end]
        Δp1 = @view u[1:N]
        T1 = @view u[N+1:2*N]
        Δp2 = @view u[2*N+1:3*N]
        T2 = @view u[3*N+1:end]
        @unpack Gr1, Le1, Gr2, Le2, Gr2_M, dTc1_m, dTc2_m, dTc1_M, dTc2_M = p

        # Couple T1 to Gr2
        dTc1 = (Δ_x*bc_T*T1+(Gr1*(1.0 .- Δp1) .^ n.*exp.(α * Ar ./ (1.0 .+ δ * T1)).-1.0).*exp.(Ar * δ * T1 ./ (1.0 .+ δ * T1)))[trunc(Int, (N + 1) / 2)]
        factor = (max(min(dTc1, dTc1_M), dTc1_m) - dTc1_m) / (dTc1_M - dTc1_m)
        Gr2u = Gr2 + factor * (Gr2_M - Gr2)

        # Oscillator 1
        ## calculating ϕ1 and q_z1 
        w_rel1 .= η2 * Kc .* exp.(-ΔAr_c ./ (1. .+ δ * T1))
        s1 .= w_rel1 ./ (1. .+ w_rel1)
        s_pad1 .= [s1[1];s1;s1[end]]
        Δϕ_chem1 .= (1. - ϕ_0) .* s1 ./ (s1 .+ η1)
        ϕ1 .= ϕ_0 .+ Δϕ_chem1
        ϕ_pad1 .= [ϕ1[1];ϕ1;ϕ1[end]]
        rhobar_s1 .= (1. .- s1)*ρ_caco3 .+ s1*ρ_cao
        ρ_m1 .= (1. .- ϕ1).*rhobar_s1 .+ ϕ1*ρ_co2
        q_z1 .= ((3. .- 2 .* ϕ1)./(ϕ1 .* (1. .- ϕ1))+(rhobar_s1 .- ρ_co2)./ρ_m1).*(D_x * ϕ_pad1) +
            (ϕ1 .* ρ_co2*(ρ_cao-ρ_caco3)./(rhobar_s1.*ρ_m1)).*(D_x * s_pad1)
        
        ## time derivatives
        dΔp1 .= Δ_x * bc_Δp * Δp1 / Le1 + q_z1 .* (D_x * bc_Δp * Δp1 / Le1) + μ1 * (1.0 .- ϕ1) .* (1 .- factor_s * s1) .* exp.(Ar * δ * T1 ./ (1.0 .+ δ * T1))
        dT1 .= Δ_x * bc_T * T1 + (Gr1 * (1.0 .- Δp1) .^ n .* exp.(α * Ar ./ (1.0 .+ δ * T1)) .- 1.0) .* exp.(Ar * δ * T1 ./ (1.0 .+ δ * T1))

        # Oscillator 2
        ## calculating ϕ2 and q_z2 
        w_rel2 .= η2 * Kc .* exp.(-ΔAr_c ./ (1. .+ δ * T2))
        s2 .= w_rel2 ./ (1. .+ w_rel2)
        s_pad2 .= [s2[1];s2;s2[end]]
        Δϕ_chem2 .= (1. - ϕ_0) .* s2 ./ (s2 .+ η1)
        ϕ2 .= ϕ_0 .+ Δϕ_chem2
        ϕ_pad2 .= [ϕ2[1];ϕ2;ϕ2[end]]
        rhobar_s2 .= (1. .- s2)*ρ_caco3 .+ s2*ρ_cao
        ρ_m2 .= (1. .- ϕ2).*rhobar_s2 .+ ϕ2*ρ_co2
        q_z2 .= ((3. .- 2 .* ϕ2)./(ϕ2 .* (1. .- ϕ2))+(rhobar_s2 .- ρ_co2)./ρ_m1).*(D_x * ϕ_pad2) + 
                (ϕ2 .* ρ_co2*(ρ_cao-ρ_caco3)./(rhobar_s2.*ρ_m2)).*(D_x * s_pad2)

        ## calculating time derivatives
        dΔp2 .= Δ_x * bc_Δp * Δp2 / Le2 + q_z2 .* (D_x * bc_Δp * Δp2 / Le2) + 
                μ2 * (1.0 .- ϕ2) .* (1. .- factor_s * s2) .* exp.(Ar * δ * T2 ./ (1.0 .+ δ * T2))
        dT2 .= Δ_x * bc_T * T2 + (Gr2u * (1.0 .- Δp2) .^ n2 .* exp.(α * Ar ./ (1.0 .+ δ * T2)) .- 1.0) .* exp.(Ar * δ * T2 ./ (1.0 .+ δ * T2))
        return du
    end

    # Initial condition as sine profile for both variables
    u0 = [Δp_i .+ (Δp_0 - Δp_i) .* sinpi.(collect(x))[2:end-1];
        T_i .+ (T_0 - T_i) .* sinpi.(collect(x))[2:end-1];
        Δp_i .+ (Δp_0 - Δp_i) .* sinpi.(collect(x))[2:end-1];
        T_i .+ (T_0 - T_i) .* sinpi.(collect(x))[2:end-1]]

    # Solve ODE
    print("Solving model for Le2=$(model_params.Le2), Gr2_M=$(model_params.Gr2_M) ")

    prob_true = ODEProblem(f, u0, tspan, model_params)
    @time sol = solve(prob_true, BS3(), reltol=1e-6,
        maxiters=Int(3e6)
    )
    n_times = length(sol.t)
    t_end = last(sol.t)
    println("Final time=$t_end, in $n_times timesteps")

    return sol
end

## Main model parameters
model_params = (Gr1 = 3.6e-8,
                Le1 = 2.,
                Gr2 = 6e-9, 
                Le2 = 7.5,
                Gr2_M = 6.6e-8, 
                dTc1_m = 5000, # min value of (dT1/dt)_c for which Gr2 starts to change
                dTc1_M = 6000, # min value of (dT1/dt)_c for which Gr2 reaches its max change
                dTc2_m = 4000, # min value of (dT2/dt)_c for which Gr1 reaches its minimum
                dTc2_M = 6000, # min value of (dT2/dt)_c for which Gr1 reaches its maximum
                )
tspan = (0.,90.)
N = 50 # number of collocation points

sol = coupled_oscillators(;N, tspan, model_params)
plot_PT(model_params, N, sol, "plots")

