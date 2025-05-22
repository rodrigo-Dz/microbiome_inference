using CSV
using DataFrames
using DifferentialEquations
using Optim
using Statistics  # Importar el módulo para usar `mean`

using Plots


# Modelo Lotka-Volterra
function lv_ode!(du, u, p, t)
    r, A = p
    du .= u .* (r .+ A * u)
end

function simulate_lv_safe(u0, r, A, tspan, t_save)
    try
        return simulate_lv(u0, r, A, tspan, t_save)
    catch e
        println("Simulation failed: ", e)
        return fill(Inf, length(t_save), length(u0))
    end
end

# Simulación desde condiciones iniciales, evaluando solo en t_save
function simulate_lv(u0, r, A, tspan, t_save)
    prob = ODEProblem(lv_ode!, u0, tspan, (r, A))
    sol = solve(prob, Rodas5(); saveat=t_save, reltol=1e-8, abstol=1e-8)
    
    # Reconstruir matriz de soluciones para los tiempos exactos
    sim = zeros(length(t_save), length(u0))
    for (i, t) in enumerate(t_save)
        sim[i, :] = sol(t)
    end
    
    return sim
end

function unpack_params(params)
    n = 3
    r = params[1:n]
    A = reshape(params[n+1:end], (n, n))
    return r, A
end

function lv_loss(params_flat, C1, t_save)
    r, A = unpack_params(params_flat)
    losses = zeros(size(C1, 1))  # Arreglo para almacenar el error promedio de cada experimento
    for exp_id in 1:size(C1, 1)
        u0 = C1[exp_id, 1, :]
        println("u0: ", u0)
        sim = simulate_lv_safe(u0, r, A, (t_save[1], t_save[end]), t_save)

        losses[exp_id] = mean(abs.(sim .- C1[exp_id, :, :]))
    end
    return mean(losses)
end


# Optimización de parámetros r y A
function opt_lv(C1::Array{Float64,3}, t_save::Vector{Float64})
    n = size(C1, 3)
    # Parámetros iniciales aleatorios
    gR0 = [1,1,1]
    I0 =  1E-5 * [0 0 0; 0 0 0; 0 0 0]     # Distribución normal con media 0 y desviación estándar 5
    params0 = vcat(gR0, vec(I0))
    loss0 = lv_loss(params0, C1, t_save)
    loss_history = [loss0]

    println("Initial loss: ", loss0)
    i = 0
    while i < 50000
        println("Iteration: ", i)
        gR1 = gR0 + 0.1 * randn(n)
        I1 = I0 + 0.001 * randn(n, n)
        params1 = vcat(gR1, vec(I1))
        loss1 = lv_loss(params1, C1, t_save)
        if loss1 < loss0
            println("Updating parameters")
            gR0 = gR1
            I0 = I1
            loss0 = loss1
            append!(loss_history, loss0)

        end
        i += 1
    end
    plot()
    plot(loss_history, label="Loss", xlabel="Iteration", ylabel="Loss", title="Optimization Progress", lw=2)
    
    println("Final loss: ", loss0)
    println("Final gR: ", gR0)
    println("Final I: ", I0)
    return gR0, I0
end



# Inicialización
communities = 4
t_points = 4
n_types = 3

# Arreglo para guardar los datos
all_data_LV = zeros(Int, communities, t_points, n_types)

# simulacion
r = [0.9, 1.7, 1.3]
A = 1E-5 * [-4.8 2.7 -5.0; -2.9 -6.3 3.3; 2.0 -4.1 -5.4]

t_save = [0.0, 24.0, 48.0, 72.0]

for i in 1:communities
    n0 = rand(500:15000, n_types)
    s = simulate_lv_safe(n0, r, A, (t_save[1], t_save[end]), t_save)
    all_data_LV[i, :, :] = round.(Int, s)
end

print(all_data_LV)


# con datos a tiempos: [0, 24, 48, 72]
t_save = [0.0, 24.0, 48.0, 72.0]
C1 = Float64.(all_data_LV)

# Ajustar modelo LV
gR, I = opt_lv(C1, t_save)



using Plots

# Supongamos que params_opt es el resultado de la optimización
# Desempaquetar parámetros: ejemplo para 3 especies
u0 = C1[1, 1, :]

# Simular con parámetros optimizados
sim_opt = simulate_lv(u0, gR, I, (t_save[1], t_save[end]), t_save)


# Condiciones iniciales: toma primer experimento, primer tiempo
for j in 1:size(C1, 1)
    u0 = C1[j, 1, :]
    p = plot()
    sim_opt = simulate_lv(u0, gR, I, (t_save[1], t_save[end]), t_save)
    for i in 1:size(C1, 3)
        scatter!(t_save, C1[j, :, i], label="Data$i", markershape=:circle)
        plot!(t_save, sim_opt[:, i], label="Sim$i", lw=2)
    end
    savefig(p, "poblacion_$j.png")
end
# Simular con parámetros optimizados

# Graficar datos reales y simulados




