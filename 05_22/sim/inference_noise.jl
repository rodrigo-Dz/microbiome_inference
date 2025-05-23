using CSV
using DataFrames
using DifferentialEquations
using Optim
using Statistics  # Importar el módulo para usar `mean`
using Distributions
using Plots

# Leer archivos
data = CSV.read("freq.csv", DataFrame)
meta = CSV.read("meta.csv", DataFrame)


function gillespie_LV()

    # Inicialización
    t = 0.0
    n = copy(n0)

    # Time series de salida
    timeseries = zeros(Int, t_points, n_types)

    # Estado inicial
    timeseries[1, :] .= n

    sample_index = 2  # Julia usa índices desde 1

    while t <= t_simulated

        # Tasas de transición
        T_up = (gR .+ I_p * n) .* n
        T_down = (I_n * n) .* n
        T_up_n_down = vcat(T_up, T_down)

        total_rate = sum(T_up_n_down)
        if total_rate == 0
            break
        end

        # Tiempo hasta el próximo evento
        time_par = 1.0 / total_rate
        choice_par = T_up_n_down .* time_par
        t_sampled = rand(Exponential(time_par))

        # Elección de reacción
        q = rand()
        p_sum = 0.0
        i = 1

        while i <= length(choice_par) && p_sum + choice_par[i] < q
            p_sum += choice_par[i]
            i += 1
        end

        # Muestreo
        while sample_index <= t_points && sampling_times[sample_index] < t + t_sampled
            timeseries[sample_index, :] .= n
            sample_index += 1
        end

        # Actualización del estado
        if i <= n_types
            n[i] += 1
        else
            n[i - n_types] -= 1
        end

        t += t_sampled
    end

    return timeseries
end

fn = include(string("FN.jl"));

# Número de tipos de especies
n_types = 3
# Tasas de crecimiento
gR = [0.9, 1.7, 1.3]
# Matriz de tasas de interacción (positiva y negativa)
I = 1e-5 .* [-4.8  2.7  -5.0;
             -2.9 -6.3  3.3;
              2.0 -4.1 -5.4]
# Separar en matrices positivas y negativas
I_p = zeros(n_types, n_types)
I_n = zeros(n_types, n_types)
for i in 1:n_types, j in 1:n_types
    if I[i, j] > 0
        I_p[i, j] = abs(I[i, j])
    elseif I[i, j] < 0
        I_n[i, j] = abs(I[i, j])
    end
end

# Poblaciones iniciales
n0 = [700, 7000, 12000]
# Tiempo total simulado
t_simulated = 72
# Número de puntos de muestreo
t_points = 4
# Tiempos de muestreo (uniformemente espaciados)
sampling_times = [0.0, 24.0, 48.0, 72.0]
# Número de réplicas (series temporales)
n_timeseries = 4



# Arreglo para guardar los datos
all_data_LV = zeros(Int, n_timeseries, t_points, n_types)


for i in 1:communities
    s = gillespie_LV()
    all_data_LV[i, :, :] = round.(Int, s)
end

print(all_data_LV)

# con datos a tiempos: [0, 24, 48, 72]
t_save = [0.0, 24.0, 48.0, 72.0]
C1 = Float64.(all_data_LV)

# Ajustar modelo LV
gR, I = fn.opt_lv(C1, t_save, 50000)



using Plots

# Supongamos que params_opt es el resultado de la optimización
# Desempaquetar parámetros: ejemplo para 3 especies
u0 = C1[1, 1, :]

# Simular con parámetros optimizados
sim_opt = fn.simulate_lv(u0, gR, I, (t_save[1], t_save[end]), t_save)

# Graficar datos reales y simulados
plot()
for i in 1:size(C1, 3)
    scatter!(t_save, C1[1, :, i], label="Datos individuo $i", markershape=:circle)
    plot!(t_save, sim_opt[:, i], label="Simulación individuo $i", lw=2)
end

xlabel!("Tiempo (hrs)")
ylabel!("Población")
title!("Simulación con parámetros optimizados vs Datos")

# Condiciones iniciales: toma primer experimento, primer tiempo
for j in 1:size(C1, 1)
    u0 = C1[j, 1, :]
    p = plot()
    sim_opt = fn.simulate_lv(u0, gR, I, (t_save[1], t_save[end]), t_save)
    for i in 1:size(C1, 3)
        scatter!(t_save, C1[j, :, i], label="Data$i", markershape=:circle)
        plot!(t_save, sim_opt[:, i], label="Sim$i", lw=2)
    end
    savefig(p, "poblacion_stoc$j.png")
end
# Simular con parámetros optimizados
