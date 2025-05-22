using CSV
using DataFrames
using DifferentialEquations
using Optim
using Statistics  # Importar el módulo para usar `mean`
using Plots

fn = include(string("FN.jl"));

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
    s = fn.simulate_lv_safe(n0, r, A, (t_save[1], t_save[end]), t_save)
    all_data_LV[i, :, :] = round.(Int, s)
end

print(all_data_LV)


# con datos a tiempos: [0, 24, 48, 72]
t_save = [0.0, 24.0, 48.0, 72.0]
C1 = Float64.(all_data_LV)

# Ajustar modelo LV
gR, I = fn.opt_lv(C1, t_save, 10000)



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
    savefig(p, "poblacion_$j.png")
end
# Simular con parámetros optimizados

# Graficar datos reales y simulados




