using CSV
using DataFrames
using DifferentialEquations
using Optim
using Statistics  # Importar el módulo para usar `mean`
using Distributions
using Plots


fn = include(string("FN.jl"));


# Leer archivos
data = CSV.read("freq.csv", DataFrame)
meta = CSV.read("meta.csv", DataFrame)

# Inicialización
communities = 4
t_points = 4
n_types = 10

# Arreglo para guardar los datos
all_data_L = zeros(Int, communities, t_points, n_types)

# Filtrar comunidad R1 y tiempo = 0
exp_n = filter(row -> row.community == "R1", meta)
pop_i = filter(row -> row.hrs == 0, exp_n)
println(pop_i)

names = pop_i[!, "Column1"]
com = select(data, names...)

# Llenar datos t=0
all_data_L[1, 1, :] = com[:, 1]
all_data_L[2, 1, :] = com[:, 2]
all_data_L[3, 1, :] = com[:, 1]
all_data_L[4, 1, :] = com[:, 2]

println("Después de llenar t=0:")
println(all_data_L)


# Filtrar tiempo > 0
exp_28a = filter(row -> row.temp == "28" && row.exp == "NS1", exp_n)
exp_28b = filter(row -> row.temp == "28" && row.exp == "NS3", exp_n)
exp_32a = filter(row -> row.temp == "32" && row.exp == "NS1", exp_n)
exp_32b = filter(row -> row.temp == "32" && row.exp == "NS3", exp_n)


names28a = exp_28a[!, "Column1"]
names28b = exp_28b[!, "Column1"]
names32a = exp_32a[!, "Column1"]
names32b = exp_32b[!, "Column1"]

com28a = select(data, names28a...)
com28b = select(data, names28b...)
com32a = select(data, names32a...)
com32b = select(data, names32b...)


all_data_L[1, 2, :] = com28a[:, 1]
all_data_L[1, 3, :] = com28a[:, 2]
all_data_L[1, 4, :] = com28a[:, 3]


all_data_L[2, 2, :] = com28b[:, 1]
all_data_L[2, 3, :] = com28b[:, 2]
all_data_L[2, 4, :] = com28b[:, 3]


all_data_L[3, 2, :] = com32a[:, 1]
all_data_L[3, 3, :] = com32a[:, 2]
all_data_L[3, 4, :] = com32a[:, 3]


all_data_L[4, 2, :] = com32b[:, 1]
all_data_L[4, 3, :] = com32b[:, 2]
all_data_L[4, 4, :] = com32b[:, 3]

println(all_data_L[:,:,:])
n_types = 5
selected_indices = [1, 2, 3, 6, 10]
C1 = all_data_L[:, :, selected_indices]


# con datos a tiempos: [0, 24, 48, 72]
t_save = [0.0, 24.0, 48.0, 72.0]
C1 = Float64.(C1)

# Ajustar modelo LV
gR, I = fn.opt_lv(C1, t_save, 50000, "C01")



using Plots


# Condiciones iniciales: toma primer experimento, primer tiempo
for j in 1:size(C1, 1)
    u0 = C1[j, 1, :]
    p = plot()
    sim_opt = fn.simulate_lv(u0, gR, I, (t_save[1], t_save[end]), t_save)
    for i in 1:size(C1, 3)
        scatter!(t_save, C1[j, :, i], label="Data$i", markershape=:circle)
        plot!(t_save, sim_opt[:, i], label="Sim$i", lw=2)
    end
    savefig(p, "C1_poblacion_$j.png")
end
# Simular con parámetros optimizados

# Graficar datos reales y simulados





