using CSV
using DataFrames
using DifferentialEquations
using Optim


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
all_data_L[2, 1, :] = com[:, 1]
all_data_L[3, 1, :] = com[:, 2]
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

println(all_data_L)

all_data_L[3, 2, :] = com32a[:, 1]
all_data_L[3, 3, :] = com32a[:, 2]
all_data_L[3, 4, :] = com32a[:, 3]


all_data_L[4, 2, :] = com32b[:, 1]
all_data_L[4, 3, :] = com32b[:, 2]
all_data_L[4, 4, :] = com32b[:, 3]

println(all_data_L[:,:,:])
selected_indices = [1, 2, 3, 6, 10]
C1 = all_data_L[:, :, selected_indices]
#=

# Filtrar comunidad R1 y tiempo = 0
exp_n = filter(row -> row.community == "R2", meta)
pop_i = filter(row -> row.hrs == 0, exp_n)
println(pop_i)
names = pop_i[!, "Column1"]
com = select(data, names...)
# Llenar datos t=0
all_data_L[5, 1, :] = com[:, 1]
all_data_L[6, 1, :] = com[:, 1]
all_data_L[7, 1, :] = com[:, 2]
all_data_L[8, 1, :] = com[:, 2]
println("Después de llenar t=0:")
println(all_data_L)
# Filtrar tiempo > 0
exp_28a = filter(row -> row.temp == "28" && row.exp == "NS1", exp_n)
exp_28b = filter(row -> row.temp == "28" && row.exp == "NS3", exp_n)
exp_32a = filter(row -> row.temp == "32" && row.exp == "NS1", exp_n)
exp_32b = filter(row -> row.temp == "32" && row.exp == "NS3", exp_n)
names28a = exp_28a[!, "Column1"]
if length(names28a) != 3
    all_data_L[5, 1, :] = [0.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]
else
    com28a = select(data, names28a...)
    all_data_L[5, 2, :] = com28a[:, 1]
    all_data_L[5, 3, :] = com28a[:, 2]
    all_data_L[5, 4, :] = com28a[:, 3]
end
names28b = exp_28b[!, "Column1"]
if length(names28a) != 3
    all_data_L[6, 1, :] = [0.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]
else 
    com28b = select(data, names28b...)
    all_data_L[6, 2, :] = com28b[:, 1]
    all_data_L[6, 3, :] = com28b[:, 2]
    all_data_L[6, 4, :] = com28b[:, 3]
end

names32a = exp_32a[!, "Column1"]
if length(names28a) != 3
    all_data_L[7, 1, :] = [0.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]
else 
    com32a = select(data, names32a...)
    all_data_L[7, 2, :] = com32a[:, 1]
    all_data_L[7, 3, :] = com32a[:, 2]
    all_data_L[7, 4, :] = com32a[:, 3]
end
names32b = exp_32b[!, "Column1"]
if length(names28a) != 3
    all_data_L[8, 1, :] = [0.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]
else
    com32b = select(data, names32b...)
    all_data_L[8, 2, :] = com32b[:, 1]
    all_data_L[8, 3, :] = com32b[:, 2]
    all_data_L[8, 4, :] = com32b[:, 3]
    
end



# Filtrar comunidad R1 y tiempo = 0
exp_n = filter(row -> row.community == "R3", meta)
pop_i = filter(row -> row.hrs == 0, exp_n)
println(pop_i)
names = pop_i[!, "Column1"]
com = select(data, names...)
# Llenar datos t=0
all_data_L[9, 1, :] = com[:, 1]
all_data_L[10, 1, :] = com[:, 1]
all_data_L[11, 1, :] = com[:, 2]
all_data_L[12, 1, :] = com[:, 2]
println("Después de llenar t=0:")
println(all_data_L)
# Filtrar tiempo > 0
exp_28a = filter(row -> row.temp == "28" && row.exp == "NS1", exp_n)
exp_28b = filter(row -> row.temp == "28" && row.exp == "NS3", exp_n)
exp_32a = filter(row -> row.temp == "32" && row.exp == "NS1", exp_n)
exp_32b = filter(row -> row.temp == "32" && row.exp == "NS3", exp_n)
names28a = exp_28a[!, "Column1"]
if length(names28a) != 3
    all_data_L[9, 1, :] = [0.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]
else
    com28a = select(data, names28a...)
    all_data_L[9, 2, :] = com28a[:, 1]
    all_data_L[9, 3, :] = com28a[:, 2]
    all_data_L[9, 4, :] = com28a[:, 3]
end
names28b = exp_28b[!, "Column1"]
if length(names28a) != 3
    all_data_L[10, 1, :] = [0.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]
else 
    com28b = select(data, names28b...)
    all_data_L[10, 2, :] = com28b[:, 1]
    all_data_L[10, 3, :] = com28b[:, 2]
    all_data_L[10, 4, :] = com28b[:, 3]
end

names32a = exp_32a[!, "Column1"]
if length(names28a) != 3
    all_data_L[11, 1, :] = [0.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]
else 
    com32a = select(data, names32a...)
    all_data_L[11, 2, :] = com32a[:, 1]
    all_data_L[11, 3, :] = com32a[:, 2]
    all_data_L[11, 4, :] = com32a[:, 3]
end
names32b = exp_32b[!, "Column1"]
if length(names28a) != 3
    all_data_L[10, 1, :] = [0.0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]
else
    com32b = select(data, names32b...)
    all_data_L[12, 2, :] = com32b[:, 1]
    all_data_L[12, 3, :] = com32b[:, 2]
    all_data_L[12, 4, :] = com32b[:, 3]
    
end
=#

println(all_data_L[:,:,:])





using DifferentialEquations
using Optim

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
    sol = solve(prob, Tsit5(); saveat=t_save, reltol=1e-8, abstol=1e-8)
    
    # Reconstruir matriz de soluciones para los tiempos exactos
    sim = zeros(length(t_save), length(u0))
    for (i, t) in enumerate(t_save)
        sim[i, :] = sol(t)
    end
    
    return sim
end


function lv_loss(params_flat, C1, t_save)
    r, A = unpack_params(params_flat)
    loss = 0.0
    for exp_id in 1:size(C1, 1)
        u0 = C1[exp_id, 1, :]
        sim = simulate_lv_safe(u0, r, A, (t_save[1], t_save[end]), t_save)
        if size(sim) != size(C1[exp_id, :, :])
            error("La simulación no tiene el mismo tamaño que los datos.")
        end
        loss += sum(abs.(sim .- C1[exp_id, :, :]))
    end
    println("Evaluated loss: ", loss)
    return loss
end


# Optimización de parámetros r y A
function fit_lv_to_data(C1::Array{Float64,3}, t_save::Vector{Float64})
    n = size(C1, 3)

    # Parámetros iniciales aleatorios
    r0 = randn(n)
    A0 = randn(n, n) * 0.01

    params0 = vcat(r0, vec(A0))

    loss_func = p -> lv_loss(p, C1, t_save)

    result = optimize(
        p -> lv_loss(p, C1, t_save),
        params0,
        NelderMead(),
        Optim.Options(iterations=1000, show_trace=true, f_tol=1e-6)
    )
    
    r_opt = result.minimizer[1:n]
    A_opt = reshape(result.minimizer[n+1:end], n, n)

    return r_opt, A_opt, result
end


# Suponiendo que ya tienes C1 cargado: tamaño [experimentos, 4, especies]
# con datos a tiempos: [0, 24, 48, 72]
t_save = [0.0, 24.0, 48.0, 72.0]
C1 = Float64.(C1)

# Ajustar modelo LV
r_opt, A_opt, result = fit_lv_to_data(C1, t_save)

println("r óptimo: ", r_opt)
println("A óptimo: ", A_opt)
println("Error final: ", result.minimum)




using Plots

# Supongamos que params_opt es el resultado de la optimización
# Desempaquetar parámetros: ejemplo para 3 especies


# Condiciones iniciales: toma primer experimento, primer tiempo
u0 = C1[1, 1, :]

# Simular con parámetros optimizados
sim_opt = simulate_lv(u0, r_opt, A_opt, (t_save[1], t_save[end]), t_save)

# Graficar datos reales y simulados
plot()
for i in 1:size(C1, 3)
    scatter!(t_save, C1[1, :, i], label="Datos individuo $i", markershape=:circle)
    plot!(t_save, sim_opt[:, i], label="Simulación individuo $i", lw=2)
end

xlabel!("Tiempo (hrs)")
ylabel!("Población")
title!("Simulación con parámetros optimizados vs Datos")

