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

function unpack_params(p, n_types)
    n = n_types
    r = p[1:n]
    A = reshape(p[n+1:end], (n, n))
    return r, A
end

function lv_loss(params_flat, C1, t_save)
    r, A = unpack_params(params_flat, size(C1, 3))
    losses = zeros(size(C1, 1))  # Arreglo para almacenar el error promedio de cada experimento
    for exp_id in 1:size(C1, 1)
        u0 = C1[exp_id, 1, :]
        sim = simulate_lv_safe(u0, r, A, (t_save[1], t_save[end]), t_save)

        losses[exp_id] = mean(abs.(sim .- C1[exp_id, :, :]))
    end
    return mean(losses)
end


# Optimización de parámetros r y A
function opt_lv(C1::Array{Float64,3}, t_save::Vector{Float64})
    n = size(C1, 3)
    # Parámetros iniciales aleatorios
    gR0 = [1,1,1,1,1]
    I0 =  1E-5 * [-1 0 0 0 0; 0 -1 0 0 0; 0 0 -1 0 0; 0 0 0 -1 0; 0 0 0 0 -1]     # Distribución normal con media 0 y desviación estándar 5
    p0 = vcat(gR0, vec(I0))
    loss0 = lv_loss(p0, C1, t_save)
    loss_history = [loss0]
    r0 = zeros(length(p0))

    println("Initial loss: ", loss0)
    i = 0
    while i < 5000
        rI = rand(MvNormal(zeros(length(p0)), zeros(length(p0)) .+ fill(0.0001, length(p0)))); # Random values to update parameters
        p1 = copy(p0); # Copy of the initial parameters
        for pI in 1:length(p1)  # Update parameter values
            r0[pI] = p1[pI]; # Save previous value
            p1[pI] += rI[pI]; # Update value
        end

        println(i)

        loss1 = lv_loss(p1, C1, t_save)

        xiC = (loss0 ^ 2) / (2 * 0.083);
		xiP = (loss1 ^ 2) / (2 * 0.083);

        c1 = rand() < exp(xiC - xiP)
        if c1
            p0 = p1
            loss0 = loss1
        end
        i += 1
    end

    gR_final = p0[1:n]
    I_final = reshape(p0[n+1:end], (n, n))
    println("Final loss: ", loss0)
    println("Final gR: ", gR_final)
    println("Final I: ", I_final)
    return gR_final, I_final
end



# Inicialización
communities = 4
t_points = 4
n_types = 5


# con datos a tiempos: [0, 24, 48, 72]
t_save = [0.0, 24.0, 48.0, 72.0]
C1 = Float64.(C1)

# Ajustar modelo LV
gR, I = opt_lv(C1, t_save)



using Plots


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




