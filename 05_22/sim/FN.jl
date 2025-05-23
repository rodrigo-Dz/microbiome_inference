module fn
    using CSV
    using DataFrames
    using DifferentialEquations
    using Optim
    using Statistics  # Importar el módulo para usar `mean`
    using Distributions
    using Plots
    using DelimitedFiles;


    # Modelo Lotka-Volterra
    function lv_ode!(du, u, p, t)
        r, A = p
        du .= u .* (r .+ A * u)
    end

    function simulate_lv_safe(u0, r, A, tspan, t_save)
        try
            r = simulate_lv(u0, r, A, tspan, t_save)
            print(r)
            return r
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
        total_loss = 0.0
        losses = zeros(size(C1, 1))
        for exp_id in 1:size(C1, 1)
            u0 = C1[exp_id, 1, :]
            sim = simulate_lv_safe(u0, r, A, (t_save[1], t_save[end]), t_save)
    
            if any(!isfinite, sim)
                losses[exp_id] = 500000  # Penalización fuerte por simulación fallida
            else
                losses[exp_id] = mean(abs.(sim .- C1[exp_id, :, :]))
            end
        end
    
        return mean(losses)
    end
    
#=
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
=#

    # Optimización de parámetros r y A
    function opt_lv(C1::Array{Float64,3}, t_save::Vector{Float64}, iters, com)
        open(string("./OUT_",com,".txt"), "w") do io

        n = size(C1, 3)
        gR0 = ones(n) .+ 10
        I0 = zeros(n, n)
        for i in 1:n
            I0[i, i] = -10
        end
        I0 .*= 1e-5
        # Parámetros iniciales aleatorios
        p0 = vcat(gR0, vec(I0))
        loss0 = lv_loss(p0, C1, t_save)
        loss_history = [loss0]
        r0 = zeros(length(p0))

        writedlm(io, [vcat(p0, loss0)],'\t')

        println("Initial loss: ", loss0)
        i = 0
        while i < iters
            gR1 = gR0 + 0.01 * randn(n)
            I1 = I0 + 0.0001 * randn(n, n)
            p1 = vcat(gR1, vec(I1)); # Copy of the initial parameters

            println(i)

            loss1 = lv_loss(p1, C1, t_save)
            dloss = loss1 - loss0
            T = 0.001
            c1 = loss1 < loss0 || rand() < exp(-dloss / T)

            if c1
                gR0 = gR1
                I0 = I1
                loss0 = loss1
                p0 = p1
                println("ok")
            else
                # If not, revert to previous parameter values

                println("no ok")
            end
            writedlm(io, [vcat(p0, loss0)],'\t')
            i += 1
        end

        gR_final = p0[1:n]
        I_final = reshape(p0[n+1:end], (n, n))
        println("Final loss: ", loss0)
        println("Final gR: ", gR_final)
        println("Final I: ", I_final)
        return gR_final, I_final
        end
    end
end