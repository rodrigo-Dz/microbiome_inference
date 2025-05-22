## Gillespie algorithm: Lotka-Volterra

def gillespie_LV():

    # initial condition
    
    t = 0
    
    n = np.copy(n0)
    
    # samples
    
    timeseries = np.zeros((t_points, n_types), dtype = int)
    
    # initial state
    
    timeseries[0, :] = n
    
    sample_index = 1
        
    while t <= t_simulated:
        
        # Transition rates
    
        T_up = (gR + np.dot(I_p, n)) * n
    
        T_down = np.dot(I_n, n) * n
        
        T_up_n_down = np.hstack((T_up, T_down))
        
        # Time to sample and propensities
        
        time_par = 1. / T_up_n_down.sum()
        
        choice_par = T_up_n_down * time_par
        
        t_sampled = np.random.exponential(time_par)
            
        q = random()
        
        p_sum = 0
        
        i = 0
        
        while p_sum + choice_par[i] < q:
            
            p_sum += choice_par[i]
            
            i += 1
            
        # sampling
            
        while sample_index < t_points and sampling_times[sample_index] < t + t_sampled:
            
            timeseries[sample_index] = n
                        
            sample_index += 1 
            
        # modify current state
            
        if i < n_types:
            
            n[i] += 1
            
        else:
            
            n[i - n_types] -= 1
            
        t += t_sampled
            
    return timeseries