import matplotlib.pyplot as mp
import numpy as np
import scipy as sp
import pickle as pc

from random import random
from datetime import timedelta

from pyabc import ABCSMC, RV, Distribution, MultivariateNormalTransition, QuantileEpsilon, LocalTransition, History
from pyabc.transition import silverman_rule_of_thumb
from pyabc.populationstrategy import AdaptivePopulationSize, ListPopulationSize
from pyabc.visualization import plot_kde_matrix
from pyabc.sampler import SingleCoreSampler

import matplotlib.lines as mlines

from matplotlib import rc
rc('xtick', labelsize=14)
rc('ytick', labelsize=14)
mp.rcParams['figure.figsize'] = (4, 4)



## Generate data (absolute and relative abundance)

def compute_moments(model, data):
    n_timeseries = 4
    # Datasets to store moments
        
    abs_abund_timeseries_df = []

    rel_abund_timeseries_df = []

    # Make Gillespie simulations
    ceros = 0
    for sample in range(0,4):
        sam = data[sample,:,:]

        # Absolute abundance
        if model == 'logistic':
            abs_abund_timeseries = pd.DataFrame(data = sam, index=['t_%i'%i for i in range(t_points)], columns=['type_%i'%i for i in range(n_types)])
        
        elif model == 'lotka-volterra':
            abs_abund_timeseries = pd.DataFrame(data = sam, index=['t_%i'%i for i in range(t_points)], columns=['type_%i'%i for i in range(n_types)])
        
        abs_abund_timeseries_df.append(abs_abund_timeseries)
        
        # Relative abundance

        rel_abund_timeseries = abs_abund_timeseries.div(abs_abund_timeseries.sum(axis=1), axis=0)

        rel_abund_timeseries_df.append(rel_abund_timeseries)

    n_timeseries = n_timeseries - ceros       
    # First moment for each type as a vector

    m_k_abs_abund = sum(abs_abund_timeseries_df) / n_timeseries

    m_k_rel_abund = sum(rel_abund_timeseries_df) / n_timeseries

    # Co-moment for types k and j as a matrix

    cm_kj_abs_abund = np.zeros((t_points, n_types, n_types))

    cm_kj_rel_abund = np.zeros((t_points, n_types, n_types))

    for t_point in range(t_points):

        for ts_index in range(n_timeseries):

            cm_kj_abs_abund[t_point,:,:] += np.outer(abs_abund_timeseries_df[ts_index].iloc[t_point,:], abs_abund_timeseries_df[ts_index].iloc[t_point,:])

            cm_kj_rel_abund[t_point,:,:] += np.outer(rel_abund_timeseries_df[ts_index].iloc[t_point,:], rel_abund_timeseries_df[ts_index].iloc[t_point,:])

    cm_kj_abs_abund /= n_timeseries

    cm_kj_rel_abund /= n_timeseries

    # Moments of scaling factor

    m_Sigma = np.zeros(t_points)

    for ts_index in range(n_timeseries):

        m_Sigma += abs_abund_timeseries_df[ts_index].sum(1)

    m_Sigma /= n_timeseries

    # Create an array of empirical moments (absolute abundance)
    
    data_abs_abund = np.zeros((t_points, n_types + n_types**2))
    data_abs_abund[:,:n_types] = m_k_abs_abund
    for t_point in range(t_points):
        data_abs_abund[t_point,n_types:] = cm_kj_abs_abund[t_point].reshape(1, n_types**2)[0]

    init_abs_abund = data_abs_abund[0,:]
    data_abs_abund = {"moments": data_abs_abund}

    # Create an array of empirical moments (relative abundance)

    data_rel_abund = np.zeros((t_points, n_types + n_types**2))
    data_rel_abund[:,:n_types] = m_k_rel_abund
    for t_point in range(t_points):
        data_rel_abund[t_point,n_types:] = cm_kj_rel_abund[t_point].reshape(1, n_types**2)[0]

    init_rel_abund = data_rel_abund[0,:]
    data_rel_abund = {"moments": data_rel_abund}
    
    return data_abs_abund, data_rel_abund, m_Sigma, init_abs_abund, init_rel_abund


import pandas as pd
import numpy as np
# Cargar el archivo CSV
data = pd.read_csv('freq.csv')
meta = pd.read_csv('meta.csv')

temp = 2
exp = 2
comunities = 4
t_points = 4
n_types = 10

all_data_L = np.zeros((comunities, t_points, n_types), dtype=int)

exp_n = meta[meta["community"] == "R1"]
pop_i = exp_n[exp_n['hrs'] == 0]
print(pop_i)
names = pop_i['Unnamed: 0'].tolist()
com = data[names]


all_data_L[0,0,:] = com.iloc[:,0]
all_data_L[1,0,:] = com.iloc[:,0]
all_data_L[2,0,:] = com.iloc[:,1]
all_data_L[3,0,:] = com.iloc[:,1]

print(all_data_L)

ceros = 0

exp_n = exp_n[exp_n['hrs'] > 0]

exp_28 = exp_n[exp_n['temp'] == 28]
exp_28_a = exp_28[exp_28["exp"] == "NS1"]
exp_28_b = exp_28[exp_28["exp"] == "NS3"]

exp_32 = exp_n[exp_n['temp'] == 32]
exp_32_a = exp_32[exp_32["exp"] == "NS1"]
exp_32_b = exp_32[exp_32["exp"] == "NS3"]


names_28a = exp_28_a['Unnamed: 0'].tolist()
names_28b = exp_28_b['Unnamed: 0'].tolist()

names_32a = exp_32_a['Unnamed: 0'].tolist()
names_32b = exp_32_b['Unnamed: 0'].tolist()

# Obtener los datos correspondientes de 'data'
com_28a = data[names_28a]
print(com_28a)
com_28b = data[names_28b]
com_32a = data[names_32a]
com_32b = data[names_32b]


if len(names_28a) != 3:
    all_data_L[0, :, :] = np.zeros((t_points, n_types), dtype=int)
    ceros = ceros + 1
else:
    for i in range(1, t_points):
        all_data_L[0, i, :] = com_28a.iloc[:, i-1]

if len(names_28b) != 3:
    all_data_L[1, :, :] = np.zeros((t_points, n_types), dtype=int)
    ceros = ceros + 1
else:
    for i in range(1, t_points):
        all_data_L[1, i, :] = com_28b.iloc[:, i-1]

if len(names_32a) != 3:
    all_data_L[2, :, :] = np.zeros((t_points, n_types), dtype=int)
    ceros = ceros + 1
else:
    for i in range(1, t_points):
        all_data_L[2, i, :] = com_32a.iloc[:, i-1]

if len(names_32b) != 3:
    all_data_L[3, :, :] = np.zeros((t_points, n_types), dtype=int)
    ceros = ceros + 1
else:
    for i in range(1, t_points):
        all_data_L[3, i, :] = com_32b.iloc[:, i-1]

print(all_data_L)

data = all_data_L
selected_indices = [0, 1, 2, 5, 9]
filtered_data = all_data_L[:, :, selected_indices]
print(filtered_data)

n_types = 5
n_timeseries = 4
sampling_times = [0,24,48,72]
logistic_data_abs_abund, logistic_data_rel_abund, m_Sigma, init_abs_abund, init_rel_abund = compute_moments('lotka-volterra', filtered_data)


print(n_timeseries)
# Upper threshold of numerical solution
upper_threshold = 3. * np.amax(logistic_data_abs_abund["moments"][:,:n_types],0)

with open('./experimental_abs_abund.pickle', 'wb') as f: 
    pc.dump({'n_types':n_types, 't_points':t_points, 'sampling_times':sampling_times, 'n_timeseries':n_timeseries, 'data':logistic_data_abs_abund, 'data_rel':logistic_data_rel_abund, 'init_abs':init_abs_abund }, f)


from sc_LV_numerics import model_abs_abund_closure_from_3rd as LV_model_abs_abund

# import warnings
# warnings.filterwarnings('ignore')

db_path_abs_abund = "sqlite:///.//logistic_inference_abs_abund.db"



def distance_abs_abund(predicted_moments_abs_abund, empirical_moments_abs_abund):
    
    empirical_data = empirical_moments_abs_abund["moments"]
    predicted_data = predicted_moments_abs_abund["moments"]
    
    # Compute the distance metric
    criteria = np.sqrt(((empirical_data - predicted_data)**2).sum())
    
    if criteria < 1.5E14: print(criteria)
    
    return criteria

def distance_rel_abund(predicted_moments_rel_abund, empirical_moments_rel_abund):
    
    empirical_data = empirical_moments_rel_abund["moments"]
    predicted_data = predicted_moments_rel_abund["moments"][:,:-1]
    
    # Compute the distance metric
    criteria = np.sqrt(((empirical_data - predicted_data)**2).sum())
        
    return criteria



with open('./experimental_abs_abund.pickle', 'rb') as f: 
    omm12_abs_abund = pc.load(f)
n_types = omm12_abs_abund['n_types']
sampling_times = omm12_abs_abund['sampling_times']
t_points = omm12_abs_abund['t_points']
n_timeseries = omm12_abs_abund['n_timeseries']
omm12_abs_abund = omm12_abs_abund['data']


## Absolute abundance

# Plot moments

m_k = omm12_abs_abund["moments"][:,:n_types]

mp.semilogy(sampling_times, m_k, 'o', '.-', linestyle=(0,(1,1)), markersize=10)
# Annotate
mp.xlabel('time', fontsize=14)
mp.ylabel(r'abundance moments $\langle n_i \rangle$', fontsize=14)
mp.savefig('moments.png')

# Plot co-moments

cm_kj = omm12_abs_abund["moments"][:,n_types:]

for k in range(n_types**2):
        mp.semilogy(sampling_times, cm_kj[:,k], 'o', '.-', linestyle=(0,(1,1)), markersize=10)
# Annotate
mp.xlabel('time', fontsize=14)
mp.ylabel(r'abundance co-moments $\langle n_i n_j \rangle$', fontsize=14)
mp.savefig('comoments.png')



gR_names = ["gR_%i"%i for i in range(n_types)]
I_intra_names, I_inter_names = [], []
for i in range(n_types):
    for j in range(n_types):
        if i == j: I_intra_names.append("I_%i_%i"%(i,j))
        else: I_inter_names.append("I_%i_%i"%(i,j))
mSigma_names = ["mSigma[0]"]


# Growth rates
gR_prior = n_types * [("uniform", 0, 10)]
# Intra-specific interactions
I_intra_prior = n_types * [("uniform", -10, 10)]
# Inter-specific interactions
I_inter_prior = (n_types**2-n_types) * [("norm", 0, 10)]
# Scaling factor
mSigma_prior = [("uniform", 1.5E4, 1E4)]

parameter_names = gR_names + I_intra_names + I_inter_names + mSigma_names
priors_shapes = gR_prior + I_intra_prior + I_inter_prior + mSigma_prior
priors_dict = dict(zip(parameter_names, priors_shapes))

priors = Distribution(**{key: RV(a, b, c) for key, (a, b, c) in priors_dict.items()})

print(priors_dict)

# Settings for inference

### For ABCSMC functionls

## For population size

# Number of samples in the first generation
start_nr_particles_p = 2000

# CV criterion (smaller values lead to more samples per generation)
mean_cv_p = 0.25

# Number of samples to estimate the eps in first generation
nr_calibration_particles_p = 400

# Number of bootstrapped samples to estimate the CV of a generation
n_bootstrap_p = 2

# Minimum number of samples allowed in a generation
min_population_size_p = 500

# Maximum number of samples allowed in a generation
max_population_size_p = 1000

## For transitions

# Factor to multiply the covariance with
scaling_p = 1.

## For eps

# Initial eps (if ='from sample', eps will be calculated from the prior)
initial_epsilon_p = 'from_sample'

# Quantile for cut-off of samples within a generation (e.g. 0.1 means top 10%)
alpha_p = 0.1

# Factor to multiply the quantile with
quantile_multiplier_p = 1.

### For running ABCSMC function

## Stopping criterion: minimum eps
minimum_epsilon_abs_abund_p = 1E13

## Stopping criterion: minimum eps
minimum_epsilon_rel_abund_p = 1E-1

## Stopping criterion: maximum number of generations
max_nr_populations_p = 6

## Stopping criterion: maximum computing time
max_walltime_p = timedelta(minutes=360)



with open('./logistic_inference_parameters.pickle', 'wb') as f:
    inference_dict = {'priors_dict': priors_dict, 'start_nr_particles_p': start_nr_particles_p, 'mean_cv_p': mean_cv_p, 'nr_calibration_particles_p': nr_calibration_particles_p, 'n_bootstrap_p': n_bootstrap_p, 'min_population_size_p': min_population_size_p, 'max_population_size_p': max_population_size_p, 'scaling_p': scaling_p, 'initial_epsilon_p': initial_epsilon_p, 'alpha_p': alpha_p, 'quantile_multiplier_p': quantile_multiplier_p, 'minimum_epsilon_abs_abund_p': minimum_epsilon_abs_abund_p, 'minimum_epsilon_rel_abund_p': minimum_epsilon_rel_abund_p, 'max_nr_populations_p': max_nr_populations_p, 'max_walltime_p': max_walltime_p}
    pc.dump(inference_dict, f)

print(inference_dict)



abc_abs_abund = ABCSMC(
    models=LV_model_abs_abund,
    parameter_priors=priors,
    distance_function=distance_abs_abund,
    population_size=AdaptivePopulationSize(start_nr_particles=start_nr_particles_p, mean_cv = mean_cv_p, nr_calibration_particles=nr_calibration_particles_p, n_bootstrap=n_bootstrap_p, min_population_size = min_population_size_p, max_population_size = max_population_size_p),
    transitions=MultivariateNormalTransition(scaling=scaling_p, bandwidth_selector=silverman_rule_of_thumb),
    eps=QuantileEpsilon(initial_epsilon=initial_epsilon_p, alpha=alpha_p, quantile_multiplier=quantile_multiplier_p),
)


abc_abs_abund.new(db_path_abs_abund, {"moments": omm12_abs_abund["moments"]});

history_abs_abund = abc_abs_abund.run(minimum_epsilon = minimum_epsilon_abs_abund_p, max_nr_populations = max_nr_populations_p, max_walltime = max_walltime_p)
print('total number of simulations: %i'%history_abs_abund.total_nr_simulations)



history_abs_abund = History(db_path_abs_abund, _id=1)
print('number of generations: %s'%history_abs_abund.max_t)



gR_lim = n_types * [(0, 3)]
I_intra_lim = n_types * [(-10, 10)]
I_inter_lim = (n_types**2-n_types) * [(-10, 10)]
mSigma_lim = [(0., 1E5)]

limits = gR_lim + I_intra_lim + I_inter_lim + mSigma_lim
limits_dict = dict(zip(parameter_names, limits))

gR_mean_posterior, I_intra_mean_posterior, I_inter_mean_posterior = [], [], []

posterior_dist = history_abs_abund.get_distribution()[0]

for par in parameter_names:
    mp.figure()
    par_posterior = posterior_dist.loc[:,par]
    
    x = np.linspace(limits_dict[par][0], limits_dict[par][1], 100)
    
    if par in gR_names: 
        gR_mean_posterior.append(np.mean(par_posterior))
        mp.plot(x, sp.stats.norm.pdf(x, priors_dict[par][1], priors_dict[par][2]), color='k', label = 'prior')
        
    if par in I_intra_names: 
        I_intra_mean_posterior.append(np.mean(par_posterior))
        mp.plot(x, sp.stats.uniform.pdf(x, priors_dict[par][1], priors_dict[par][2]), color='k', label = 'prior')
    
    if par in I_inter_names: 
        I_inter_mean_posterior.append(np.mean(par_posterior))
        mp.plot(x, sp.stats.norm.pdf(x, priors_dict[par][1], priors_dict[par][2]), color='k', label = 'prior')
        
    mp.hist(par_posterior, density=True, bins = 25, alpha=0.8, label = 'posterior')
    mp.xlim(limits_dict[par])
    mp.legend()
    mp.title('%s'%par)
    mp.savefig('%s.png'%par)
    mp.close()
    
print('true: \t\t', np.array(gR_names)[np.argsort(gR)])
print('posterior: \t', np.array(gR_names)[np.argsort(gR_mean_posterior)],'\n')

I_intra = I[np.eye(n_types,dtype=bool)]

print('true: \t\t', np.array(I_intra_names)[np.argsort(I_intra)])
print('posterior: \t', np.array(I_intra_names)[np.argsort(I_intra_mean_posterior)],'\n')

I_inter = I[~np.eye(n_types,dtype=bool)]

print('true: \t\t', np.array(I_inter_names)[np.argsort(I_inter)])