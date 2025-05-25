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



from sc_LV_numerics import model_abs_abund_closure_from_2nd as LV_model_abs_abund
from sc_LV_numerics import model_rel_abund_closure_from_2nd as LV_model_rel_abund

# import warnings
# warnings.filterwarnings('ignore')

db_path_abs_abund = "sqlite:///./data/LV_inference_abs_abund.db"
db_path_rel_abund = "sqlite:///./data/LV_inference_rel_abund.db"


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


with open('./data/C1_abs_abund.pickle', 'rb') as f: 
    C1_abs_abund = pc.load(f)
n_types = C1_abs_abund['n_types']
sampling_times = C1_abs_abund['sampling_times']
t_points = C1_abs_abund['t_points']
n_timeseries = C1_abs_abund['n_timeseries']
C1_abs_abund = C1_abs_abund['data']


with open('./data/C1_rel_abund.pickle', 'rb') as f: 
    C1_rel_abund = pc.load(f)
C1_rel_abund = C1_rel_abund['data']


gR_names = ["gR_%i"%i for i in range(n_types)]
I_intra_names, I_inter_names = [], []
for i in range(n_types):
    for j in range(n_types):
        if i == j: I_intra_names.append("I_%i_%i"%(i,j))
        else: I_inter_names.append("I_%i_%i"%(i,j))
mSigma_names = ["mSigma[0]"]

# Growth rates
gR_prior = n_types * [("uniform", 0.25, 72)]
# Intra-specific interactions
I_intra_prior = n_types * [("uniform", -0.0001, 0.0001)]
# Inter-specific interactions
I_inter_prior = (n_types**2-n_types) * [("norm", 0, 0.0001)]
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
# Settings for inference

### For ABCSMC function

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
max_walltime_p = timedelta(minutes=6*720)



with open('./LV_inference_parameters.pickle', 'wb') as f:
    inference_dict = {'priors_dict': priors_dict, 'start_nr_particles_p': start_nr_particles_p, 'mean_cv_p': mean_cv_p, 'nr_calibration_particles_p': nr_calibration_particles_p, 'n_bootstrap_p': n_bootstrap_p, 'min_population_size_p': min_population_size_p, 'max_population_size_p': max_population_size_p, 'scaling_p': scaling_p, 'initial_epsilon_p': initial_epsilon_p, 'alpha_p': alpha_p, 'quantile_multiplier_p': quantile_multiplier_p, 'minimum_epsilon_abs_abund_p': minimum_epsilon_abs_abund_p, 'minimum_epsilon_rel_abund_p': minimum_epsilon_rel_abund_p, 'max_nr_populations_p': max_nr_populations_p, 'max_walltime_p': max_walltime_p}
    pc.dump(inference_dict, f)

print(inference_dict)

'''

abc_abs_abund = ABCSMC(
    models=LV_model_abs_abund,
    parameter_priors=priors,
    distance_function=distance_abs_abund,
    population_size=AdaptivePopulationSize(start_nr_particles=start_nr_particles_p, mean_cv = mean_cv_p, nr_calibration_particles=nr_calibration_particles_p, n_bootstrap=n_bootstrap_p, min_population_size = min_population_size_p, max_population_size = max_population_size_p),
    transitions=MultivariateNormalTransition(scaling=scaling_p, bandwidth_selector=silverman_rule_of_thumb),
    eps=QuantileEpsilon(initial_epsilon=initial_epsilon_p, alpha=alpha_p, quantile_multiplier=quantile_multiplier_p),
)


abc_abs_abund.new(db_path_abs_abund, {"moments": C1_abs_abund["moments"]});

history_abs_abund = abc_abs_abund.run(minimum_epsilon = minimum_epsilon_abs_abund_p, max_nr_populations = max_nr_populations_p, max_walltime = max_walltime_p)
print('total number of simulations: %i'%history_abs_abund.total_nr_simulations)
'''


history_abs_abund = History(db_path_abs_abund, _id=1)
print('number of generations: %s'%history_abs_abund.max_t)


gR_lim = n_types * [(0, 75)]
I_intra_lim = n_types * [(-.002, 0.002)]
I_inter_lim = (n_types**2-n_types) * [(-0.002, 0.002)]
mSigma_lim = [(0., 1E5)]



limits = gR_lim + I_intra_lim + I_inter_lim + mSigma_lim
limits_dict = dict(zip(parameter_names, limits))

gR_mean_posterior, I_intra_mean_posterior, I_inter_mean_posterior = [], [], []

posterior_dist = history_abs_abund.get_distribution()[0]

for par in parameter_names:
    
    par_posterior = posterior_dist.loc[:,par]
    
    x = np.linspace(limits_dict[par][0], limits_dict[par][1], 100)
    mp.figure()

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
    mp.savefig('./res/%s_abs'%par)
    mp.close()

print('posterior: \t', np.array(gR_names)[np.argsort(gR_mean_posterior)],'\n')


print('posterior: \t', np.array(I_intra_names)[np.argsort(I_intra_mean_posterior)],'\n')


print('posterior: \t', np.array(I_inter_names)[np.argsort(I_inter_mean_posterior)],'\n')
