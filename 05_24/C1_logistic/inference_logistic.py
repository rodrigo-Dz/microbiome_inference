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



from sc_logistic_numerics import model_abs_abund_closure_from_3rd as logistic_model_abs_abund
from sc_logistic_numerics import model_rel_abund_closure_from_3rd as logistic_model_rel_abund

# import warnings
# warnings.filterwarnings('ignore')

db_path_abs_abund = "sqlite:///./data/C1_logistic_inference_abs_abund.db"
db_path_rel_abund = "sqlite:///./data/C1_logistic_inference_rel_abund.db"


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
dR_names = ["dR_%i"%i for i in range(n_types)]
N_names = ["N"]
mSigma_names = ["mSigma[0]"]

# Growth rates
gR_prior = n_types * [("uniform", 0.25, 80)]
# Death rates
dR_prior = n_types * [("uniform", 0, 2E6)]
# Carrying capacity
N_prior = [("uniform", 1.4E7, 2E6)]
# Scaling factor
mSigma_prior = [("uniform", 1E7, 9E7)]

parameter_names = gR_names + dR_names + N_names + mSigma_names
priors_shapes = gR_prior + dR_prior + N_prior + mSigma_prior
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
minimum_epsilon_abs_abund_p = 1E8

## Stopping criterion: minimum eps
minimum_epsilon_rel_abund_p = 1E-3

## Stopping criterion: maximum number of generations
max_nr_populations_p = 80

## Stopping criterion: maximum computing time
max_walltime_p = timedelta(minutes=6*720)



with open('./logistic_inference_parameters.pickle', 'wb') as f:
    inference_dict = {'priors_dict': priors_dict, 'start_nr_particles_p': start_nr_particles_p, 'mean_cv_p': mean_cv_p, 'nr_calibration_particles_p': nr_calibration_particles_p, 'n_bootstrap_p': n_bootstrap_p, 'min_population_size_p': min_population_size_p, 'max_population_size_p': max_population_size_p, 'scaling_p': scaling_p, 'initial_epsilon_p': initial_epsilon_p, 'alpha_p': alpha_p, 'quantile_multiplier_p': quantile_multiplier_p, 'minimum_epsilon_abs_abund_p': minimum_epsilon_abs_abund_p, 'minimum_epsilon_rel_abund_p': minimum_epsilon_rel_abund_p, 'max_nr_populations_p': max_nr_populations_p, 'max_walltime_p': max_walltime_p}
    pc.dump(inference_dict, f)

print(inference_dict)


abc_rel_abund = ABCSMC(
    models=logistic_model_rel_abund,
    parameter_priors=priors,
    distance_function=distance_rel_abund,
    population_size=AdaptivePopulationSize(start_nr_particles=start_nr_particles_p, mean_cv = mean_cv_p, nr_calibration_particles=nr_calibration_particles_p, n_bootstrap=n_bootstrap_p, min_population_size = min_population_size_p, max_population_size = max_population_size_p),
    transitions=MultivariateNormalTransition(scaling=scaling_p, bandwidth_selector=silverman_rule_of_thumb),
    eps=QuantileEpsilon(initial_epsilon=initial_epsilon_p, alpha=alpha_p, quantile_multiplier=quantile_multiplier_p),
)

abc_rel_abund.new(db_path_rel_abund, {"moments": C1_rel_abund["moments"]});

history_rel_abund = abc_rel_abund.run(minimum_epsilon = minimum_epsilon_rel_abund_p, max_nr_populations = max_nr_populations_p, max_walltime = max_walltime_p)
print('total number of simulations: %i'%history_rel_abund.total_nr_simulations)



history_rel_abund = History(db_path_rel_abund, _id=1)
print('number of generations: %s'%history_rel_abund.max_t)


gR_lim = n_types * [(0, 75)]
dR_lim = n_types * [(0, 2.1E6)]
N_lim = [(1.2E7, 1.7E7)]
mSigma_lim = [(1E7, 1E8)]

limits = gR_lim + dR_lim  + N_lim + mSigma_lim
limits_dict = dict(zip(parameter_names, limits))

gR_mean_posterior, dR_mean_posterior, mR_mean_posterior = [], [], []

posterior_dist = history_rel_abund.get_distribution(m=0,t=history_rel_abund.max_t)[0]

for par in parameter_names:
    
    par_posterior = posterior_dist.loc[:,par]
    
    x = np.linspace(limits_dict[par][0], limits_dict[par][1], 100)
    mp.figure()
    if par in gR_names: 
        gR_mean_posterior.append(np.mean(par_posterior))
        mp.plot(x, sp.stats.uniform.pdf(x, priors_dict[par][1], priors_dict[par][2]), color='k', label = 'prior')
        
    if par in dR_names: 
        dR_mean_posterior.append(np.mean(par_posterior))
        mp.plot(x, sp.stats.uniform.pdf(x, priors_dict[par][1], priors_dict[par][2]), color='k', label = 'prior')

    if par == 'N': 
        mp.plot(x, sp.stats.uniform.pdf(x, priors_dict[par][1], priors_dict[par][2]), color='k', label = 'prior')
        
    mp.hist(par_posterior, density=True, bins = 25, alpha=0.8, label = 'posterior')
    mp.xlim(limits_dict[par])
    mp.legend()
    mp.title('%s'%par)
    mp.savefig('./res/%s_rel'%par)
    mp.close()

print('posterior: \t', np.array(gR_names)[np.argsort(gR_mean_posterior)],'\n')

print('posterior: \t', np.array(dR_names)[np.argsort(dR_mean_posterior)],'\n')

