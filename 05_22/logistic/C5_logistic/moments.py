import pandas as pd
import numpy as np
from random import random
import pickle as pc

import matplotlib.pyplot as mp
import matplotlib.lines as mlines
from matplotlib import rc
rc('xtick', labelsize=14)
rc('ytick', labelsize=14)
mp.rcParams['figure.figsize'] = (4, 4)

def compute_moments(data, n_timeseries):
    # Datasets to store moments
        
    abs_abund_timeseries_df = []
    rel_abund_timeseries_df = []

    # Make Gillespie simulations
    ceros = 0
    for sample in range(0,4):
        sam = data[sample,:,:]
        if np.all(sam == 0):
            ceros = ceros + 1
        else:
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
    
    return data_abs_abund, data_rel_abund, m_Sigma, init_abs_abund, init_rel_abund, n_timeseries


# Cargar el archivo CSV
data = pd.read_csv('./data/freq.csv')
meta = pd.read_csv('./data/meta.csv')

com = "R5"
exps = ["NS1", "NS3"]
n_exp = 4
t_points = 4
n_types = 10
bact = [ 0, 2, 3, 5, 9]

all_data_L = np.zeros((n_exp, t_points, n_types), dtype=int)

exp_n = meta[meta["community"] == com]
pop_i = exp_n[exp_n['hrs'] == 0]
names = pop_i['Unnamed: 0'].tolist()
com = data[names]

all_data_L[0,0,:] = com.iloc[:,0]
all_data_L[1,0,:] = com.iloc[:,0]
all_data_L[2,0,:] = com.iloc[:,1]
all_data_L[3,0,:] = com.iloc[:,1]

exp_n = exp_n[exp_n['hrs'] > 0]

exp_28 = exp_n[exp_n['temp'] == 28]
exp_28_a = exp_28[exp_28["exp"] == exps[0]]
exp_28_b = exp_28[exp_28["exp"] == exps[1]]
exp_32 = exp_n[exp_n['temp'] == 32]
exp_32_a = exp_32[exp_32["exp"] == exps[0]]
exp_32_b = exp_32[exp_32["exp"] == exps[1]]


names_28a = exp_28_a['Unnamed: 0'].tolist()
print(names_28a)
names_28b = exp_28_b['Unnamed: 0'].tolist()
print(names_28b)    
names_32a = exp_32_a['Unnamed: 0'].tolist()
print(names_32a)
names_32b = exp_32_b['Unnamed: 0'].tolist()
print(names_32b)
# Obtener los datos correspondientes de 'data'
com_28a = data[names_28a]
com_28b = data[names_28b]
com_32a = data[names_32a]
com_32b = data[names_32b]


if len(names_28a) != 3:
    all_data_L[0, :, :] = np.zeros((t_points, n_types), dtype=int)
else:
    for i in range(1, t_points):
        all_data_L[0, i, :] = com_28a.iloc[:, i-1]

if len(names_28b) != 3:
    all_data_L[2, :, :] = np.zeros((t_points, n_types), dtype=int)
else:
    for i in range(1, t_points):
        all_data_L[2, i, :] = com_28b.iloc[:, i-1]

if len(names_32a) != 3:
    all_data_L[1, :, :] = np.zeros((t_points, n_types), dtype=int)
else:
    for i in range(1, t_points):
        all_data_L[1, i, :] = com_32a.iloc[:, i-1]

if len(names_32b) != 3:
    all_data_L[3, :, :] = np.zeros((t_points, n_types), dtype=int)
else:
    for i in range(1, t_points):
        all_data_L[3, i, :] = com_32b.iloc[:, i-1]


print(all_data_L)

data = all_data_L[:, :, bact]
print(data)
n_types = 5

sampling_times = [0,24,48,72]

logistic_data_abs_abund, logistic_data_rel_abund, m_Sigma, init_abs_abund, init_rel_abund, n_timeseries= compute_moments(data, n_exp)


with open('./data/C1_abs_abund.pickle', 'wb') as f: 
    pc.dump({'n_types':n_types, 't_simulated':sampling_times[-1], 't_points':t_points, 'sampling_times':sampling_times, 'n_timeseries':n_timeseries, 'init_abs':init_abs_abund, 'data':logistic_data_abs_abund }, f)

with open('./data/C1_rel_abund.pickle', 'wb') as f: 
    pc.dump({'n_types':n_types, 't_simulated':sampling_times[-1], 't_points':t_points, 'sampling_times':sampling_times, 'n_timeseries':n_timeseries, 'init_rel':init_abs_abund, 'data':logistic_data_rel_abund }, f)


with open('./data/C1_rel_abund.pickle', 'rb') as f: 
    C1_abs_abund = pc.load(f)
n_types = C1_abs_abund['n_types']
sampling_times = C1_abs_abund['sampling_times']
t_points = C1_abs_abund['t_points']
n_timeseries = C1_abs_abund['n_timeseries']
C1_abs_abund = C1_abs_abund['data']

with open('./data/C1_rel_abund.pickle', 'rb') as f: 
    C1_rel_abund = pc.load(f)
C1_rel_abund = C1_rel_abund['data']

## Absolute abundance

# Plot moments

m_k = C1_abs_abund["moments"][:,:n_types]

mp.figure()
mp.semilogy(sampling_times, m_k, 'o', '.-', linestyle=(0,(1,1)), markersize=10)
# Annotate
mp.xlabel('time', fontsize=14)
mp.ylabel(r'abundance moments $\langle n_i \rangle$', fontsize=14)
mp.savefig('./res/abs_moments.png')
mp.close()

# Plot co-moments

cm_kj = C1_abs_abund["moments"][:,n_types:]

mp.figure()
for k in range(n_types**2):
        mp.semilogy(sampling_times, cm_kj[:,k], 'o', '.-', linestyle=(0,(1,1)), markersize=10)
# Annotate
mp.xlabel('time', fontsize=14)
mp.ylabel(r'abundance co-moments $\langle n_i n_j \rangle$', fontsize=14)
mp.savefig('./res/abs_comoments.png')
mp.close()



## Relative abundance

# Plot moments

m_k = C1_rel_abund["moments"][:,:n_types]

mp.figure()
mp.semilogy(sampling_times, m_k, 'o', '.-', linestyle=(0,(1,1)), markersize=10)
# Annotate
mp.xlabel('time', fontsize=14)
mp.ylabel(r'rel. abund. moments $\langle x_i \rangle$', fontsize=14)
mp.savefig('./res/rel_moments.png')
mp.close()

# Plot co-moments

cm_kj = C1_rel_abund["moments"][:,n_types:]

mp.figure()
for k in range(n_types**2):
        mp.semilogy(sampling_times, cm_kj[:,k], 'o', '.-', linestyle=(0,(1,1)), markersize=10)
# Annotate
mp.xlabel('time', fontsize=14)
mp.ylabel(r'rel. abund. co-moments $\langle x_i x_j \rangle$', fontsize=14)
mp.savefig('./res/rel_comoments.png')
mp.close()

print(n_timeseries)
