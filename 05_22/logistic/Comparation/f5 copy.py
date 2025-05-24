import matplotlib.pyplot as mp
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import pickle as pc
from pyabc import History
import scipy as sp
import numpy as np


# Make figure
fig = mp.figure(figsize=(13, 13))

# Create array to build figure panels
gs0 = gs.GridSpec(25, 1, figure=fig)

# Create top panels (Fig 5A)
gsA = gs0[:8].subgridspec(21, 30)

axA00 = fig.add_subplot(gsA[:, :10])
axA01 = fig.add_subplot(gsA[:-5, 14:])

# Create middle panels (Fig 5B)
gsB = gs0[9:].subgridspec(45, 34)

axB00 = fig.add_subplot(gsB[:15, :5])
axB01 = fig.add_subplot(gsB[:15, 7:12])
axB02 = fig.add_subplot(gsB[:15, 14:19])
axB03 = fig.add_subplot(gsB[:15, 21:26])
axB04 = fig.add_subplot(gsB[:15, 28:-1])

axB05 = fig.add_subplot(gsB[17:32, :5])
axB06 = fig.add_subplot(gsB[17:32, 7:12])
axB07 = fig.add_subplot(gsB[17:32, 14:19])
axB08 = fig.add_subplot(gsB[17:32, 21:26])
axB09 = fig.add_subplot(gsB[17:32, 28:-1])

# Name microbial types
species = ['Leifsonia sp.', 'Pseudomonas sp.', 'Mycobacterium sp.', 'Agrobacterium sp.', 'Bacillus sp.',
           'Arthrobacter sp.', 'Rhodococcus erythropolis', 'Variovorax paradoxus', 'Rhizobium sp.', 'Paenibacillus sp.']
keys = {
    "ST00042":'Leifsonia sp.',
    "ST00046":'Bacillus sp.', 
    "ST00060":'Arthrobacter sp.', 
    "ST00094":'Rhodococcus erythropolis', 
    "ST00101":'Pseudomonas sp.', 
    "ST00109":'Mycobacterium sp.', 
    "ST00110":'Variovorax paradoxus', 
    "ST00143":'Paenibacillus sp.', 
    "ST00154":'Agrobacterium sp.', 
    "ST00164":'Rhizobium sp.'
}

axx ={
    "ST00042":axB00,
    "ST00046":axB01,    
    "ST00060":axB02,
    "ST00094":axB03,
    "ST00101":axB04,
    "ST00109":axB05,
    "ST00110":axB06,
    "ST00143":axB07,
    "ST00154":axB08,
    "ST00164":axB09,
} 

colors = {
    'Leifsonia sp.': '#BDD7E7',
    'Bacillus sp.': '#25383C',
    'Arthrobacter sp.': '#FCAE91',
    'Rhodococcus erythropolis': "#FA4D4D",
    'Pseudomonas sp.': "#6BEEF3",
    'Mycobacterium sp.': '#4682B4',
    'Variovorax paradoxus': '#EE6A50',
    'Paenibacillus sp.': '#8B2323',
    'Agrobacterium sp.': '#08519C',
    'Rhizobium sp.': '#CD2626'
}
# OMM-12 microbial types named and coloured
axA00.axis('off')
axA00.legend([Line2D([0],[0], color=colors[name],lw=4) for name in species], species, labelcolor='linecolor', loc='upper center', ncol=2, prop=dict(style='italic',size=16), frameon=False, handletextpad=0.5, labelspacing=0.6, handlelength=0.6, columnspacing = 1.9)
axA00.set_title(r'oligo-mouse-microbiota (OMM$\bf{^{12}}$)', weight='bold', fontsize=18)

    
for ax in [axB00, axB01, axB02, axB03, axB04, axB05, axB06, axB07, axB08, axB09]:
    ax.locator_params(axis='x', nbins=3)

# Import OMM-12 absolute abundance data and parameters
with open('../C1_logistic/data/C1_abs_abund.pickle', 'rb') as f: 
    C1_abs_abund = pc.load(f)
C1_abs_abund = C1_abs_abund['data']["moments"]

with open('../C1_logistic/logistic_inference_parameters.pickle', 'rb') as f:
    logistic_inf_par =  pc.load(f)
logistic_priors_dict = logistic_inf_par['priors_dict']
    
sampling_times = np.array([0, 24, 48, 72])

# General parameters
t_points = len(sampling_times)
n_types = 5
n_timeseries = 4

# Build dictionary from short to long microbial types names
C1_df_col_names = ['ST00101', 'ST00154', 'ST00042', 'ST00046', 'ST00109']
C1_names_short_2_long = {'ST00101': 'Pseudomonas sp.', 'ST00154': 'Agrobacterium sp.', 'ST00042': 'Leifsonia sp.', 'ST00046': 'Bacillus sp.', 'ST00109': 'Mycobacterium sp.'}

# Assign panels to each microbial type
#names_2_axes = {'ST00101': axC00, 'ST00154': axC01, 'ST00042': axC02, 'ST00046': axC03, 'ST00109': axC10}
#names_2_axes_t = {'ST00101': axC00_t, 'ST00154': axC01_t, 'ST00042': axC02_t, 'ST00046': axC03_t, 'ST00109': axC10_t}

## Plot Fig 5A (OMM dynamics data)
# Plot timeseries of first moments
for i in range(n_types):
    type_name = C1_names_short_2_long[C1_df_col_names[i]]
    axA01.semilogy(sampling_times, C1_abs_abund[:, i], 'o', '.-', linestyle=(0,(1,1)), color = colors[type_name])
axA01.set_xlim(xmin=0)
axA01.set_xticks(sampling_times)

# Import inference posteriors from OMM-12
C1_logistic_history_abs_abund = History("sqlite:///../C1_logistic/data/logistic_inference_abs_abund.db", _id=1)
C1_logistic_posteriors = C1_logistic_history_abs_abund.get_distribution(m=0,t=C1_logistic_history_abs_abund.max_t)[0]

#with open('../data/omm12/experimental_abs_abund.pickle', 'rb') as f:
#    omm12_data_abs_abund =  pc.load(f)
#omm12_types_names = omm12_data_abs_abund['types_names']
omm12_types_names =  ['ST00101', 'ST00154', 'ST00042', 'ST00046', 'ST00109']

# Assign figure panels for the ranking of "most certain" inferred parameters
axB_by_certainty = [axB00, axB01, axB02, axB03, axB04, axB05, axB06, axB07, axB08, axB09]

# Plot Fig 5B (Most certain posteriors)
# Identify and plot most certain inferred parameters
for i in range(5):
    
    n_type = par_uncertainty_argsort[:5][i]
        
    ax = axB_by_certainty[i]
    
    # Growth Rates
    if n_type < n_types:
        
        ax.set_ylim(ymin=9E-3)

        type_name_long = C1_names_short_2_long[omm12_types_names[n_type]]
                
        type_color = colors[type_name_long]
        
        # Plot posterior
        ax.hist(C1_logistic_posteriors.loc[:,'gR_%i'%n_type], color = type_color, alpha = 1., density=True, bins = 15, log=True)
        
        # Plot prior
        x = np.linspace(0., 72, 100)
        y = sp.stats.uniform.pdf(x, 0., 72)
        ax.plot(x, y, color='k', label = 'prior')

        ax.set_xticks([0,36,72])
        
        ax.set_xlabel(r'growth (day$^{-1}$)', fontsize=17)
        ax.set_title('%s'%type_name_long, fontsize=14, color = type_color, style='italic')
    
    # Death rates
    elif n_type >= n_types and n_type < 2 * n_types:

        ax.set_ylim(ymin=9E-8, ymax=1E-5)

        n_type = n_type - n_types

        type_name_long = C1_names_short_2_long[omm12_types_names[n_type]]
                
        type_color = colors[type_name_long]
        
        # Plot posterior
        ax.hist(C1_logistic_posteriors.loc[:,'dR_%i'%n_type], color = type_color, alpha = 1., density=True, bins = 15, log=True)
        
        # Plot prior
        x = np.linspace(0, 2E6, 100)
        y = sp.stats.uniform.pdf(x, 0, 2E6)
        ax.plot(x, y, color='k', label = 'prior')

        ax.set_xticks([0,1E6,2E6])

        ax.xaxis.offsetText.set_fontsize(16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
        
        ax.set_xlabel(r'death (cells/day)', fontsize=16, labelpad=20)
        ax.set_title('%s'%type_name_long, fontsize=14, color = type_color, style='italic')

    # Migration rates
    else:

        ax.set_ylim(ymin=9E-8, ymax=1E-5)

        n_type = n_type - 2 * n_types

        type_name_long = C1_names_short_2_long[omm12_types_names[n_type]]
                
        type_color = colors[type_name_long]
        
        

        # Plot posterior
        ax.hist(C1_logistic_posteriors.loc[:,'mR_%i'%n_type], color = type_color, alpha = 1., density=True, bins = 15, log=True)
        
        # Plot prior
        x = np.linspace(0, 2E6, 100)
        y = sp.stats.uniform.pdf(x, 0, 2E6)
        ax.plot(x, y, color='k', label = 'prior')

        ax.set_xticks([0,1E6,2E6])

        ax.xaxis.offsetText.set_fontsize(16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)

        ax.set_xlabel(r'immi. (cells/day)', fontsize=16, labelpad=20)
        ax.set_title('%s'%type_name_long, fontsize=14, color = type_color, style='italic')
    
# Plot Fig 5C (All posteriors)



axA01.set_title(r'absolute abundance $\langle n_k \rangle$', fontsize=18)

axA01.set_xlabel('time (days)', fontsize=18)
    
axB00.set_ylabel('prob. density', fontsize=18) 
    
for ax in [axA01, axB00, axB01, axB02, axB03, axB04]:
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)

for ax in [axA01, axB00, axB01, axB02, axB03, axB04]:
    ax.tick_params(axis='both', direction='in',top=True, right=True)

# Annotate panels
mp.gcf().text(0.05, 0.9, "A", weight='bold', fontsize=22)
mp.gcf().text(0.05, 0.66, "B", weight='bold', fontsize=22)

# Save figure
#mp.tight_layout()
mp.savefig('./fig5.pdf', dpi=300, format='pdf', bbox_inches='tight')