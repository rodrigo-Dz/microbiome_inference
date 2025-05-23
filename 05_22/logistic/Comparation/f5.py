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

axA00 = fig.add_subplot(gsA[5:, :10])

# Create middle panels (Fig 5B)
gsB = gs0[8:].subgridspec(45, 34)

axB00 = fig.add_subplot(gsB[:15, :5])
axB01 = fig.add_subplot(gsB[:15, 7:12])
axB02 = fig.add_subplot(gsB[:15, 14:19])
axB03 = fig.add_subplot(gsB[:15, 21:26])
axB04 = fig.add_subplot(gsB[:15, 28:-1])

axB05 = fig.add_subplot(gsB[20:35, :5])
axB06 = fig.add_subplot(gsB[20:35, 7:12])
axB07 = fig.add_subplot(gsB[20:35, 14:19])
axB08 = fig.add_subplot(gsB[20:35, 21:26])
axB09 = fig.add_subplot(gsB[20:35, 28:-1])

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

names_2_axes ={
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
axA00.legend([Line2D([0],[0], color=colors[name],lw=4) for name in species], species, labelcolor='linecolor', loc='upper center', ncol=2, prop=dict(style='italic',size=14), frameon=False, handletextpad=0.5, labelspacing=0.6, handlelength=0.6, columnspacing = 1.9)

    
for ax in [axB00, axB01, axB02, axB03, axB04, axB05, axB06, axB07, axB08, axB09]:
    ax.locator_params(axis='x', nbins=3)
    

# Build dictionary from short to long microbial types names
col_names = [['ST00101', 'ST00154', 'ST00042', 'ST00046', 'ST00109'],
            ['ST00101', 'ST00154', 'ST00060', 'ST00046', 'ST00109'],
            ['ST00101', 'ST00154', 'ST00042', 'ST00046', 'ST00094'],
            ['ST00154', 'ST00042', 'ST00046', 'ST00110', 'ST00109'],
            ['ST00101', 'ST00042', 'ST00164', 'ST00046', 'ST00109'],
            ['ST00101', 'ST00154', 'ST00042', 'ST00143', 'ST00109'],
            ['ST00164', 'ST00060', 'ST00143', 'ST00110', 'ST00094'],
            ['ST00042', 'ST00164', 'ST00143', 'ST00110', 'ST00094'],
            ['ST00164', 'ST00060', 'ST00143', 'ST00110', 'ST00109'],
            ['ST00101', 'ST00164', 'ST00060', 'ST00143', 'ST00094'],
            ['ST00154', 'ST00060', 'ST00143', 'ST00110', 'ST00094'],
            ['ST00164', 'ST00060', 'ST00046', 'ST00110', 'ST00094']]
#C1_names_short_2_long = {'ST00101': 'Pseudomonas sp.', 'ST00154': 'Agrobacterium sp.', 'ST00042': 'Leifsonia sp.', 'ST00046': 'Bacillus sp.', 'ST00109': 'Mycobacterium sp.'}

col_comunities = ["#E22D2D", "#F19721", "#EEEB30", "#87E937", "#30FA8B", "#48F7F7", "#3255F1", "#8A4DEC", "#F34FE5", "#836C3C", "#EEADC2", "#91ADFA"]

for i in range(1,13):
# Import OMM-12 absolute abundance data and parameters
    with open('../C%i_logistic/data/C1_abs_abund.pickle'%i, 'rb') as f: 
        abs_abund = pc.load(f)
    abs_abund = abs_abund['data']["moments"]

    with open('../C%i_logistic/logistic_inference_parameters.pickle'%i, 'rb') as f:
        logistic_inf_par =  pc.load(f)
    logistic_priors_dict = logistic_inf_par['priors_dict']
    # Import inference posteriors from OMM-12
    logistic_history_abs_abund = History("sqlite:///../C%i_logistic/data/logistic_inference_abs_abund.db"%i, _id=1)
    logistic_posteriors = logistic_history_abs_abund.get_distribution(m=0,t=logistic_history_abs_abund.max_t)[0]

    com = col_names[i-1]
    print(com)
    j = 0
    for type in col_names[i-1]:
        
        type_name = keys[type]
            
        ax = names_2_axes[type]
        
        ax.set_ylim(0.01, 0.06)  
        ax.set_xlim(20, 72)    

        #ax.set_ylim(ymin=9E-3)
                
        type_color = colors[type_name]
        
        # Plot posterior
        ax.hist(logistic_posteriors.loc[:, f'gR_{j}'], color = col_comunities[i-1], alpha = 0.4, density=True, bins = 20, log=True)

        mean_value = logistic_posteriors.loc[:, f'gR_{j}'].mean()
        ax.axvline(mean_value, color=col_comunities[i-1], linestyle='--', linewidth=2)

        
        # Plot prior
        x = np.linspace(0., 72, 100)
        y = sp.stats.uniform.pdf(x, 0., 72)
        ax.plot(x, y, color='k', label = 'prior')

        ax.set_xticks([0,36,72])
        
        ax.set_title('%s'%type_name, fontsize=14, color = type_color, style='italic')
        j = j + 1
    

for ax in [axB05, axB06, axB07, axB08, axB09]:
    ax.set_xlabel(r'growth (day$^{-1}$)', fontsize=15)

axB00.set_ylabel('prob. density', fontsize=15) 
axB05.set_ylabel('prob. density', fontsize=15) 

    
for ax in [axB00, axB01, axB02, axB03, axB04]:
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)

for ax in [axB00, axB01, axB02, axB03, axB04]:
    ax.tick_params(axis='both', direction='in',top=True, right=True)

#mp.tight_layout()
mp.savefig('./fig5.pdf', dpi=300, format='pdf', bbox_inches='tight')