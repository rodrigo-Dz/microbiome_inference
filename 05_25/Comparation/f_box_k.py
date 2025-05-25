import matplotlib.pyplot as mp
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
import pickle as pc
from pyabc import History
import numpy as np
from collections import defaultdict
from scipy.stats import f_oneway

# Make figure
fig = mp.figure(figsize=(13, 13))

# Create array to build figure panels
gs0 = gs.GridSpec(25, 1, figure=fig)


# Create middle panels (Fig 5B)
gsB = gs0[8:].subgridspec(45, 34)
axB00 = fig.add_subplot(gsB[:, :])

# Mapping
species = ['Leifsonia sp.', 'Pseudomonas sp.', 'Mycobacterium sp.', 'Agrobacterium sp.', 'Bacillus sp.',
           'Arthrobacter sp.', 'Rhodococcus erythropolis', 'Variovorax paradoxus', 'Rhizobium sp.', 'Paenibacillus sp.']

keys = {
    "ST00042": 'Leifsonia sp.',
    "ST00046": 'Bacillus sp.',
    "ST00060": 'Arthrobacter sp.',
    "ST00094": 'Rhodococcus erythropolis',
    "ST00101": 'Pseudomonas sp.',
    "ST00109": 'Mycobacterium sp.',
    "ST00110": 'Variovorax paradoxus',
    "ST00143": 'Paenibacillus sp.',
    "ST00154": 'Agrobacterium sp.',
    "ST00164": 'Rhizobium sp.'
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

# Community colors and compositions
col_names = [
    ['ST00101', 'ST00154', 'ST00042', 'ST00046', 'ST00109'],
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
    ['ST00164', 'ST00060', 'ST00046', 'ST00110', 'ST00094']
]

col_comunities = ["#E22D2D", "#F19721", "#EEEB30", "#87E937", "#30FA8B", "#48F7F7", "#3255F1", "#8A4DEC", "#F34FE5", "#836C3C", "#EEADC2", "#91ADFA"]

# OMM-12 microbial types named and coloured

# Almacenar los datos para ANOVA
values = []
labels = []

# Llenar los datos y preparar para boxplots
for i in range(1, 13):
    with open(f'../C{i}_logistic/data/C1_abs_abund.pickle', 'rb') as f:
        abs_abund = pc.load(f)['data']["moments"]

    with open(f'../C{i}_logistic/logistic_inference_parameters.pickle', 'rb') as f:
        logistic_inf_par = pc.load(f)

    logistic_history = History(f"sqlite:///../C{i}_logistic/data/logistic_inference_abs_abund.db", _id=1)
    logistic_posteriors = logistic_history.get_distribution(m=0, t=logistic_history.max_t)[0]

    com = col_names[i - 1]

    values.append(logistic_posteriors.loc[:, 'N'].values)
    labels.append(f'C{i}')



box = axB00.boxplot(values, labels=labels, patch_artist=True)
# Colorear las cajas según la comunidad
for patch, color in zip(box['boxes'], col_comunities):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

axB00.set_ylim(1.3E7, 1.5E7)  # Ajustar el límite del eje y
#axB00.set_xticklabels(ax.labels, rotation=90, fontsize=10)
#axB00.set_title(keys[labels], fontsize=14, color=colors[keys[labels]], style='italic')
#axB00.tick_params(axis='y', labelsize=12)
# Etiquetas y formato
axB00.set_ylabel('K', fontsize=15)
axB00.set_xlabel('Comunidad', fontsize=13)

for ax in [axB00]:
    ax.tick_params(axis='both', direction='in', top=True, right=True)

# Guardar figura
mp.savefig('./fig5_k.pdf', dpi=300, format='pdf', bbox_inches='tight')

