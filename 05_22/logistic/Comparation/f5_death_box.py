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

names_2_axes = {
    "ST00042": axB00,
    "ST00046": axB01,
    "ST00060": axB02,
    "ST00094": axB03,
    "ST00101": axB04,
    "ST00109": axB05,
    "ST00110": axB06,
    "ST00143": axB07,
    "ST00154": axB08,
    "ST00164": axB09,
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
axA00.axis('off')
axA00.legend([Line2D([0], [0], color=colors[name], lw=4) for name in species], species,
             labelcolor='linecolor', loc='upper center', ncol=2,
             prop=dict(style='italic', size=14), frameon=False,
             handletextpad=0.5, labelspacing=0.6, handlelength=0.6, columnspacing=1.9)

for ax in names_2_axes.values():
    ax.locator_params(axis='x', nbins=3)

# Almacenar los datos para ANOVA
data_para_anova = defaultdict(list)

# Llenar los datos y preparar para boxplots
for i in range(1, 13):
    with open(f'../C{i}_logistic/data/C1_abs_abund.pickle', 'rb') as f:
        abs_abund = pc.load(f)['data']["moments"]

    with open(f'../C{i}_logistic/logistic_inference_parameters.pickle', 'rb') as f:
        logistic_inf_par = pc.load(f)

    logistic_history = History(f"sqlite:///../C{i}_logistic/data/logistic_inference_abs_abund.db", _id=1)
    logistic_posteriors = logistic_history.get_distribution(m=0, t=logistic_history.max_t)[0]

    com = col_names[i - 1]
    j = 0
    for type in com:
        type_name = keys[type]
        ax = names_2_axes[type]
        values = logistic_posteriors.loc[:, f'dR_{j}'].values

        data_para_anova[type].append(values)

        if 'all_values' not in ax.__dict__:
            ax.all_values = []
            ax.labels = []
            ax.colors = []

        ax.all_values.append(values)
        ax.labels.append(f'C{i}')
        ax.colors.append(col_comunities[i - 1])
        j += 1

# Hacer los boxplots después de acumular todo
for tipo, ax in names_2_axes.items():
    if hasattr(ax, 'all_values'):
        ax.clear()
        bp = ax.boxplot(ax.all_values, labels=ax.labels, patch_artist=True)

        for patch, color in zip(bp['boxes'], ax.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylim(0.0, 1.)
        ax.set_xticklabels(ax.labels, rotation=90, fontsize=10)
        ax.set_title(keys[tipo], fontsize=14, color=colors[keys[tipo]], style='italic')
        ax.tick_params(axis='y', labelsize=12)

# Etiquetas de ejes


axB00.set_ylabel(r'growth (day$^{-1}$)', fontsize=15)
axB05.set_ylabel(r'growth (day$^{-1}$)', fontsize=15)

for ax in [axB00, axB01, axB02, axB03, axB04]:
    ax.tick_params(axis='both', direction='in', top=True, right=True)

# Guardar figura
mp.savefig('./death_boxplots.pdf', dpi=300, format='pdf', bbox_inches='tight')

# Hacer ANOVA
print("\nResultados del ANOVA por especie:")
for tipo, listas in data_para_anova.items():
    if len(listas) > 1:
        f_stat, p_val = f_oneway(*listas)
        print(f"{keys[tipo]} (ID: {tipo}): F = {f_stat:.3f}, p = {p_val:.3e}")
    else:
        print(f"{keys[tipo]} (ID: {tipo}): Solo aparece en una comunidad, no se puede hacer ANOVA")
