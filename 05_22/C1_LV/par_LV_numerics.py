#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@project: microbiome inference (lotka-volterra model - parameters)
@author: Roman Zapien-Campos - 2023
"""

# Import packages
import pickle as pc


# Import parameters used in simulations
with open('./data/C1_abs_abund.pickle', 'rb') as f:
    data_par = pc.load(f)
# Number of microbial types
n_types = data_par['n_types']
# Time simulated
t_simulated = 72
# Sampling time points
t_points = data_par['t_points']
sampling_times = data_par['sampling_times']
# Initial absolute abundance
init_abs_abund = data_par['init_abs']
# Initial relative abundance
with open('./data/C1_rel_abund.pickle', 'rb') as f:
    data_par = pc.load(f)
init_rel_abund = data_par['init_rel']
#init_rel_abund = data_par['init_rel_abund']
# Threshold to stop diverging numerical simulations
upper_threshold = 3E7