#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:55:05 2022

@author: olympio
"""

#%% Modules
import numpy as np
import matplotlib.pyplot as plt
import data_gen as dg
import graphs_tools_experiments as gr
from aux_functions import *
import train_classifier as tc
def der_dist(tab, step):
    der_tab = (tab[:-1]-tab[1:])/step
    return np.linalg.norm(der_tab)
from numpy.random import default_rng
rng = default_rng()

#%% Loading graph data
# Available options are: `"MUTAG"`,`"COX2"`, `"DHFR"`, `"PROTEINS"`, `"NCI1"`, `"NCI109"`,`"IMDB-BINARY"`, `"IMDB-MULTI"`. '"REDDIT5K"', '"REDDIT12K"', '"COLLAB"'

dataset = "MUTAG"

visualize = True

gr.generate_diagrams_and_features(dataset, path_dataset="./"+dataset+"/")
diags_dict, F, L = gr.load_data(dataset, path_dataset="./"+dataset+"/", verbose=True)

F = np.array(F, dtype=np.float64)
get_id_class = lambda x:np.where(x==1)[0][0]
labels = np.apply_along_axis(get_id_class, 1, L)
Y = 2*labels-1 #from {0, 1} to {-1, 1}
n_data = F.shape[0]

print('Available dgms:', diags_dict.keys())			# Helps for the choice of t
size_test = int(0.1*n_data)
size_train = n_data-size_test
data=[]
t=10.0
for i in range(n_data):
    Ord0, Rel1, Ext0, Ext1 = diags_dict['Ord0_'+str(t)+'-hks'][i], diags_dict['Rel1_'+str(t)+'-hks'][i], diags_dict['Ext0_'+str(t)+'-hks'][i], diags_dict['Ext1_'+str(t)+'-hks'][i]
    diag = np.concatenate((Ord0, Ext1), axis=0)
    #diag = Rel1
    diag = np.unique(diag, axis=0)
    intervals = rot_diag(diag)
    data.append(intervals)
data = np.array(data)
score = 0

data_train, data_test, Y_train, Y_test = dg.split_train_test(size_train, size_test, data, Y) 


#%% Boosting

N_radius = 15
size_balls = np.linspace(0, 0.08, N_radius)[1:] 
thresh = np.array([1,2, 3, 4, 5])
T=5
N_balls = 50 
N_sub = 20
C=tc.get_centers_all_dataset(data_train, N_balls)
C_arr=[]
for t in range(T):
    sub_ind = rng.choice(np.shape(C)[0], size = N_sub, replace = False)
    C_arr.append(C[sub_ind])
balls_centers, balls_sizes, best_thresh, grad_weights, switch = tc.boosting_ball(C_arr, size_balls, thresh, Y_train, data_train, T, switch_labels=True)

#%% Validation boosting
scores = np.zeros(T)
for t in range(1, T+1):
    scores[t-1] = tc.validation_boosting_ball(balls_centers, balls_sizes, best_thresh, grad_weights, switch, data_test, Y_test, t)
print(scores)
