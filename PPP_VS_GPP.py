#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: olympio
"""
import numpy as np
import data_gen as dg
from aux_functions import *
import train_classifier as tc
from numpy.random import default_rng
rng = default_rng()
import dpp
#%% Data generation
N1=100 #Number of points in class 1
N2=100 #Number of points in class 2
size_train = 200
size_test = 200
size_tot = size_train + size_test
    
data = []
Y=[]
for i in range(size_tot//2): #150 first are spheres
    X = dpp.sample(np.sqrt(N1), kernel=dpp.kernels['ginibre'])
    X_GPP=np.zeros((len(X), 2))
    X_GPP[:,0]=np.real(X)
    X_GPP[:,1]=np.imag(X)
    intervals = dg.gen_dgm(X_GPP, [1])[0]
    intervals = rot_diag(intervals)
    data.append(intervals)      
    Y.append(1)
for i in range(size_tot//2): #150 second are tori
    X_PPP = dg.gen_disk(np.sqrt(N2), N2)
    intervals = dg.gen_dgm(X_PPP, [1])[0]
    intervals = rot_diag(intervals)
    data.append(intervals)
    Y.append(-1)
         
print('data generated')
 
data=np.array(data)
Y=np.array(Y)
    
#Split train and test
data_train, data_test, Y_train, Y_test = dg.split_train_test(size_train, size_test, data, Y)
 #%%Learning
#Generate grid of parameters 
N_size = 11
size_balls = np.linspace(0, 0.5, N_size)[1:] 
thresh = np.array([1,2, 3,4, 5])
T=10
N_centers_tot = 100 
N_sub = 20
C=tc.get_centers_all_dataset(data_train, N_centers_tot)
C_arr=[]
for t in range(T):
    sub_ind = rng.choice(np.shape(C)[0], size = N_sub, replace = False)
    C_arr.append(C[sub_ind])

#Boosting
scores = np.zeros(T)
balls_centers, balls_sizes, best_thresh, grad_weights, switch = tc.boosting_ball(C_arr, size_balls, thresh, Y_train, data_train, T, switch_labels=True)
for t in range(1, T+1):
    scores[t-1] = tc.validation_boosting_ball(balls_centers, balls_sizes, best_thresh, grad_weights, switch, data_test, Y_test, t)
print(scores)
    
    #%%Viz
