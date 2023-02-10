#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:41:06 2021

@author: olympio
"""

import random as rd
import matplotlib.pyplot as plt
import numpy as np
import gudhi as gd
import os
os.chdir('/home/olympio/Documents/Boulot/Thèse/Lasso_PI/Experiments')
import data_gen as dg
from aux_functions import *
import train_classifier as tc
import dpp

N1=50 #Number of points in image 1
N2=100 #Number of points in image 2
sigmas = np.arange(1,2)
scores = np.zeros(len(sigmas))
size_train = 300
size_test = 250
size_tot = size_train + size_test
for alpha, sigma in enumerate(sigmas):
    print(sigma)
    data = []
    Y=[]
    #for i in range(size_tot//2): #150 first are uniform samplings
        # X_unif = dg.gen_disk(np.sqrt(N1),N1)
        # intervals_unif = dg.gen_dgm(X_unif, [1])
        # intervals_unif = rot_diag(intervals_unif[0])
        # data.append(intervals_unif)
        # Y.append(1)
    # for i in range(size_tot//2): #150 second are GPP
    #     X_gin = dg.gen_gin(N2)
    #     intervals_gin = dg.gen_dgm(X_gin, [1])
    #     intervals_gin = rot_diag(intervals_gin[0])
    #     data.append(intervals_gin)
    #     Y.append(1)
    for i in range(size_tot//2):
        X=dg.gen_pcp(N1, 3, 1)
        intervals = dg.gen_dgm(X, [1])
        intervals = rot_diag(intervals[0])
        data.append(intervals)
        Y.append(0)
    for i in range(size_tot//2):
        X=dg.gen_pcp(N2, 6, 1)
        intervals = dg.gen_dgm(X, [1])
        intervals = rot_diag(intervals[0])
        data.append(intervals)
        Y.append(1)
        
    print('data generated')
     
    data=np.array(data)
    Y=np.array(Y)
        
    #Split train and test
    data_train, data_test, Y_train, Y_test = dg.split_train_test(size_train, size_test, data, Y)
    
    x_max, y_max = find_bounds(data_train)
    x_max = 2
    N_pts = 10 #sqrt of the number of points
    N_box = 10
    size_box = np.linspace(0, 1 , N_box) #change 3 by better parameter
    N_thresh = 8
    #Training
    C = tc.gen_C(x_max, y_max, N_pts)
    rectangle1, s1 = tc.find_best_rectangle(C, size_box, N_thresh, Y_train, data_train)
    #rectangle2, s2 = tc.find_best_rectangle(C, size_box, N_thresh, 1-Y_train, data_train)
    print('training done')
    
    #Validation
    scores[alpha]=tc.validation_score(rectangle1, s1, data_test, Y_test)
    false_neg, false_pos = tc.misclassified(rectangle1, s1, data_train, Y_train)
    misclass= false_pos+false_neg

    
    data_mis = data_train[misclass]
    Y_mis = Y_train[misclass]
    rectangle_mis, s_mis = tc.find_best_rectangle(C, size_box, N_thresh, Y_mis, data_mis)
    
    plt.scatter(data[0][:,0], data[0][:,1], label = 'Uniform sampling')
    plt.scatter(data[-1][:,0], data[-1][:,1], label = 'PCP')
    plot_rectangle(rectangle1, lab='Best rectangle')
    plot_rectangle(rectangle_mis, col = 'm', lab = 'rectangle on misclassified data')
    plt.legend()
    
