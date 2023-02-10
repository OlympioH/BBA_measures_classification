#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:43:10 2021

@author: olympio
"""

import matplotlib.pyplot as plt
import numpy as np
import gudhi as gd
import os
import tadasets
os.chdir('/home/olympio/Documents/Boulot/Thèse/Lasso_PI/Experiments')
import data_gen as dg
from aux_functions import *
import train_classifier as tc
from data_gen_curvature import distance_matrix
from numpy.random import default_rng
rng = default_rng()
import dpp
import time
#%%
N1=500 #Number of points in image 1
N2=500 #Number of points in image 2
sigmas = np.arange(0.1, 0.6, 0.5)
scores = np.zeros(len(sigmas))
size_train = 200
size_test = 200
size_tot = size_train + size_test
for alpha, sigma in enumerate(sigmas):
    print(sigma)
    data = []
    Y=[]
    for i in range(size_tot//2): #150 first are spheres
        #N1 = np.random.randint(300, 1500)
        #X=dg.gen_sphere(N1, sigma, [6, 6])
        X=dg.gen_torus(N2, sigma, [2, 3], [3, 4])
        # X = dpp.sample(np.sqrt(100), kernel=dpp.kernels['ginibre'])
        # X_gin=np.zeros((len(X), 2))
        # X_gin[:,0]=np.real(X)
        # X_gin[:,1]=np.imag(X)
        #DM = distance_matrix(2, N1)
        #intervals = dg.gen_dgm_from_DM(DM, [1])[0]
        #X = dg.gen_orbit(N1, 2.5)
        #data.append(X)
        intervals = dg.gen_dgm(X, [1])[0]
        #intervals = dg.gen_dgm(X_gin, [1])[0]
        intervals = rot_diag(intervals)
        data.append(intervals)
       
        Y.append(1)
    for i in range(size_tot//2): #150 second are tori
        #N2 = np.random.randint(300, 1500)
        X=dg.gen_sphere(N2, sigma, [5, 7])
        #X = dg.gen_orbit(N2, 3.5)
        #data.append(X)
        #X_unif = dg.gen_disk(np.sqrt(100), 100)
        #DM = distance_matrix(4, N2)
        #intervals = dg.gen_dgm_from_DM(DM, [1])[0]
        #intervals = dg.gen_dgm(X_unif, [1])[0]
        intervals = dg.gen_dgm(X, [1])[0]
        intervals = rot_diag(intervals)
        data.append(intervals)
        Y.append(-1)
        

        
    print('data generated')
     
    data=np.array(data)
    Y=np.array(Y)
        
    #Split train and test
    data_train, data_test, Y_train, Y_test = dg.split_train_test(size_train, size_test, data, Y)
 #%%Balls   
    N_box = 11
    size_box = np.linspace(0, 0.5, N_box)[1:] 
    thresh = np.array([1,2, 3,4, 5])
    T=1
    N_pts = 100 
    N_sub = 20
    C=tc.get_centers_all_dataset(data_train[2900:3100], N_pts)
    C_arr=[]
    for t in range(T):
        sub_ind = rng.choice(np.shape(C)[0], size = N_sub, replace = False)
        C_arr.append(C[sub_ind])
    start=time.time()
    balls_centers, balls_sizes, best_thresh, grad_weights, switch = tc.boosting_ball(C_arr, size_box, thresh, Y_train, data_train, T, switch_labels=True)
    end=time.time()
    print('Computational time for one weak-classifier:', end-start)
    
    scores = np.zeros(T)
    for t in range(1, T+1):
        scores[t-1] = tc.validation_boosting_ball(balls_centers, balls_sizes, best_thresh, grad_weights, switch, data_test, Y_test, t)
    print(scores)
    
    #%%Viz
    plt.scatter(data[0][:,0], data[0][:,1])
    plt.scatter(data[-1][:,0], data[-1][:,1])
    plt.scatter(C_arr[0][:,0], C_arr[0][:,1])
    

    #%%Rectangles 

    #Learning parameters
    #x_max, y_max = find_bounds(data_train)
    x_max, y_max = 2, 3 
    #x_max = 6 #manually set x_max to remove outliers
    N_pts = 7 #sqrt of the number of points
    N_box = 10
    size_box = np.linspace(0, 0.5, N_box) #change 3 by better parameter
    thresh = np.array([1,2, 3,4, 5])
    #Training
    C = tc.gen_C(x_max, y_max, N_pts)
    rectangle1, s1 = tc.find_best_rectangle(C, size_box, thresh, Y_train, data_train, weights = np.ones(len(data_train)))
    print('training done')
    
    #Validation
    scores[alpha]=tc.validation_score(rectangle1, s1, data_test, Y_test)
    misclass = tc.misclassified(rectangle1, s1, data_train, Y_train)
    print(scores)
    
    #%%
    """Test Boosting"""
    x_max, y_max = 2, 4 
    #x_max = 6 #manually set x_max to remove outliers
    N_box = 20
    N_pts= 20
    C = tc.gen_C(x_max, y_max, N_pts)
    size_box = np.linspace(0, 2.5, N_box)[1:] #change 3 by better parameter
    thresh = np.array([1,2, 3,4, 5, 6, 7])
    T=5
    N_subsamples = 20
    C_arr=[]
    for t in range(T):
        sub_ind = rng.choice(N_pts**2, size = N_subsamples, replace = False)
        C_arr.append(C[sub_ind])
    rectangles, best_thresh, grad_weights, switch = tc.boosting(C_arr, size_box, thresh, Y_train, data_train, T, switch_labels=True)
#score
    for t in range(1, T+1):
        scores = tc.validation_boosting(rectangles, best_thresh, grad_weights, switch, data_test, Y_test, t)
        print(scores)

    #%% Plot multiple rectangles
    plt.scatter(data[3][:,0], data[3][:,1], label = 'Torus')
    plt.scatter(data[-1][:,0], data[-1][:,1], label = 'Sphere')
    for k, rectangle in enumerate(rectangles):
        plot_rectangle(rectangle, col = np.random.rand(3), lab='rectangle %s, threshold = %d' %(k, best_thresh[k]))
    plt.legend()
        
    #%%
plt.figure()
plt.scatter(data[0][:,0], data[0][:,1], label = 'K=2')
plt.scatter(data[-1][:,0], data[-1][:,1], label = 'K=4')
plot_rectangle(rectangle1, col = 'r')
plt.title('Classification : K=2 iff more than %s points in the rectangle' %(s1))
plt.legend()

# plt.figure()
# plt.plot(sigmas, scores)  
# plt.xlabel('Noise std')
# plt.ylabel('Classification score')
# plt.title('Classification torus vs sphere random radii, N=1000 points, trained on 50 tested on 250, classification by best square')

#%% Study total mass
N1=100 #Number of points in image 1
N2=100 #Number of points in image 2
sigmas = np.arange(0,0.5,0.5)
size_train = 50
size_test = 50
size_tot = size_train + size_test
for alpha, sigma in enumerate(sigmas):
    print(sigma)
    data = []
    Y=[]
    for i in range(size_tot//2): #150 first are spheres
        X=dg.gen_sphere(N1, sigma, [6, 6])
        intervals = dg.gen_dgm(X, [1])[0]
        intervals = rot_diag(intervals)
        data.append(intervals)
        Y.append(-1)
    for i in range(size_tot//2): #150 second are tori
        X=dg.gen_torus(N2, sigma, [2, 2], [4, 4])
        intervals = dg.gen_dgm(X, [1])[0]
        intervals = rot_diag(intervals)
        data.append(intervals)
        Y.append(1)
        
    print('data generated')
     
    data=np.array(data)
    Y=np.array(Y)
masses = np.zeros(len(data))
for k, d in enumerate(data):
    masses[k] = total_mass(d, weight_function = lambda x:x**3)
    
plt.plot(masses)

#%%
sigma = 1
X1 = dg.gen_sphere(N1, sigma, [6, 6])
X2 = dg.gen_torus(N2, sigma, [2, 2], [4, 4])
Xconc = np.concatenate([X1, X2], axis = 0)
Y=np.ones(N1+N2)
Y[:N1] = -1
plot3Dmfold(Xconc, Y)