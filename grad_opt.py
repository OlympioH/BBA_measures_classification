#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: olympio
"""

import matplotlib.pyplot as plt
import numpy as np
import data_gen as dg
from aux_functions import *
import train_classifier as tc
from numpy.random import default_rng
rng = default_rng()
import tensorflow as tf

N1=500 #Number of points in class 1
N2=500 #Number of points in class 2
size_train = 100
size_test = 100
size_tot = size_train + size_test
data = []
Y=[]
for i in range(size_tot//2): #Sphere data
    X=dg.gen_sphere(N1, 0, [6, 6])
    intervals = dg.gen_dgm(X, [1])[0]
    intervals = rot_diag(intervals)
    data.append(intervals)
    Y.append(0)
for i in range(size_tot//2): #Torus data
    X=dg.gen_torus(N2, 0, [2, 2], [4, 4])
    intervals = dg.gen_dgm(X, [1])[0]
    intervals = rot_diag(intervals)
    data.append(intervals)
    Y.append(1)

print('data generated')
 
data=np.array(data)
Y=np.array(Y)
    
#Split train and test
data_train, data_test, Y_train, Y_test = dg.split_train_test(size_train, size_test, data, Y)
    
#%% Learning, distance to ball

N_centers = 1
C=tc.get_centers_all_dataset(data_train, N_centers)
scores_test = np.zeros(N_centers)
scores_train = np.zeros(N_centers)
#betainit = np.array([0.2, 3.5, 0.4, 1, 0.1, 0.1])
for c in range(N_centers):
    mean_mass=0 #initialize parameters
    center = C[c]
    std = np.random.rand(1)
    radius = np.random.rand(1) 
    for d in data_train:
        mean_mass+=lap_dist_to_ball(d, center, std, radius)
    mean_mass/=size_train #Normalize threshold parameter
    thresh = np.random.rand(1)
    betainit = np.concatenate([center, std, thresh, radius])
    beta = tf.Variable(initial_value=np.array(betainit[:,np.newaxis], dtype=np.float64), trainable=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)      
    #Gradient descent
    losses, betas = [], []
    N_epochs = 40
    for epoch in range(N_epochs):   #number of epochs of SGD
        with tf.GradientTape() as tape:
            loss = 0
            for k, d in enumerate(data_train):      
                p = sigmo (lap_dist_to_ball(d, tf.transpose(beta[:2]), beta[2], beta[2+2])/mean_mass - beta[2+1])
                loss-=(Y_train[k]*tf.math.log(p)+(1-Y_train[k])*tf.math.log(1-p)) #cross-entropy loss
                                    
                # Compute and apply gradients
        gradients = tape.gradient(loss, [beta])
        optimizer.apply_gradients(zip(gradients, [beta]))
                    
        losses.append(loss.numpy())
        betas.append(beta.numpy()) 
    beta_final=betas[-1]
    
    score_test = tc.validation_dist_ball(beta_final, mean_mass, data_test, Y_test, 2)
    score_train = tc.validation_dist_ball(beta_final, mean_mass, data_train, Y_train, 2)
    scores_test[c] = score_test
    scores_train[c] = score_train
    print(scores_train)
    print(scores_test)
    


#%%Visualization
plt.scatter(data[0][:,0], data[0][:,1], label = 'Persistence diagram of a sphere')
plt.scatter(data[-1][:,0], data[-1][:,1], label = 'Persistence diagram of a torus')
plt.scatter(C[:,0], C[:,1], label = 'K-means centers')
plt.scatter(beta_final[0], beta_final[1], color = 'r', label='center of ball')
plt.legend()