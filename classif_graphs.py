#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:55:05 2022

@author: olympio
"""

#%% Modules
import os
os.chdir("/home/olympio/Documents/Boulot/Thèse/Lasso_PI/Experiments")
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
import time
#%% Loading data
# Available options are: `"MUTAG"`,`"COX2"`, `"DHFR"`, `"PROTEINS"`, `"NCI1"`, `"NCI109"`,`"IMDB-BINARY"`, `"IMDB-MULTI"`. '"REDDIT5K"', '"REDDIT12K"', '"COLLAB"'
# NB: small one is MUTAG

dataset = "MUTAG"

visualize = True

gr.generate_diagrams_and_features(dataset, path_dataset="./data_graphs/"+dataset+"/")
diags_dict, F, L = gr.load_data(dataset, path_dataset="./data_graphs/"+dataset+"/", verbose=True)

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

#%%

PI_model = gd.representations.PersistenceImage(bandwidth=0.01, weight=lambda x: x[1]**2, resolution=[40,40])
t1=time.time()
data_PI=[]
for i in range(size_train):
    interval = data_train[i]
    PI = PI_model.fit_transform ([interval]) [0]
    data_PI.append(PI)

t2=time.time()

print('vectorization time:', (t2-t1))
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
data_PI=np.array(data_PI)
Y=np.array(Y)
#% Logistic regression 

#data_PI_train, data_PI_test, Y_train, Y_test = dg.split_train_test(size_train, size_test, data_PI, Y)
    
t3=time.time()
clf = LogisticRegression(solver = 'liblinear', penalty = 'l2', n_jobs = 4)
clf.fit(data_PI, Y_train)
t4=time.time()
print('classification time', t4-t3)
print('classification score', clf.score(data_PI_test, Y_test))

#%%

batch_size = 50
N_centers = 10
C=tc.get_centers_all_dataset(data_train[:100], N_centers)
for c in range(N_centers):
    print(c)
    mean_mass=0
    center = C[c]
    #std = np.random.rand(1)
    std=np.array([1])
    for d in data_train:
        mean_mass+=lap_dist_to_point(d, center, std)
    mean_mass/=size_train
    thresh = np.random.rand(1)
    betainit = np.concatenate([center, std, thresh])
    beta = tf.Variable(initial_value=np.array(betainit[:,np.newaxis], dtype=np.float64), trainable=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)      
    #Gradient descent
    losses, betas = [], []
    N_epochs = 100
    start = time.time()
    for epoch in range(N_epochs):   #number of epochs of SGD
        with tf.GradientTape() as tape:
            #sigma_gauss = beta[5]
            loss = 0
            #idx = np.random.permutation(size_train)
            #data_train, Y_train = data_train[idx], Y_train[idx]
            batch_ind = np.random.randint(0, size_train, batch_size)
            batch_subset = data_train[batch_ind]
            Y_subset = Y_train[batch_ind]
            for k, d in enumerate(batch_subset):
        
        
                    
                    # Compute persistence diagram
                    # Cross-entropy loss
                    #p=sigmo(mass_PC(d, beta[0:2], beta[2], beta[3], beta[5])/mass_tot-beta[4])
                p = sigmo (lap_dist_to_point(d, tf.transpose(beta[:2]), beta[2])/mean_mass - beta[2+1])
                loss-=(Y_subset[k]*tf.math.log(p)+(1-Y_subset[k])*tf.math.log(1-p))
                                    
                # Compute and apply gradients
        gradients = tape.gradient(loss, [beta])
        optimizer.apply_gradients(zip(gradients, [beta]))
                    
        losses.append(loss.numpy())
                    #dgms.append(dgm)
        betas.append(beta.numpy()) 
    beta_final=betas[-1]
    
    end=time.time()
    print("computational time for training one weak classifier", 2*(end-start))


#%%
x_max, y_max = 0.2, 0.1
#x_max = 6 #manually set x_max to remove outliers
N_pts = 100 #sqrt of the number of points
N_box = 15
size_box = np.linspace(0, 0.04, N_box)[1:] 
thresh = np.array([1,2, 3,4, 5])
#Training
#C = tc.gen_C(x_max, y_max, N_pts)
C=tc.get_centers_all_dataset(data_train, N_pts)
rectangle1, s1 = tc.find_best_rectangle(C, size_box, thresh, Y_train, data_train, weights = np.ones(len(data_train)))
print('training done')

#Validation
score=tc.validation_score(rectangle1, s1, data_test, Y_test)
misclass = tc.misclassified(rectangle1, s1, data_train, Y_train)
print(score)


# score=0
# for i in range(10):
#     data_train, data_test, Y_train, Y_test = dg.split_train_test(size_train, size_test, data, Y) 
#     score+=tc.validation_score(rectangle1, s1, data_test, Y_test)
# score/=10
# print(score)   
 

#%% Plot single rectangle
plt.figure()
plt.scatter(data[0][:,0], data[0][:,1], label = '+1')
plt.scatter(data[-1][:,0], data[-1][:,1], label = '-1')
plt.scatter(C[:,0],C[:,1])
plot_rectangle(rectangle1, col = 'r')
plt.title('Classification : label +1 iff more than %s points in the rectangle' %(s1))
plt.legend()

#%% Boosting
sub_ind = np.random.randint(0, 4500, 1000)
data_train = data_train[sub_ind]
Y_train = Y_train[sub_ind]

N_box = 10
size_box = np.linspace(0, 0.04, N_box)[1:] 
thresh = np.array([1,2, 3])
T=10
N_pts = 75 
N_sub = 20
C=tc.get_centers_all_dataset(data_train[:2], N_pts)
C_arr=[]
start = time.time()
for t in range(T):
    sub_ind = rng.choice(np.shape(C)[0], size = N_sub, replace = False)
    C_arr.append(C[sub_ind])
balls_centers, balls_sizes, best_thresh, grad_weights, switch = tc.boosting_ball(C_arr, size_box, thresh, Y_train, data_train, T, switch_labels=True)
end = time.time()
print((end-start)/T)
#%% Validation boosting
scores = np.zeros(T)
for t in range(1, T+1):
    scores[t-1] = tc.validation_boosting_ball(balls_centers, balls_sizes, best_thresh, grad_weights, switch, data_test, Y_test, t)
print(scores)

#%% Plot multiple balls
fig, ax = plt.subplots()
plt.scatter(data_test[0][:,0], data_test[0][:,1], label = 'class -1')
plt.scatter(data_test[-1][:,0], data_test[-1][:,1], label = 'class 1')
for t in range(3):
    circle=plt.Circle((balls_centers[t]), balls_sizes[t], fill=False)
    ax.add_patch(circle)
plt.legend()
#%%
save_data = False
if save_data :
    np.savetxt('graphs_saved/IMDB_B/best_thresh1.txt', best_thresh)
    np.savetxt('graphs_saved/IMDB_B/grad_weights1.txt', grad_weights)
    np.savetxt('graphs_saved/IMDB_B/rectangles1.txt', rectangles)
    np.savetxt('graphs_saved/IMDB_B/switch1.txt', switch)

#%% Gradient learning
# if -1 in Y :
#     Y=(Y+1)/2
# data_train, data_test, Y_train, Y_test = dg.split_train_test(size_train, size_test, data, Y) 
# #data_train = data_train_copy
# #size_train= len(data_train)
# N_centers = 10
# batch_size = 100
# C=tc.get_centers_all_dataset(data_train, N_centers)
# scores_test = np.zeros(N_centers)
# scores_train = np.zeros(N_centers)
# #betainit = np.array([0.2, 3.5, 0.4, 1, 0.1, 0.1])
# for c in range(N_centers):
#     print(c)
#     mean_mass=0
#     center = C[c]
#     std = np.random.rand(1)
#     radius = np.random.rand(1) 
#     for d in data_train:
#         mean_mass+=lap_dist_to_ball(d, center, std, radius)
#     mean_mass/=size_train
#     thresh = np.random.rand(1)
#     betainit = np.concatenate([center, std, thresh, radius])
#     beta = tf.Variable(initial_value=np.array(betainit[:,np.newaxis], dtype=np.float64), trainable=True)
#     optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)      
#     #Gradient descent
#     losses, betas = [], []
#     N_epochs = 150
#     for epoch in range(N_epochs):   #number of epochs of SGD
#         with tf.GradientTape() as tape:
#             #sigma_gauss = beta[5]
#             loss = 0
#             #idx = np.random.permutation(size_train)
#             #data_train, Y_train = data_train[idx], Y_train[idx]
#             batch_ind = np.random.randint(0, size_train, batch_size)
#             batch_subset = data_train[batch_ind]
#             Y_subset = Y_train[batch_ind]
#             for k, d in enumerate(batch_subset):
        
        
                    
#                     # Compute persistence diagram
#                     # Cross-entropy loss
#                     #p=sigmo(mass_PC(d, beta[0:2], beta[2], beta[3], beta[5])/mass_tot-beta[4])
#                 p = sigmo (lap_dist_to_ball(d, tf.transpose(beta[:2]), beta[2], beta[2+2])/mean_mass - beta[2+1])
#                 loss-=(Y_subset[k]*tf.math.log(p)+(1-Y_subset[k])*tf.math.log(1-p))
                                    
#                 # Compute and apply gradients
#         gradients = tape.gradient(loss, [beta])
#         optimizer.apply_gradients(zip(gradients, [beta]))
                    
#         losses.append(loss.numpy())
#                     #dgms.append(dgm)
#         betas.append(beta.numpy()) 
#     beta_final=betas[-1]
    
#     score_test = tc.validation_dist_ball(beta_final, mean_mass, data_test, Y_test, 2)
#     score_train = tc.validation_dist_ball(beta_final, mean_mass, data_train, Y_train, 2)
#     scores_test[c] = score_test
#     scores_train[c] = score_train
#     print(scores_train)
#     print(scores_test)
    