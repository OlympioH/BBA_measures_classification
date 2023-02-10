#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:52:18 2021

@author: olympio
"""

import matplotlib.pyplot as plt
import numpy as np
import gudhi as gd
from aux_functions import count_rectangle, mass_PC, count_ball, lap_dist_to_point, lap_dist_to_ball
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

def gen_C(x_max, y_max, N_pts, x_min=0, y_min=0):
    C = np.zeros((N_pts**2, 2))
    X = np.linspace(x_min, x_max, N_pts) #Remove low pers points
    Y = np.linspace(y_min, y_max, N_pts)
    for k, x in enumerate(X):
        for l, y in enumerate(Y):
            C[k*N_pts+l, 0]=x
            C[k*N_pts+l, 1]=y
    return C

def gen_C_sub(min_bounds, max_bounds, N_pts):
    dim = len(min_bounds)
    return (np.random.rand(N_pts, dim)*(max_bounds-min_bounds)+min_bounds)

def get_centers_all_dataset(data, N_centers): #Here N_centers is the total number of centers
    l=len(data)
    data_conc=data[0]
    for k in range(1, l):
        data_conc = np.concatenate((data_conc, data[k]))
    kmeans = KMeans(n_clusters=N_centers, random_state=0).fit(data_conc)
    return(kmeans.cluster_centers_)

def get_centers_by_data(data, N_centers): #Here, N_centers is the number of centers by datum. Total number is N_centers*len(data)
    kmeans = KMeans(n_clusters = N_centers, random_state = 0).fit(data[0])
    centers = kmeans.cluster_centers_
    for d in data[1:]:
        kmeans = KMeans(n_clusters = N_centers, random_state = 0).fit(d)
        means = kmeans.cluster_centers_
        centers = np.concatenate((centers, means))
    return (centers)

def find_best_ball(C, radii, thresh, Y_train, data_train, weights, return_error = False):
    N_box = len(radii)
    N_pts = len(C)

    Errors = np.zeros((N_pts, N_box, len(thresh)))
    for j, c in enumerate(C):
        for b, box in enumerate(radii):
            for s, t in enumerate(thresh):
                error = 0
                for k, d in enumerate(data_train):
                    counter = count_ball(d, c, box )
                    if Y_train[k]==-1:
                        if counter>=t:
                            error+=weights[k]
                    if Y_train[k]==1:
                        if counter<t:
                            error+=weights[k]
                Errors[j, b, s]=error
                    
    ind_c, ind_box, s = np.unravel_index(Errors.argmin(), Errors.shape)
    best_c = C[ind_c]
    best_box = radii[ind_box]
    best_s=thresh[s]   
    if return_error:
        return (best_c, best_box, best_s, np.min(Errors))
    else :
        return (best_c, best_box, best_s)


def find_best_rectangle(C, size_box, thresh, Y_train, data_train, weights, return_error = False, wf = lambda x : 1):
    N_box = len(size_box)
    N_pts = len(C)
    
    Errors = np.zeros((N_pts, N_box, len(thresh)))
    for j, c in enumerate(C):
        for b, box in enumerate(size_box):
            for s, t in enumerate(thresh):
                error = 0
                rectangle = [c[0]-box, c[0]+box, c[1]-box, c[1]+box]
                for k, interval in enumerate(data_train):
                    counter = count_rectangle(interval, rectangle, wf)
                    if Y_train[k]==-1:
                        if counter>=t:
                            error+=weights[k]
                    if Y_train[k]==1:
                        if counter<t:
                            error+=weights[k]
                Errors[j, b, s]=error
                    
                    
                    
                    
    ind_c, ind_box, s = np.unravel_index(Errors.argmin(), Errors.shape)
    best_c = C[ind_c]
    best_box = size_box[ind_box]
    best_s=thresh[s]   
    rectangle = [best_c[0]-best_box, best_c[0]+best_box, best_c[1]-best_box, best_c[1]+best_box]
    if return_error:
        return (rectangle, best_s, np.min(Errors))
    else :
        return (rectangle, best_s)
    

def boosting(C_arr, size_box, thresh, Y, data, T, switch_labels=True):
    m=len(data)
    data_weights=1/m*np.ones(m)
    rectangles = []
    grad_weights = np.zeros(T)
    best_thresh = np.ones(T)
    switch = np.ones(T)
    for t in range(T):
        print(t)
        rectangle, best_s , error = find_best_rectangle(C_arr[t], size_box, thresh, Y, data, data_weights, return_error=True)
        if switch_labels:
            rectangle_switch, best_s_switch , error_switch = find_best_rectangle(C_arr[t], size_box, thresh, -Y, data, data_weights, return_error=True)
            if error_switch <error:
                rectangle = rectangle_switch
                best_s = best_s_switch
                switch[t]=-1
        rectangles.append(rectangle)
        best_thresh[t]=best_s
        if error >=0.5:
            print('error too large!')
            break
        grad_weights[t]=1/2*np.log((1-error)/error)
        Z=2*np.sqrt(error*(1-error))
        misc = misclassified(rectangle, best_s, data, switch[t]*Y)
        data_weights[misc] = data_weights[misc]*np.exp(grad_weights[t])/Z
        mask = np.ones(m, bool)
        mask[misc]=False
        data_weights[mask] = data_weights[mask]*np.exp(-grad_weights[t])/Z
    return (rectangles, best_thresh, grad_weights, switch)

def boosting_ball(C_arr, size_box, thresh, Y, data, T, switch_labels=True):
    m=len(data)
    data_weights=1/m*np.ones(m)
    balls_centers=[]
    balls_sizes=[]
    grad_weights = np.zeros(T)
    best_thresh = np.ones(T)
    switch = np.ones(T)
    for t in range(T):
        print(t)
        ball_center, ball_radius, best_s , error = find_best_ball(C_arr[t], size_box, thresh, Y, data, data_weights, return_error=True)
        if switch_labels:
            ball_center_switch, ball_radius_switch, best_s_switch , error_switch = find_best_ball(C_arr[t], size_box, thresh, -Y, data, data_weights, return_error=True)
            if error_switch <error:
                ball_center = ball_center_switch
                ball_radius = ball_radius_switch
                best_s = best_s_switch
                switch[t]=-1
        balls_centers.append(ball_center)
        balls_sizes.append(ball_radius)
        best_thresh[t]=best_s
        if error >=0.5:
            print('error too large!')
            break
        if error ==0:
            print('Perfect classification : stop boosting')
            grad_weights[t]=1
            for k in range(t+1, T):
                balls_centers.append(balls_centers[0])
                balls_sizes.append(0)
            break #Keep grad_weights = 0
        grad_weights[t]=1/2*np.log((1-error)/error)
        Z=2*np.sqrt(error*(1-error))
        misc = misclassified_ball(ball_center, ball_radius, best_s, data, switch[t]*Y)
        data_weights[misc] = data_weights[misc]*np.exp(grad_weights[t])/Z
        mask = np.ones(m, bool)
        mask[misc]=False
        data_weights[mask] = data_weights[mask]*np.exp(-grad_weights[t])/Z
    return (balls_centers, balls_sizes, best_thresh, grad_weights, switch)



def validation_boosting(rectangles, thresh, grad_weights, switch, data_test, Y_test, max_iter):
    score = 0
    size_test = len(Y_test)
    for k, interval in enumerate(data_test):
        pred = 0
        for t in range(max_iter):
            if switch[t]==1:
                if count_rectangle(interval, rectangles[t])<thresh[t]:
                    pred-=grad_weights[t]
                if count_rectangle(interval, rectangles[t])>=thresh[t]:
                    pred+=grad_weights[t]
            if switch[t]==-1:
                if count_rectangle(interval, rectangles[t])>=thresh[t]:
                    pred-=grad_weights[t]
                if count_rectangle(interval, rectangles[t])<thresh[t]:
                    pred+=grad_weights[t]
        pred = np.sign(pred)
        score += (Y_test[k]==pred)       
    score/=size_test   
    return(score)
        

def validation_boosting_ball(balls_centers, balls_sizes, thresh, grad_weights, switch, data_test, Y_test, max_iter):
    size_test = len(Y_test)
    Y_pred = np.zeros(size_test)
    for k, interval in enumerate(data_test):
        pred = 0
        for t in range(max_iter):
            if switch[t]==1:
                if count_ball(interval, balls_centers[t], balls_sizes[t])<thresh[t]:
                    pred-=grad_weights[t]
                if count_ball(interval, balls_centers[t], balls_sizes[t])>=thresh[t]:
                    pred+=grad_weights[t]
            if switch[t]==-1:
                if count_ball(interval, balls_centers[t], balls_sizes[t])>=thresh[t]:
                    pred-=grad_weights[t]
                if count_ball(interval, balls_centers[t], balls_sizes[t])<thresh[t]:
                    pred+=grad_weights[t]
        pred = np.sign(pred)
        Y_pred[k]=pred    
    return(f1_score(Y_test, Y_pred, average = 'weighted'))
        

def validation_score(rectangle, best_s, data_test, Y_test):
    score = 0
    size_test = len(Y_test)
    for k, interval in enumerate(data_test):
        counter = count_rectangle(interval, rectangle)
        label = Y_test[k]
        if counter<best_s:
            if label ==-1:
                score +=1
        if counter>=best_s:
            if label ==1:
                score+=1
    score/=size_test   
    return(score)

def misclassified(rectangle, best_s, data, Y):
    misc=[]
    for k, interval in enumerate(data):
        label = Y[k]
        if count_rectangle(interval, rectangle)<best_s:
            if label ==1:
                misc.append(k)
        if count_rectangle(interval, rectangle)>=best_s:
            if label==-1:
                misc.append(k)
    return misc

def misclassified_ball(ball_center, ball_radius, best_s, data, Y):
    misc=[]
    for k, interval in enumerate(data):
        label = Y[k]
        if count_ball(interval, ball_center, ball_radius)<best_s:
            if label ==1:
                misc.append(k)
        if count_ball(interval, ball_center, ball_radius)>=best_s:
            if label==-1:
                misc.append(k)
    return misc


def validation_dist_ball(beta_best,mean_mass, data_test, Y_test, dim):
    center = beta_best[:dim]
    std = beta_best[dim]
    thresh = beta_best[dim+1]
    radius = beta_best[dim+2]
    score = 0
    size_test = len(Y_test)
    for k, d in enumerate(data_test):
        label = Y_test[k]
        pred = lap_dist_to_ball(d, center.T, std, radius)/mean_mass-thresh
        if pred>0:
            if label==1:
                score+=1
        if pred<0:
            if label==0:
                score+=1
    score/=size_test
    return(score)
        
def validation_boosting_grad_bealls(betas_best, alphas, data_test, Y_test, T, ED):
    size_test = len(Y_test)
    score = 0
    for k, d in enumerate(data_test):
        pred = 0
        for t in range(T):
            center = betas_best[:ED, t]
            std = betas_best[ED,t]
            thresh = betas_best[ED+1,t]
            radius = betas_best[ED+2, t]
            mean_mass=0
            for d in data_test:
                mean_mass+=lap_dist_to_ball(d, center, std, radius)
            mean_mass/=size_test
            pred += alphas[t]*(lap_dist_to_ball(d, center.T, std, radius)/mean_mass-thresh)
        if np.sign(pred)==Y_test[k]:
            score+=1
    return (score/size_test)

def misclassified_smooth_rect(beta_best, norm, data, Y, lambd=1):
    center = beta_best[0:2]
    l1 = beta_best[2]
    l2 = beta_best[3]
    thresh = beta_best[4]
    misc=[]
    for k, d in enumerate(data):
        label = Y[k]
        pred = mass_tot_exp_dist(d, center, l1, l2, lambd)/norm-thresh
        if pred<0:
            if label ==1:
                misc.append(k)
        if pred>0:
            if label==0:
                misc.append(k)
    return misc

    
    
def boosting_pred_from_file_balls(interval, T, j, i):
    balls_centers = np.loadtxt('orbit_%s_%d/balls_centers.txt' %(j,i))
    balls_sizes = np.loadtxt('orbit_%s_%d/balls_sizes.txt' %(j,i))
    switch = np.loadtxt('orbit_%s_%d/switch.txt' %(j,i))
    thresh = np.loadtxt('orbit_%s_%d/best_thresh.txt' %(j,i))
    grad_weights = np.loadtxt('orbit_%s_%d/grad_weights.txt' %(j,i))
    pred = 0
    for t in range(T):
        if switch[t]==1:
            if count_ball(interval, balls_centers[t], balls_sizes[t])<thresh[t]:
                pred-=grad_weights[t]
            if count_ball(interval, balls_centers[t], balls_sizes[t])>=thresh[t]:
                pred+=grad_weights[t]
        if switch[t]==-1:
            if count_ball(interval, balls_centers[t], balls_sizes[t])>=thresh[t]:
                pred-=grad_weights[t]
            if count_ball(interval, balls_centers[t], balls_sizes[t])<thresh[t]:
                pred+=grad_weights[t]
    if pred <=0:
        return j
    else : 
        return i

def one_VS_one(data_test, labels, N_labels, T):
    score = 0
    Y_pred = []
    for k, interval in enumerate(data_test):
        duels_won = np.zeros(N_labels)
        for i in range(N_labels):
            for j in range(i):
                pred = boosting_pred_from_file_balls(interval, T, j ,i)
                duels_won[pred]+=1
        winners = np.argwhere(duels_won==np.amax(duels_won)).flatten().tolist()
        winner = np.random.choice(winners)
        Y_pred.append(winner)
        #winner = np.max(winners)
        if winner==labels[k]:
            score+=1
    score/=len(data_test)
    #score = f1_score(labels, Y_pred, average = 'weighted')
    return(score)

            
    
    