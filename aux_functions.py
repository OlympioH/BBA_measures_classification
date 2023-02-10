#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:49:53 2021

@author: olympio
"""
import matplotlib.pyplot as plt
import numpy as np
import gudhi as gd
import tensorflow as tf

def sigmo(x):
    return (1/(1+tf.math.exp(-x)))

def plot3Dmfold(X, Y): #plot a function Y=f(X) as a color map where X 3D array of the points coordinates in the sphere
    fig=plt.figure()
    ax=plt.axes(projection='3d')
    surf=ax.scatter(X[:,0], X[:,1], X[:,2], c=Y)
    fig.colorbar(surf,shrink=0.5, aspect=5)
    
def rot(X, theta): #X 2D array
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return (np.dot(R, X))

def rot_PC (PC, theta):
    rotated_PC = np.zeros((len(PC), 2))
    for k, x in enumerate(PC):
        rotated_PC[k] = rot(x, theta)
    return rotated_PC

def rot_diag(PC):
    PC[:,1] = PC[:,1]-PC[:,0]
    return PC
        
def count_ball(X, center, radius):
    return (np.sum(np.linalg.norm(X-center, axis = 1)<=radius))

def find_bounds(data):
    N = len(data)
    max_x = np.zeros(N)
    max_y = np.zeros(N)
    for k,  interval in enumerate(data):
        max_x[k] = np.max(interval[:,0])
        max_y[k] = np.max(interval[:,1])
    return np.max(max_x), np.max(max_y)


def is_in_rectangle(x, rectangle):
    if rectangle[0]<= x[0] <= rectangle[1]:
        if rectangle[2]<= x[1] <= rectangle[3]:
            return True
    return False

def is_in_hypercube(x, hypercube): #hypercube has format (center, size) in Rd x R
    center = hypercube[0]
    size = hypercube[1]
    if (center-size<=x).all():
        if (center+size>=x).all():
            return True
    return False
def count_hypercube (X, hypercube):
    counter = 0
    for x in X:
        if is_in_hypercube(x, hypercube):
            counter+=1
    return counter
def count_rectangle(X, rectangle, weight_function = lambda x:1):
    counter = 0
    for x in X:
        if is_in_rectangle(x, rectangle):
            counter+=weight_function(x[1])
    return counter

def total_mass(X, weight_function = lambda x:1):
    mass=0
    for x in X:
        mass+=weight_function(x[1])
    return mass
        
def plot_rectangle(rectangle, col ='r', lab ='Best rectangle'):#rectangle has format [a, b, c, d]
    rectangle_coords = np.array([[rectangle[0], rectangle[2]], [rectangle[0], rectangle[3]], [rectangle[1], rectangle[3]], [rectangle[1], rectangle[2]],[rectangle[0], rectangle[2]]])
    plt.plot(rectangle_coords[:,0], rectangle_coords[:,1], color = col, label = lab)
    plt.legend()
    
def mass_gaussian(loc, center, l1, l2, sigma):
    #Bounds in x-axis :
    r2 = tf.math.sqrt(2.)
    lb = center[0]-l1
    ub = center[0]+l1
    mass_x = (tf.math.erf((ub-loc[0])/(r2*sigma))-tf.math.erf((lb-loc[0])/(r2*sigma)))
    #Bounds in y-axis :
    lb = center[1]-l2
    ub = center[1]+l2
    mass_y = (tf.math.erf((ub-loc[1])/(r2*sigma))-tf.math.erf((lb-loc[1])/(r2*sigma)))
    return (mass_x*mass_y)

def mass_gaussian_alt(loc, center, l1, l2, sigma, res=40):
    Im = np.histogram2d(loc[:,0], loc[:,1], bins = [res, res])
    

def mass_PC(locs, center, l1, l2, sigma):
    tot_mass = 0
    for loc in locs :
        tot_mass += mass_gaussian(loc, center, l1, l2, sigma)
    return tot_mass
        
def sq_dist_to_rec(point, center, l1, l2):
    dx = tf.math.maximum(tf.abs(point[0]-center[0])-tf.abs(l1), 0)
    dy = tf.math.maximum(tf.abs(point[1]-center[1])-tf.abs(l2), 0)
    return (tf.math.multiply(dx, dx)+tf.math.multiply(dy, dy))

def lap_dist_to_point(PC, center, std):
    dist = tf.math.reduce_euclidean_norm(PC-center, axis = 1)
    return (tf.reduce_sum(tf.exp(-std*dist)))

def lap_dist_to_ball(PC, center, std, radius):
    dist = tf.cast(tf.math.reduce_euclidean_norm(PC-center, axis = 1)>=radius, dtype=tf.float64)*(tf.math.reduce_euclidean_norm(PC-center, axis = 1)-radius)
    return (tf.reduce_sum(tf.exp(-std*dist)))
    
def smoothed_lap_dist_to_point(PC, center, std, radius):
    dist = tf.math.reduce_euclidean_norm(PC-center, axis = 1)
    dist = sigmo(dist-radius)
    return (tf.reduce_sum(tf.exp(-std*dist)))

def lap_dsit_to_ball(PC, radius, center, std):
    dist = tf.math.reduce_euclidean_norm((PC-center), axis = 1)

def mass_tot_exp_dist(PC, center, l1, l2, lbd):
    sq_dist = sq_dist_to_rec(PC.T, center, l1, l2)
    return (tf.reduce_sum(tf.exp(-lbd*sq_dist)))

    