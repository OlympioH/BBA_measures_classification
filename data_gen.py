#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: olympio
"""
import numpy as np
import gudhi as gd
import gudhi.representations
import tadasets
import random as rd
import dppy


def gen_disk(radius, n_pts):
    R = radius*np.sqrt(np.random.rand(n_pts))
    theta = 2*np.pi*np.random.rand(n_pts)
    X=np.zeros((n_pts, 2))
    X[:,0]=R*np.cos(theta)
    X[:,1]=R*np.sin(theta)
    return (X)

def gen_torus(N_pts, noise, a, c): #a and c intervals
    a = a[0]+(a[1]-a[0])*np.random.rand() 
    c = c[0]+(c[1]-c[0])*np.random.rand()
    X = np.zeros((N_pts, 3))
    n_filled = 0
    while n_filled <N_pts:
        theta = np.random.uniform(0, 2*np.pi)
        eta = np.random.uniform(0, 1/np.pi)
        fx = (1+(a/c)*np.cos(theta))/(2*np.pi)
        if eta < fx:
            phi = np.random.uniform(0, 2*np.pi)
            X[n_filled] = [(c+a*np.cos(theta))*np.cos(phi),(c+a*np.cos(theta))*np.sin(phi), a*np.sin(theta)]
            n_filled+=1
    return X+noise*np.random.randn(*X.shape)

def gen_torus_non_uniform(N_pts, noise, a, c):
    a = a[0]+(a[1]-a[0])*np.random.rand() 
    c = c[0]+(c[1]-c[0])*np.random.rand()
    return(tadasets.torus(N_pts, c, a, noise))

def gen_sphere(N_pts, noise, r): #r interval
    r = r[0]+ (r[1]-r[0])*np.random.rand() 
    return (tadasets.dsphere(N_pts,2,  r, noise))

def gen_dgm(X, dim): #dim list of dimensions on which to consider the homology
    AC = gd.AlphaComplex(X).create_simplex_tree()
    dgm = AC.persistence(min_persistence = 1e-4)
    intervals=[]
    for d in dim:
        del_ind =[]
        int_d = AC.persistence_intervals_in_dimension(d)
        if d==0:
            for i in range(len(int_d)): #remove infinite bars
                if int_d[i][1]==np.inf:
                    del_ind.append(i)
            int_d=np.delete(int_d, del_ind, axis=0)
        intervals.append(int_d)
    return intervals


def split_train_test(size_train, size_test, data, Y):
    size_tot = size_train+size_test
    train_indices=rd.sample(range(size_tot), size_train)
    train_indices = np.sort(train_indices)
    test_indices=np.delete(np.arange(size_tot), train_indices)
    test_indices = np.sort (test_indices)
    data_train=data[train_indices]
    data_test=data[test_indices]
    Y_train=Y[train_indices]
    Y_test=Y[test_indices]
    return data_train, data_test, Y_train, Y_test

def gen_gin(N_pts):
    X = dppy.sample(np.sqrt(N_pts), kernel=dppy.kernels['ginibre'], quiet=True)
    X_gin=np.zeros((len(X), 2))
    X_gin[:,0]=np.real(X)
    X_gin[:,1]=np.imag(X)
    return(X_gin)

def gen_orbit(N_pts, rho):
    X=np.zeros((N_pts, 2))
    X[0]=np.random.rand(2)
    for i in range(1, N_pts):
        x, y = X[i-1]
        x_new=(x+rho*y*(1-y))%1
        y_new=(y+rho*x_new*(1-x_new))%1
        X[i]=np.array([x_new, y_new])
    return X

        
        
        