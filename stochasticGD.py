#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 23:05:07 2016

@author: shrawan
STICHASTIC LINEAR NEURON CLASSIFIER
"""
import numpy as np
from numpy.random import seed
class StochasticGD(object):
    
    def __init__(self,eta=0.01,n_iter=50, shuffle = True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
        
    def fit(self, X,y):
        #Set weights to 0
        self._initialize_weights(X.shape[1])
        self.cost_= []
        for i in range(self.n_iter):
            #start the iteration with shuffling X and Y together
            if self.shuffle:
                X,y = self._shuffle(X,y)
            cost = []
            print("xi")
            for xi, target in zip(X,y):   
                #update the weights for each entry in X
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self
        
    def partial_fit(self, X, y):
        """Fit training data without reiitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] >1:
            for xi, target in zip(X,y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X,y)
        return self
            
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) +self.w_[0]

    def activation(self, X):
        return self.net_input(X)
        
    def predict(self, X):
        return np.where(self.activation(X) >=0.0 ,1,-1)
    
            
    def _initialize_weights(self, m):
        self.w_ = np.zeros(1+m)
        self.w_initialized= True
    
    def _shuffle(self, X,y):
        #fetches a random order of numbers from 1-n
        r = np.random.permutation(len(y))
        print "random" , X[r]
        return  X[r], y[r]

    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = target -output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] = self.eta * error
        cost = 0.5 * error**2
        return cost
    
    