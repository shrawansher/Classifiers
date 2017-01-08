#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 07:05:32 2016

@author: shrawan
"""

import numpy as np
class Perceptron(object):
    """Perceptron classifier.
    Parameters:
    ----------
    eta: float
        Learning rate between 0.0 and 1.0
    n_iter: int
        Passes over training dataset
        
    Attributes
    ----------
    w_: 1d-array
        Weights after fitting
    errors_ : list
        Number of misclassifications in every epoch
        
    """
    
    def __init__(self, eta=0.01,n_iter=10):
        self.eta =eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        """Fit Training data
        
        Parameters
        ----------
        X: {array like}, shape =[n_samples,n_features]
           Training vectors, where n_samples: number of samples
           and n_features= number of features
        y: array like, shape = [n_samples]
            Target Values
            
        Returns
        -------
        self : object
        """
        
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors =0
            for xi, target in zip(X,y):
                "zip merges x and y into a single tuple"
                update = self.eta*(target-self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0]  += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
            
        
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

                
    
       
       
