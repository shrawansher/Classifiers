#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 15:57:43 2016

@author: shrawan
"""

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0,random_state=0)
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,y_combined,classifier=lr,test_idx=range(105,150))
plt.xlabel('petal length[std]')
plt.ylabel('petal width [std]')
plt.legend(loc='upper left')
plt.show()
lr.predict_proba(X_test_std[0,:])
