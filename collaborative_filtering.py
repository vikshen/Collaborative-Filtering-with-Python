#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:33:20 2018

@author: vikasshenoy
"""

#####################################################
# Collaborative Filtering for Loan Recommendations  #
#####################################################
import numpy as np
import random
import csv
import locale # for currency formatting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def split(param, n_lenders, n_loans, n_features):
    X = param[:n_loans*n_features]
    Theta = param[n_loans*n_features:] 
    X = X.reshape(n_loans,n_features)
    Theta = Theta.reshape(n_lenders,n_features)
    return X, Theta

def costFun(param, Y, r, n_lenders, n_loans, n_features, lamba):
    # returns the cost
    reg1=0
    reg2=0
    J=0
    X, Theta = split(param, n_lenders, n_loans, n_features)
    for i in range(n_loans):
        for j in range(n_lenders):
            if(r[i,j]==1):
                J+=(np.dot(Theta[j].T,X[i])-Y[i,j])**2
    J=J/2    
    
    for i in range(n_features):
        reg1+=(np.sum(X[:,i])**2)*lamba/2
        reg2+=(np.sum(Theta[:,i])**2)*lamba/2
    
    J = J+reg1+reg2
    return J

def costGrad(param, Y, r, n_lenders, n_loans, n_features, lamba): 
    # return the gradient
    X, Theta = split(param, n_lenders, n_loans, n_features)
    XGrad = np.zeros((10,100))
    for k in range(n_features):
        for i in range(n_loans):
            for j in range(n_lenders):
                if(r[i,j]==1):
                    XGrad[k,i]=(np.dot(Theta[j].T,X[i])-Y[i,j])*Theta[k,j]+lamba*X[i,k]
   
    ThetaGrad = np.zeros((10,10))
    for k in range(n_features):
        for j in range(n_lenders):
            for i in range(n_loans):
                if(R[i,j]==1):
                    ThetaGrad[j,k] = (np.dot(Theta[j].T,X[i])-Y[i,j])*X[i,k]+lamba*Theta[k,j]

    grad = [XGrad[:],ThetaGrad[:]]
    return grad
    
def optimizeCost(param, Y, r, n_lenders, n_loans, n_features, lamba, step, maxrun):
    X, Theta = split(param, n_lenders, n_loans, n_features)
    
    (m,n) = X.shape
    (tm,tn) = Theta.shape
    costRange = np.matrix(np.zeros((maxrun,1)))
    for j in list(range(0,maxrun)):
        costRange[j,0] = costFun(param, Y, r, n_lenders, n_loans, n_features, lamba)

        (dX,dTheta) = costGrad(param, Y, r, n_lenders, n_loans, n_features, lamba)
        dX = dX.reshape(n_loans * n_features, 1, order = "F")
        dTheta = dTheta.reshape(n_lenders * n_features, 1, order = "F")
        
        for i in list(range(0,n_loans * n_features)):
            param[i] = param[i] - step * (1/m) * (dX[i] + (lamba/m) * param[i])
        for k in list(range(0,n_lenders * n_features)):
            param[k] = param[k] - step * (1/tm) * (dTheta[k] + (lamba/tm) * param[k])

    param = np.asarray(param)
    return param,costRange

locale.setlocale(locale.LC_ALL, '')
datPath = r"/Users/vikasshenoy/Desktop/Financial Data Mining/assignment_09/loandata.csv"
with open(datPath, newline='') as csvfile:
    csvData = csv.reader(csvfile)
    datList = []
    for row in csvData:
        datList.append(row)
txt = np.array(datList.pop(0))  # get the colnames in the first row and remove it
Y = np.array(datList)[range(100),:]
Y = Y.astype(float)
R = (Y != 0)
R = R.astype(float)

datPath = r"/Users/vikasshenoy/Desktop/Financial Data Mining/assignment_09/loan.csv"
with open(datPath, newline='') as csvfile:
    csvData = csv.reader(csvfile)
    datList = []
    for row in csvData:
        datList.append(row)
header = np.array(datList.pop(0)) # get the colnames in the first row and remove it
info = np.array(datList)[range(100),:]

n_lenders = np.size(Y, 1)
n_loans = np.size(Y, 0)
n_features = 10
# Initialization
X = np.random.normal(loc = 0.0, scale = 1.0, size = (n_loans, n_features))
Theta = np.random.normal(loc = 0.0, scale = 1.0, size = (n_lenders,n_features))
init_param = np.concatenate((X.reshape(n_loans * n_features, 1, order = "F"),
                             Theta.reshape(n_lenders * n_features, 1, order = "F")))
init_param = np.squeeze(init_param)

# Optimization
lamba = 10
maxrun = 10000
step = 0.001

param,cost_range = optimizeCost(init_param, Y, R, n_lenders, n_loans, \
                     n_features, lamba, step, maxrun)

# now plot the cost
plt.plot(cost_range,"b.",markersize=1,label="Cost") # note: this is 0-based
plt.show()
    
# Extract X and Theta from param vector
X = param[0:(n_loans * n_features)]
Theta = param[(n_loans * n_features):len(param)]
X = X.reshape(n_loans, n_features, order = "F")
Theta = Theta.reshape(n_lenders, n_features, order = "F")
pred = np.dot(X, Theta.T)

# print out top 3 ratings for each lender
top_n = 3
for j in range(n_lenders):
    rating = np.sort(pred[:, j])[::-1]
    ind = np.argsort(pred[:, j])[::-1]
    a = info[ind,:]
    print('\nTop %d recommendations for lender %d:\n' % (top_n, (j+1)))
    for i in range(top_n):
        print('Predicted rating %.1f for loan of  %s  for %s with %s purpose at %.1f percent interest\n' %
        (rating[i], locale.currency(np.float(a[i, 0]),grouping=True), a[i, 1], a[i, 6], np.float(a[i,2])))  
