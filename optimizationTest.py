# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:05:28 2017

@author: alessandro
"""

import numpy as np


        
nIntervals = 11
maxBudget = 100
budgets = np.linspace(1,maxBudget,nIntervals)

nItems = 10

values = np.ones(shape=(nItems,len(budgets)))

m = np.zeros(shape=(nItems,len(budgets)))
h = np.zeros(shape=(nItems,len(budgets)))
h=h.tolist()


def valueForBudget(itemIdx,budget):
    idx = np.argwhere(budget<=budgets).max()
    return values[itemIdx,idx]
    
def firstRow():
    firstRow = np.zeros(len(budgets)).tolist()
    for i,b in enumerate(budgets):
        firstRow[i]=[[values[0,i]],[0],[b]]
    return firstRow

# FIRST ROW
valIdx = 0
itIdx = 1
bIdx = 2

a=0
m[0,:] = values[0,:]
h[0] = firstRow()

for i in range(1,nItems):
    for j,b in enumerate(budgets):
        #m[i,j] = np.max(m[i-1,0:j] + values[b])
        #bRipart        
        #for bi in range(0,j):
        h[i] = h[i-1][:]
        maxVal = 0
        for bi in range(0,j):
            #print (np.sum(h[i-1][valIdx]) + valueForBudget(i,b - budgets[bi]))
            #print maxVal
            if ((np.sum(h[i-1][valIdx]) + valueForBudget(i,b - budgets[bi]))>maxVal):
                val = h[i-1][bi][valIdx][:]
                newValues =val.append(valueForBudget(i,b - budgets[bi]))
                #print newValues                
                #print valueForBudget(i,b - budgets[bi])
                items = h[i-1][bi][itIdx][:]
                newItems = items.append(i)
                selBudgets = h[i-1][bi][bIdx][:]
                newSelBudgets = selBudgets.append(b - budgets[bi])
                h[i][j]=[newValues,newItems,newSelBudgets]
                #print newValues
                maxVal = np.sum(newValues)
            
            
            

