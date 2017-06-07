# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:05:28 2017

@author: alessandro
"""

import numpy as np


nIntervals = 4
maxBudget = 100
budgets = np.linspace(0,maxBudget,nIntervals)
nItems = 4
values = np.ones(shape=(nItems,len(budgets)))
values = np.array([[0,2,4,4],[0,3,6,6],[0,2,4,8],[0,1,2,9]])
m = np.zeros(shape=(nItems,len(budgets)))
h = np.zeros(shape=(nItems,len(budgets)))
h=h.tolist()

def valueForBudget(itemIdx,budget):
    idx = np.argwhere(budget>=budgets)
    #print "Indici: ",idx
    #print "maxIndice",idx.max()
    return values[itemIdx,idx.max()]
    
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
        h[i][j] = h[i-1][j][:]
        maxVal = 0
        for bi in range(0,j+1):
            #print (np.sum(h[i-1][valIdx]) + valueForBudget(i,b - budgets[bi]))
            #print maxVal

            if ((np.sum(h[i-1][bi][valIdx]) + valueForBudget(i,b - budgets[bi])) >maxVal):
                val = h[i-1][bi][valIdx][:]
                val.append(valueForBudget(i,b - budgets[bi]))
                newValues = val[:]
                #print newValues                
                #print valueForBudget(i,b - budgets[bi])
                items = h[i-1][bi][itIdx][:]
                items.append(i)
                newItems = items[:]
                print newItems
                selBudgets = h[i-1][bi][bIdx][:]
                selBudgets.append(b - budgets[bi])
                newSelBudgets = selBudgets[:]
                h[i][j]=[newValues,newItems,newSelBudgets]
                maxVal = np.sum(newValues)