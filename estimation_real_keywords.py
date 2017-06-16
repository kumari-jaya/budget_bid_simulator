import numpy as np
from Agent import *
from Plotter import *
from matplotlib import pyplot as plt
import csv


def readData( path):
    idKeyword = []
    data = []
    bid = []
    idCampagna = []
    costs = []
    clicks=[]
    conversionsValue = []
    budgets = []
    ora = []
    with open(path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            results = np.array(row)
            data= np.append(data,results[0])
            idKeyword=np.append(idKeyword,int(results[1]))
            bid=np.append(bid,float(results[2]))
            idCampagna = np.append(idCampagna,float(results[4]))
            clicks = np.append(clicks,float(results[6]))
            budgets = np.append(budgets,float(results[11]))
            ora = np.append(ora,results[12])
    return [idKeyword, data,bid, idCampagna, clicks, budgets, ora]


[idKeyword, data,bid, idCampagna, clicks, budgets, ora]= readData('data/keywords.csv')






keywords = np.unique(idKeyword)


for i in range(0,len(keywords)):
    idx = np.argwhere(idKeyword==keywords[i]).reshape(-1)
    bid_k= bid[idx]
    bud_k = budgets[idx]
    clicks_k = clicks[idx]
    ora_k = ora[idx]
    agent= Agent(budgetTot=1000, deadline= 100, ncampaigns=1, nIntervals=10, nBids=10,maxBudget=1000.0)
    agent.prevBids = np.atleast_2d(bid_k)
    agent.prevBudgets = np.atleast_2d(bud_k)
    agent.prevClicks = np.atleast_2d(clicks_k)
    agent.initGPs()
    plotter = Plotter(agent=agent)
    #plt.figure(i)
    #plt.plot(bid_k)
    mean = np.mean(bid_k)
    a =np.logical_not(np.isclose(bid_k,mean))
    div = np.argwhere(np.logical_not(np.isclose(bid_k,mean))==True).reshape(-1)
    if(len(div)>=1):
        print i
        print a
        plt.figure(i)
        plt.plot(bid_k)
        plotter.plotGP(0,fixedBid=True,bid=bid_k[0])


