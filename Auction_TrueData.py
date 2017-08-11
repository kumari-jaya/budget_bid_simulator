import numpy as np
from numpy import genfromtxt
import scipy.stats as stats

class Auction_TrueData:

    def __init__(self, nbidders, nslots, lambdas):
        self.nbidders= nbidders
        self.nslots= nslots
        self.lambdas = lambdas
        if(nbidders<nslots):
            print "nBidders should be >= than nslots"

# pclick da 10-3 10-1
# fare sampling da una beta per la prob di click anche dei competitor

    def simulateAuction(self, mybid):
        [bids,pClick] =  self.SampleFromRealData()
        bids[::-1].sort() #decreasing order
        indexes = np.argwhere(mybid > bids).reshape(-1)
        cpc=0
        mypos=len(bids)
        if(len(indexes)>0):
            cpc = bids[indexes[0]]
            mypos = indexes[0] # slot numerati da 0 a n-1
        pobs = 0
        if(mypos<self.nslots):
            pobs = self.lambdas[mypos]
        return [cpc,mypos,pobs,pClick]

    def simulateMultipleAuctions(self, nauctions, mybid):
        cpc= np.array([])
        mypos= np.array([])
        pobs= np.array([])
        pClick = np.array([])
        for i in range(0,int(nauctions)):
            [cpctemp,mypostemp,pobstemp,pClicktemp] = self.simulateAuction(mybid)
            cpc = np.append(cpc, cpctemp)
            mypos = np.append(mypos, mypostemp)
            pobs = np.append(pobs, pobstemp)
            pClick = np.append(pClick,pClicktemp)
        return [cpc,mypos,pobs,pClick]

    def SampleFromRealData(self):
        ## conversione in python della funzione ssa_generator.m
        truebids = genfromtxt('BidData.csv', delimiter=',')
        values = np.zeros(self.nbidders)
        for i in range(0,self.nbidders):
            index = np.random.randint(0,100)
            minBid = truebids[index,0]
            maxBid = truebids[index,1]
            meanBid = truebids[index,2]
            varBid = truebids[index,3]
            values[i] = stats.truncnorm.rvs((minBid-meanBid)/np.sqrt(varBid),(maxBid-meanBid)/np.sqrt(varBid),
                                            loc=meanBid,scale=np.sqrt(varBid),size=1)
        index = np.random.randint(0,100)
        betaA = truebids[index,4]
        betaB = truebids[index,5]
        quality = np.random.beta(betaA,betaB)
        #normalization
        values = values / values.max()
        return values,quality
