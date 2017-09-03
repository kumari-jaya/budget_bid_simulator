import numpy as np
from numpy import genfromtxt
import scipy.stats as stats
from Auction import *

class AuctionTrueData(Auction):

    def __init__(self, nBidders, nSlots, lambdas, myClickProb, path='./data/BidData.csv',fixedIndex=-1):
        self.nBidders = nBidders
        self.nSlots = nSlots
        self.lambdas = lambdas
        self.lambdas = lambdas
        self.pClick = myClickProb

        allData = genfromtxt(path, delimiter=',')
        if fixedIndex==-1:
            index = np.random.randint(0, 100)
        else:
            index = fixedIndex
        self.auction = allData[index, :]
        if(nBidders < nSlots):
            print "nBidders should be >= than nslots"

    def simulateMultipleAuctions(self, nAuctions, myBid):
        # Generate other bids and qualities
        minBid = self.auction[0]
        maxBid = self.auction[1]
        meanBid = self.auction[2]
        varBid = self.auction[3]
        unNormBids = stats.truncnorm.rvs((minBid - meanBid) / np.sqrt(varBid), (maxBid - meanBid) / np.sqrt(varBid), loc=meanBid, scale=np.sqrt(varBid), size=(self.nBidders, nAuctions))
        bids = 0.05 + (unNormBids - unNormBids.min()) / (unNormBids.max() - unNormBids.min()) * (5 - 0.05)
        pClicks = np.random.beta(self.auction[4], self.auction[5], (self.nBidders, nAuctions))
        expValue = bids * pClicks

        # Executing the auction
        indexMatr = (expValue > (myBid * self.pClick)).astype(int)

        myPos = np.sum(indexMatr, axis=0)
        cpc = np.max(bids * (1-indexMatr), axis=0)
        pObsSlot = np.zeros(nAuctions)
        idxVisible = np.argwhere(myPos < self.nSlots).reshape(-1)
        pObsSlot[idxVisible] = self.lambdas[myPos[idxVisible]]
        return [cpc, myPos, pObsSlot]
