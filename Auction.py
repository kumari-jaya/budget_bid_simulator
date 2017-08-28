import numpy as np


class Auction:

    def __init__(self, nbidders, nslots, mu, sigma, lambdas, myClickProb):
        self.nbidders = nbidders
        self.nslots = nslots
        self.mu = mu
        self.sigma = sigma
        self.lambdas = lambdas
        self.pClick = myClickProb
        if(nbidders < nslots):
            print "nBidders should be >= than nslots"

# pclick da 10-3 10-1
# fare sampling da una beta per la prob di click anche dei competitor

    def simulateAuction(self, mybid):
        bids = np.random.randn(self.nbidders) * self.sigma + self.mu
        pClicks = np.random.beta(1, 1, self.nbidders)

        expValue = np.maximum(bids * pClicks, 0)
        expValue[::-1].sort() #decreasing order
        indexes = np.argwhere(mybid * self.pClick > expValue).reshape(-1)
        if len(indexes) > 0:
            cpc = bids[indexes[0]]
            mypos = indexes[0] # slot numerati da 0 a n-1
        else:
            cpc = 0
            mypos = len(bids)  # last position

        if mypos < self.nslots:
            pObsSlot = self.lambdas[mypos]
        else:
            pObsSlot = 0

        return [cpc, mypos, pObsSlot]

    def simulateMultipleAuctions(self, nauctions, mybid):
        cpc = np.zeros(nauctions)
        mypos = np.zeros(nauctions)
        pObsSlot = np.zeros(nauctions)
        for i in range(0, int(nauctions)):
            [cpc[i], mypos[i], pObsSlot[i]] = self.simulateAuction(mybid)

        return [cpc, mypos, pObsSlot]
