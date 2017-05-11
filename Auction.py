import numpy as np

class Auction:

    def __init__(self, nbidders, nslots, mu, sigma, lambdas):
        self.nbidders= nbidders
        self.nslots= nslots
        self.mu = mu
        self.sigma = sigma
        self.lambdas = lambdas

    def simulateAuction(self, mybid):
        bids = np.random.randn(self.nbidders)*self.sigma + self.mu
        bids[::-1].sort() #decreasing order
        indexes = np.argwhere(mybid > bids).reshape(-1)
        cpc = bids[indexes[0]]
        mypos = indexes[0] # slot numerati da 0 a n-1
        pobs = self.lambdas[mypos]
        return [cpc,mypos,pobs]

    def simulateMultipleAuctions(self, nauctions, mybid):
        cpc= np.array([])
        mypos= np.array([])
        pobs= np.array([])
        for i in range(0,int(nauctions)):
            [cpctemp,mypostemp,pobstemp] = self.simulateAuction(mybid)
            cpc = np.append(cpc, cpctemp)
            mypos = np.append(mypos, mypostemp)
            pobs = np.append(pobs, pobstemp)
        return [cpc,mypos,pobs]
