#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
from Campaign import *
from Environment import *
from Auction import *

#clickParams=np.array([1000.0,0.2,30.0, 0.1])
convparams=np.array([0.4,100,200])
# ho messo prob di conversione a 0.4 a caso,mentre 100 e 200 sono i due estremi della uniforme per generare le revenues
lambdas = np.array([0.8, 0.6, 0.5, 0.4, 0.35])
a1= Auction(10,5,0.5,0.1, lambdas)
c1 = Campaign(a1,10000.0,0.5,convparams)
#c2 = Campaign(9000.0,clickParams,convparams)
#bid = np.array([40.0,35.0])
#budget = np.array([14300.0,13600.0])
bid = 0.4
budget = 50
c1.generateObservations(bid, budget)
#campaigns = np.array([c1,c2])
#env = Environment(campaigns,100)
#env.generateObservationsforCampaigns(bid,budget)
#c.generateObservations(40.0,14300.0)
print "Numero clicks: %f e %f" % c1.clicks
print "Costi: %f e %f" % c1.costs
print "Ora esaurimento: %f e %f" % c1.hours
print "Numero conversioni: %f e %f" % c1.conversions
print "Totale revenues: %f e %f" % c1.revenues
