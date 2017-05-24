#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
from Campaign import *
from Environment import *
from Auction import *
from Agent import *

#clickParams=np.array([1000.0,0.2,30.0, 0.1])
convparams=np.array([0.4,100,200])
# ho messo prob di conversione a 0.4 a caso,mentre 100 e 200 sono i due estremi della uniforme per generare le revenues
lambdas = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
a1= Auction(4,5,0.5,0.1, lambdas)
a2= Auction(5,5, 0.8, 0.2, lambdas)
c1 = Campaign(a1,1000.0,0.5,convparams)
c2 = Campaign(a2,,convparams)
#bid = np.array([40.0,35.0])
#budget = np.array([14300.0,13600.0])
bid = 0.5
budget = 150
c1.generateObservations(bid, budget)
#campaigns = np.array([c1,c2])
#env = Environment(campaigns,100)
#env.generateObservationsforCampaigns(bid,budget)
#c.generateObservations(40.0,14300.0)
print "Numero clicks: %f " % c1.clicks
print "Costi: %f " % c1.costs
print "Ora esaurimento: %f " % c1.hours
print "Numero conversioni: %f " % c1.conversions
print "Totale revenues: %f e" % c1.revenues
