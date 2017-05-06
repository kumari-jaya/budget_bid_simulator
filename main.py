#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
from Campaign import *
from Environment import *

clickParams=np.array([1000.0,0.2,30.0, 0.1])
convparams=np.array([0.4,100,200])
# ho messo prob di conversione a 0.4 a caso,mentre 100 e 200 sono i due estremi della uniforme per generare le revenues
c1 = Campaign(10000.0,clickParams,convparams)
c2 = Campaign(9000.0,clickParams,convparams)
bid = np.array([40.0,35.0])
budget = np.array([14300.0,13600.0])
campaigns = np.array([c1,c2])
env = Environment(campaigns,100)
env.generateObservationsforCampaigns(bid,budget)
#c.generateObservations(40.0,14300.0)
print "Numero clicks: %f e %f" % (c1.clicks,c2.clicks)
print "Costi: %f e %f" % (c1.costs,c2.costs)
print "Ora esaurimento: %f e %f" % (c1.hours,c2.hours)
print "Numero conversioni: %f e %f" % (c1.conversions,c2.conversions)
print "Totale revenues: %f e %f" % (c1.revenues,c2.revenues)
