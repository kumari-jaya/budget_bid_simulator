#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
from Campaign import *
from Environment import *
from Auction import *
from Agent import *
from Core import *


convparams=np.array([0.4,100,200])
# ho messo prob di conversione a 0.4 a caso,mentre 100 e 200 sono i due estremi della uniforme per generare le revenues
# 1 0.71 0.56 0.53 0.49 0.47 0.44 0.44 0.43 0.43
lambdas = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
deadline=10
a1= Auction(4,5,0.5,0.1, lambdas)
a2= Auction(5,5, 0.6, 0.2, lambdas)
c1 = Campaign(a1,1000.0,0.5,convparams)
c2 = Campaign(a2,1500.0, 0.5,convparams)

nBids=10
nIntervals=10
agent = Agent(1000, deadline, 2,nIntervals,nBids,1500)
agent.initGPs()
env = Environment([c1,c2])
core = Core(agent, env, deadline)


#core.step()
core.runEpisode()
