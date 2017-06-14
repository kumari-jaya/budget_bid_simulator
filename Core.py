

from Agent import Agent
from Environment import Environment
import time as time
import numpy as np
import time as time

class Core:
    def __init__(self,agent,environment, deadline):
        self.agent=agent
        self.environment=environment
        self.deadline = deadline
        self.t=0



    def step(self):
        [budget,bid] = self.agent.chooseAction(sampling=True)
        observations = self.environment.step(bid,budget)

        lastClicks = observations[0]
        lastConversions = observations[1]
        lastCosts = observations[2]
        lastRevenues = observations[3]
        lastHours = observations[4]
        self.agent.updateState(bid,budget,lastClicks,lastConversions,lastCosts,lastRevenues,lastHours)

        self.t+=1




    def runEpisode(self):
        for t in range(0,self.deadline):
            print "Day : ",t
            #start = time.time()
            self.step()
            #print time.time() - start

