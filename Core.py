

from Agent import Agent
from Environment import Environment
import time as time
import numpy as np

class Core:
    def __init__(self,agent,environment):
        self.agent=agent
        self.environment=environment
        self.deadline = self.agent.deadline
        self.t=0
        


    def step(self):
        [bid,budget] = self.agent.chooseAction()
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
            self.step()
            
                      
