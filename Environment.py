#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from Campaign import *

class Environment:

    def __init__(self, campaigns=[], deadline=100):
        self.campaigns = campaigns
        self.deadline = deadline
        self.t

    def generateObservationsforCampaigns(self,bid,budget):
        for i in range(0,len(self.campaigns)):
            self.campaigns[i].generateObservations(bid[i],budget[i])
            
    def step(self,bid,budget):
        if(self.t <deadline):        
            self.generateObservationsforCampaigns(bid,budget)
            self.t+=1
        
