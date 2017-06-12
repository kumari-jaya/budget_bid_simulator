#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from Campaign import *

class Environment:

    def __init__(self, campaigns=[]):
        self.campaigns = campaigns
        self.t=0

    def generateObservationsforCampaigns(self,bid,budget):
        clicks = np.array([])
        conversions = np.array([])
        costs = np.array([])
        revenues = np.array([])
        hours = np.array([])

        for i in range(0,len(self.campaigns)):
            observations=self.campaigns[i].generateObservations(bid[i],budget[i])
            clicks = np.append(clicks,observations[0])
            conversions = np.append(conversions,observations[1])
            costs = np.append(costs,observations[2])
            revenues= np.append(revenues,observations[3])
            hours = np.append(hours,observations[4])

        return [clicks,conversions,costs,revenues,hours]



    def step(self,bid,budget):

        self.t+=1
        return self.generateObservationsforCampaigns(bid,budget)
