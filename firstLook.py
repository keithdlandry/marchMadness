# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:52:28 2016

@author: keithlandry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.special import stdtr

from sknn.mlp import Classifier, Layer



path = "/Users/keithlandry/Desktop/marchMadness/march-machine-learning-mania-2016-v2/"
seasonFile = "Seasons.csv"
seasonDF = pd.read_csv(path+seasonFile)

teamsFile = "Teams.csv"
teamsDF = pd.read_csv(path+teamsFile)

regSeasComResFile = "RegularSeasonCompactResults.csv"
regSeasComResDF = pd.read_csv(path+regSeasComResFile)

regSeasDetResFile = "RegularSeasonDetailedResults.csv"
regSeasDetResDF = pd.read_csv(path+regSeasDetResFile) 

tournComResFile = "TourneyCompactResults.csv"
tournComResDF = pd.read_csv(path+tournComResFile)

tournDetResFile = "TourneyDetailedResults.csv"
tournDetResDF = pd.read_csv(path+tournDetResFile)

tournSeedsFile = "TourneySeeds.csv"
tournSeedsDF = pd.read_csv(path+tournSeedsFile)

tournSlotsFile = "TourneySlots.csv"
tournSlotsDF = pd.read_csv(path+tournSlotsFile)

sampleFile = "SampleSubmission.csv"
sampleDF = pd.read_csv(path+sampleFile)

regSeasComResDF[regSeasComResDF["Season"] == 1985]

year = 2000

len(regSeasComResDF["Wteam"].unique())


regSeasComResDF[~regSeasComResDF["Wloc"].isin(["N"])]

alpha = pd.DataFrame(regSeasComResDF[ (regSeasComResDF["Season"] == year) & (~regSeasComResDF["Wloc"].isin(["N"]))]["Wteam"].value_counts())
beta  = pd.DataFrame(regSeasComResDF[ (regSeasComResDF["Season"] == year) & (~regSeasComResDF["Wloc"].isin(["N"]))]["Lteam"].value_counts())

aDF = pd.DataFrame(alpha)
bDF = pd.DataFrame(beta)

newDF = pd.concat([aDF,bDF], axis = 1)

newDF["nGames"] = newDF["Wteam"] + newDF["Lteam"]

#Teams play a different number of games. 
#This could complicate things.



plt.figure()
regSeasComResDF.Wscore.plot()

regSeasComResDF.Lscore.plot()

regSeasComResDF["Wloc"].value_counts() 


homeWins = regSeasComResDF[regSeasComResDF["Wloc"] == "H"]
homeLoss = regSeasComResDF[regSeasComResDF["Wloc"] == "A"]
homeScoreWin = homeWins["Wscore"]
homeScoreLoss = homeLoss["Lscore"]
homeScore = homeScoreWin.append(homeScoreLoss)
homeScore.sum()/len(homeScore)

awayWins = regSeasComResDF[regSeasComResDF["Wloc"] == "A"]
awayLoss = regSeasComResDF[regSeasComResDF["Wloc"] == "H"]
awayScoreWin = awayWins["Wscore"]
awayScoreLoss = awayLoss["Lscore"]
awayScore = awayScoreWin.append(awayScoreLoss)
awayScore.sum()/len(awayScore)


neutGames = regSeasComResDF[regSeasComResDF["Wloc"] == "N"]
neutScoreWin = neutGames["Wscore"]
neutScoreLoss = neutGames["Lscore"]
neutScore = neutScoreWin.append(neutScoreLoss)
neutScore.sum()/len(neutScore)


np.sqrt(homeScore.var())
np.sqrt(awayScore.var())
np.sqrt(neutScore.var())


t, p = ttest_ind(homeScore, awayScore, equal_var=False)
print t,p


#a = np.random.normal(5,2,100)
#b = np.random.normal(5.2,2.5,100)
#t, p = ttest_ind(a, b, equal_var=False)
#print t, p




homeWins.Wscore.hist()


