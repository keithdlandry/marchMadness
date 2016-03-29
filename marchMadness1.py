# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 18:14:02 2016

@author: keithlandry
"""
from __future__ import division

import numpy as np
import pandas as pd
from random import randint
from sknn.mlp import Classifier, Layer

from colabFiltering import colabFilCostFunc
from colabFiltering import colabFilGrad
from colabFiltering import trainColabFilter
from colabFiltering import thinMatrix
from colabFiltering import varOfListOfMat
from colabFiltering import mean

from footballColabFilterLearning import findBestLambdaNfeats


def getAvgHomeAndAwayScores(df):
    homeWins = df[df["Wloc"] == "H"]
    homeLoss = df[df["Wloc"] == "A"]
    homeScoreWin = homeWins["Wscore"]
    homeScoreLoss = homeLoss["Lscore"]
    homeScore = homeScoreWin.append(homeScoreLoss)
    avgHscore = homeScore.sum()/len(homeScore)
    
    awayWins = df[df["Wloc"] == "A"]
    awayLoss = df[df["Wloc"] == "H"]
    awayScoreWin = awayWins["Wscore"]
    awayScoreLoss = awayLoss["Lscore"]
    awayScore = awayScoreWin.append(awayScoreLoss)
    avgAscore = awayScore.sum()/len(awayScore) 
    return avgHscore, avgAscore
    
def normalizeScoresByLoc(df):
    normDf = df.copy()
    avgHscore, avgAscore = getAvgHomeAndAwayScores(df)
    diff = avgHscore - avgAscore
    
    normDf.ix[df["Wloc"] == "H", "Lscore"] = df.ix[df["Wloc"] == "H", "Lscore"] + diff
    normDf.ix[df["Wloc"] == "A", "Wscore"] = df.ix[df["Wloc"] == "A", "Wscore"] + diff
    return normDf
    
def getTeamIds(df):
    winTeamIds = df["Wteam"].unique()
    lossTeamIds = df["Lteam"].unique()
    allTeamIds = np.concatenate((winTeamIds,lossTeamIds), axis = 0)
    allTeamIds = np.unique(allTeamIds)
    return allTeamIds# - 1101 #to start at 0
    
def makeIndivTeamDf(df, teamId):
    df = df[(df["Wteam"] == teamId) | (df["Lteam"] == teamId)]
    df["deltaS"] = df["Wscore"] - df["Lscore"]
    df.loc[df["Wteam"] != teamId, "oppId"] = df["Wteam"]
    df.loc[df["Lteam"] != teamId, "oppId"] = df["Lteam"]    
    df.loc[df["Lteam"] == teamId, "deltaS"] = -1*df["deltaS"]  #seems like .loc is = to .ix?
    #print df[["oppId","deltaS"]]   
    return df[["oppId","deltaS"]]
       
def replaceOppIdWithTeamStrength(df, teamStrengths):
    opponents  = pd.unique(df["oppId"])
    for opp in opponents:
        df.loc[df["oppId"] == opp, "oppStrength"] = teamStrengths[opp] 
    return df[["oppStrength","deltaS"]]
     
     
def addSeed(array):
    print "not finished yet"     
     
def findTeamStrengths(mat):
    R = (mat!=0)
    x,y = np.shape(mat)
    print "finng best parameters"
    bestL, bestNum_feat, bestMiss, YTrain = findBestLambdaNfeats(mat,R,.15)
    print bestL, bestNum_feat
    #bestL = 3
    #bestNum_feat = 5 
    print "training"
    learnedT, learnedX = trainColabFilter(mat,R,bestL,bestNum_feat,x,y)
    prediction = np.dot(learnedX,learnedT.T)
    
    teamStrengths = {}
    for teamId in range(x):
        scores1 = np.array(prediction[teamId,:]).flatten()
        scores2 = np.array(prediction[:,teamId]).flatten()
        allScores = np.append(scores1,scores2)        
        
        s = sum(allScores)/(len(allScores)-2)
        teamStrengths[teamId] = s   
    return teamStrengths, prediction
    
def populateColFilMat(Id, df, sizeA, sizeB):
    matrix = np.matrix(np.zeros(shape = (sizeA, sizeB)))          
    opponents  = pd.unique(df["oppId"])
    for opp in opponents:
        scores = df.loc[df["oppId"] == opp, "deltaS"]
        matrix[Id, opp] = np.mean(scores)
    return matrix
    
def randConcat(a, b):
    rand = randint(0, 1)
    if rand == 0:
        concat = np.append(a,b)
    if rand == 1:
        concat = np.append(b,a)
    return concat, rand
    
def thinMatrix(Y, R, fracToThin):
   

    num_obj = np.size(R,0)
    num_people = np.size(R,1)
    
    #print num_obj
   
    r = np.array(R).flatten()

    #find the spots in r where we actually have data
    indxOfData = np.array(np.where(r==True))
    indxOfData.shape = (np.size(indxOfData,1)) #not sure why this is needed

    indToRemove = np.random.rand(np.size(indxOfData)) < fracToThin

    #set aside indexes to cross validation data if we want to use it
    indForCV = indxOfData[indToRemove]
    y = np.array(Y).flatten()
    cvAns = y[indForCV]

    r[indxOfData[indToRemove]] = 0
    
    R_thin = r.reshape(num_obj,num_people)
    Y_thin = np.multiply(Y,R_thin)
    
    return Y_thin, R_thin, cvAns, indForCV

    
#def findBestLambdaNfeats(Y,R,cvPer):
#
#    #print "thining matrix"    
#    
#    YTrain, RTrain, cvAns, indForCV = thinMatrix(Y,R,cvPer)
#    
#    num_teams = np.size(R,0)
#    num_players = np.size(R,1)  
#    
#    bestL = 0
#    bestNum_feat = 0
#    bestMiss = 100000
#    
#    
#    for lmbd in range(1,20):
#        for num_feat in range(5,30):
#            
#            print lmbd
#            print num_feat
#    
#            learnedT, learnedX = trainColabFilter(YTrain,RTrain,lmbd,num_feat,num_teams,num_players)
#        
#            pred = np.dot(learnedX,learnedT.T)
#        
#            predArray = np.array(pred).flatten()
#            avgMiss = 1/len(cvAns) * sum(np.abs(predArray[indForCV] - cvAns))
#            print avgMiss
#    
#            if bestMiss > avgMiss:
#                bestMiss = avgMiss
#                bestL = lmbd
#                bestNum_feat = num_feat
#                
#    return bestL, bestNum_feat, bestMiss, YTrain

########################################################
########################################################
########################################################
path = "/Users/keithlandry/Desktop/marchMadness/march-machine-learning-mania-2016-v2/"
regSeasComResFile = "RegularSeasonCompactResults.csv"
regSeasComResDf = pd.read_csv(path+regSeasComResFile)
tournComResFile = "TourneyCompactResults.csv"
tournComResDf = pd.read_csv(path+tournComResFile)
tournSeedsFile = "TourneySeeds.csv"
tournSeedsDf = pd.read_csv(path+tournSeedsFile)
tournSlotsFile = "TourneySlots.csv"
tournSlotsDf = pd.read_csv(path+tournSlotsFile)
teamsFile = "Teams.csv"
teamsDf = pd.read_csv(path+teamsFile)

#%reset_selective regSeasComResFile
#%reset_selective tournComResFile
#%reset_selective tournSeedsFile
#%reset_selective tournSlotsFile
#%reset_selective teamsFile

X_train = []
Y_train = []

normRegSeasComResDf = normalizeScoresByLoc(regSeasComResDf)

#have team ids start at 0 instead of 1101
normRegSeasComResDf["Wteam"] = normRegSeasComResDf["Wteam"] - 1101
normRegSeasComResDf["Lteam"] = normRegSeasComResDf["Lteam"] - 1101
teamsDf["Team_Id"] = teamsDf["Team_Id"] - 1101
tournComResDf["Wteam"] = tournComResDf["Wteam"] - 1101
tournComResDf["Lteam"] = tournComResDf["Lteam"] - 1101

nTeams = len(teamsDf)

for year in regSeasComResDf["Season"].unique():
    print "working on year ", year
    yearDf = normRegSeasComResDf[normRegSeasComResDf["Season"] == year]
    yearDf = yearDf.sort_values(by=["Daynum"], ascending=[True])
    tournyYearDf = tournComResDf[tournComResDf["Season"] == year]

    teamDataFrames = {}
    teamIds = getTeamIds(yearDf) 
    
    colFilMat = np.matrix(np.zeros(shape = (nTeams, nTeams)))          
    
    for teamId in teamIds:
        teamDf = makeIndivTeamDf(yearDf,teamId)
        teamDf = teamDf.astype(int).convert_objects()

        tempMat = populateColFilMat(teamId, teamDf, nTeams, nTeams)
        colFilMat = colFilMat + tempMat
                
        teamDataFrames[teamId] = teamDf

    teamStrengths, prediction = findTeamStrengths(colFilMat)
    
    for teamId in teamIds:
        teamDf = teamDataFrames[teamId]
        teamDataFrames[teamId] = replaceOppIdWithTeamStrength(teamDf,teamStrengths)
        
    for Wteam, Lteam in zip(tournyYearDf["Wteam"], tournyYearDf["Lteam"]):
        team0 = np.array(teamDataFrames[Wteam]).flatten()
        team1 = np.array(teamDataFrames[Lteam]).flatten()
        xrow, winnerPos = randConcat(team0,team1)
        X_train.append(xrow[-100:]) #take last 25 games since some teams play more games than others 
        Y_train.append(winnerPos)   #0 indicates first half of row is winner 1 indicates second half of row  winner
        
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        
        
#-----End of building X_train and Y_train




