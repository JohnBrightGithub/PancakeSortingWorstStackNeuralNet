import numpy as np

from WorstStackNN.worstStack2D import TrainRandomPrevState
from WorstStackNN.loadTrainingData import loadTrainingData

def TrainWorstStack(maxColumns=35):
  
    nA = []
    dfResults, naResults = loadTrainingData(maxColumns)
    lenResults = len(dfResults[0])
    dfTraining = np.zeros((lenResults, 2, maxColumns))
    for i in range(lenResults):
        dfTraining[i][0] = dfResults[0][i]
        dfTraining[i][1] = dfResults[1][i]
    nA = np.array(naResults)
    TrainRandomPrevState(dfTraining, nA, maxColumns)