from BruteForce.MakeChunks import makeChunks
from BruteForce.MakeChunks import makeMixedStack
from ASearch.Ctype import getDistRange
import numpy as np
import os
noAdj5 = [
     [1, 3, 5, 2, 4], 
     [1, 4, 2, 5, 3]
]
def printToFile(str, outFileName):
    fileName = outFileName
    with open(fileName, "a") as myfile:
        myfile.write(str + "\n")
        myfile.close()

def getNewState(state):
    print(state)
    stateT = state.replace('[', '')
    stateT = stateT.replace(']', '')
    stateT = [x.strip() for x in stateT.split(',')]
    print(stateT)
    n = len(stateT)
    newState = []
    for i in range(1,n):
        newState.append(int(stateT[i]))
    return newState
def testChunks(state, max):
        for j in range(max):
            testState = makeChunks(state, j+1)
            dist = getDistRange(testState, 7*(j+1), 7*(j+1)+4)
            printToFile(str(testState)+ " " + str(dist), "BruteForce\\testChunks.txt")
def testChunksMixed(state, max):
        for j in range(max):
            for mixState in noAdj5:
                testState = makeMixedStack(state, mixState, j+1)
                dist = getDistRange(testState, 7*(j+1)-2, 7*(j+1)+2)
                printToFile(str(testState)+ " " + str(dist), "BruteForce\\testChunksMixed.txt")
