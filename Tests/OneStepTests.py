from Common.DataLoader import getResultsDataRandom
from Common.DataLoader import getTuplesIteration
from Common.DataLoader import getPrev
from Common.DataLoader import loadAllNeededFiles
from Common.DataLoader import getNumRowsFile
from Common.DataLoader import getNandDistFromFilename
from Common.DataLoader import loadGetEstimatesNextStates

from ASearch.Ctype import getDist
import numpy as np


def testGetResultsDataRandom():
    print(getResultsDataRandom(19, 22, 20))

def testGetTuplesIteration():
    testState = np.array(getPrev([13 ,15, 17, 11, 2, 6, 9, 3 ,10, 5, 8, 1, 7, 4, 19, 16, 12, 14, 18]))
    num = len(testState)
    dist = getDist(testState)
    loadAllNeededFiles(num, dist)
    getTuplesIteration([testState, num, dist])

def testGetPrev():
    testState = [1, 3, 7, 25, 5, 2, 6, 4, 8, 12, 10, 14, 9, 13, 11, 15, 23, 21, 17, 19, 22, 20, 18, 16, 24]
    num = 25
    dist = 28
    print(getPrev(testState))

def testGetNumRowsFile():
    fileName = "Results\\28-32\\Out-27-30.txt"
    print(getNumRowsFile(fileName, 28))

def testGetNandDistFromFilename():
    fileName = "Results\\28-32\\Out-27-30.txt"
    print(getNandDistFromFilename(fileName))

def testLoadGetEstimatesNextStates():
    fileName = "Results\\21-23\\Out-20-22.txt"
    loadGetEstimatesNextStates(fileName, True, 99, 100)

testLoadGetEstimatesNextStates()