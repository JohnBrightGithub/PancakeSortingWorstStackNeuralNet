# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 18:04:17 2023

@author: johnp
"""
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import argparse

from WorstStackNN.worstStack2D import getInputPrevState
from WorstStackNN.TrainWorstStack import TrainWorstStack

from Common.DataLoader import getResultsDataRandom
from Common.DataLoader import getPrev
from Common.DataLoader import loadAllNeededFiles
from Common.DataLoader import isResultsFile


maxColumns = 35
iterations = 1000
testN = 19
testDist = 22
trainFirst = False

class CommandLineTest:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-mc", "--MaxColumns", help = " max columns for neural network data (default = 35)", required = False, default = 35)
        parser.add_argument("-tf", "--TrainFirst", help = " train a new model before testing (default = False)", required = False, default = False)
        parser.add_argument("-it", "--Iterations", help = " number of iterations for the test (default = 1000)", required = False, default = 1000)
        parser.add_argument("-n", "--TestN", help = " n value for test (default = 19)", required = False, default = 19)
        parser.add_argument("-d", "--TestDist", help = " dist value for test (default = 22)", required = False, default = 22)
        
        argument = parser.parse_args()
        
        if argument.MaxColumns:
            global maxColumns
            maxColumns = int(argument.MaxColumns)
        if argument.TrainFirst:
            global trainFirst
            trainFirst = bool(argument.TrainFirst)
        if argument.Iterations:
            global iterations
            iterations = int(argument.Iterations)
        if argument.TestN:
            global testN
            testN = int(argument.TestN)
        if argument.TestDist:
            global testDist
            testDist = int(argument.TestDist)
            print("testDist set ", testDist)
        

if __name__ == '__main__':       
    app = CommandLineTest()
    if(trainFirst):
       TrainWorstStack(maxColumns)
    model = tf.keras.models.load_model("Models\\worstPancakeStackV2.keras")
    hits=0
    randomHits = 0
    print("loading needed files: ")
    loadAllNeededFiles(testN-1, testDist)
    print("testing on stacks: ")
    for i in tqdm(range (iterations)):
        testDist2  = testDist - (random.randint(0, 4) -2)
        timeout=0
        while not isResultsFile(testN, testDist, testDist2) and timeout<=100:
            timeout+=1
            testDist2  = testDist - (random.randint(0, 4) -2)
            
        if(timeout>100):
            print("Needed files do not exist for n = ",(testN-1))
            break
        arr1, dist = getResultsDataRandom(testN, testDist, testDist2)
        arr2 = getPrev(arr1)
        retArr = getInputPrevState(arr1, arr2)
        nextMoveArr = np.array(model(np.expand_dims(retArr,0)))
        nextMove = nextMoveArr.argmax()
        print("dist: ", dist, " nextMove ", nextMove)
        dist = (testDist-testDist2)+2
        nextMoveRandom = random.randint(0, 4)
        if nextMove==dist:
            hits+=1
        if nextMoveRandom==dist:
            randomHits+=1
    print("hits: ", hits, " percent: ", hits/iterations)
    print("randomHits: ", randomHits, " randomPercent ", randomHits/iterations)

        