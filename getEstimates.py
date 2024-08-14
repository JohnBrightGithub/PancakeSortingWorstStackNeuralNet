# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 17:36:37 2023

@author: johnp
"""
import multiprocessing
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import os

from ASearch.Ctype import getDistRange
from WorstStackNN.worstStack2D import getInputPrevState
from Common.cleanDuplicates import cleanDuplicatesFolder
from Common.cleanDuplicates import setPrintDiff




import sys
import argparse

from OneStepData.OneStepDataLoader import loadGetEstimatesNextStates

model = tf.keras.models.load_model("Models\\worstPancakeStackV2.keras")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

nPlus = 4
origDist = 23
n = 21
Partial = False
Parts = 1
xMoreThanN = -2
maxCores = multiprocessing.cpu_count()
numCores = maxCores
PrintDiff = False

class CommandLineEstimate:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--nPlus", help = " number 0-4 representing the minimum threshold (-2 to +2 original distance) for which stacks to calculate distance for ", required = False, default = 4)
        parser.add_argument("-xM", "--xMoreThanN", help = " minimum threshold (-2 to +2 original distance) for which stacks to write to Results folder ", required = False, default = -2)
        parser.add_argument("-n", "--n", help = " n value of stacks to test", required = False, default = 21)
        parser.add_argument("-dst", "--origDist", help = " dist value of stacks to test ", required = False, default = 23)
        parser.add_argument("-parts", "--Parts", help = " for large files, reads file in #parts parts", required = False, default = 1)
        parser.add_argument("-cores", "--numCores", help = " number of cores to use during", required = False, default = maxCores)
        parser.add_argument("-pd", "--PrintDiff", help = " set to True to print arrays that are removed when removing duplicates ", required = False, default = False)
        
        argument = parser.parse_args()
        
        if argument.nPlus:
            global nPlus
            nPlus = int(argument.nPlus)
        if argument.xMoreThanN:
            global xMoreThanN
            xMoreThanN = bool(argument.xMoreThanN)
        if argument.n:
            global n
            n = int(argument.n)
        if argument.origDist:
            global origDist
            origDist = int(argument.origDist)
        if argument.Parts:
            global Parts
            Parts = int(argument.Parts)
            global Partial
            Partial = True
        if argument.numCores:
            global numCores
            numCores = int(argument.numCores)
            if(numCores>maxCores):
                print("requested number of cores exceeds number of cores on machine ")
                raise argparse.ArgumentTypeError("requested number of cores %s exceeds number of cores on machine " % numCores)
        if argument.PrintDiff:
            global PrintDiff
            PrintDiff = bool(argument.PrintDiff)
        

def getTheEstimatesNext(element):
    nPlus = element[2][0]
    n=len(element[1])
    nextMove=0
    orig = element[0]
    retArr = getInputPrevState(element[1], orig)
    nextMoveArr = np.array(model(np.expand_dims(retArr,0)))
    nextMove = nextMoveArr.argmax()
    print("nextMove ", nextMove, " nPlus ", nPlus)
    if(nextMove>=nPlus):
        return element[1]
    return None
def processHardStacks(hardStack):
    global origDist
    state = [i for i in hardStack if i != 0]
    
    state = list(state)
    n=len(state)
    for i in range(n):
        state[i] = int(state[i])
    dist = getDistRange(state, origDist-2, origDist+2)
    if(dist>=n+xMoreThanN):
        state.append(dist)
        return state
        
def appendToFileSpec(permStr, n, dist, origDist):
    path = "Results/"+str(n)+ "-"+ str(dist)
    if(not os.path.exists(path)):
        os.makedirs(path)
    fileName = path + "/Out-" + str(n-1) + "-" + str(origDist) + ".txt"
    with open(fileName, "a") as myfile:
        myfile.write(permStr + "\n")
        myfile.close()
def printerFileSpec(arr, n, dist, origDist):
    printStr = ""
    for ele in arr:
        ele = int(ele)
        printStr = printStr + str(ele) + " "
    printStr = printStr.strip()
    appendToFileSpec(printStr, n, dist, origDist)    
def writeNumToFile(num):
    with open("Common/lastNum.txt", "a") as myfile:
        myfile.write(str(num) + "\n")
        myfile.close()
def appendToFile(permStr):
    with open("Out.txt", "a") as myfile:
        myfile.write(permStr + "\n")
        myfile.close()
def printerFile(arr):
    printStr = ""
    for ele in arr:
        printStr = printStr + str(ele) + " "
    printStr = printStr.strip()
    appendToFile(printStr)            
def printer(arr, length, additional):
    for element in arr:
        printStr = ""
        for ele in element:
            printStr = printStr + str(ele) + " "
        printStr = printStr.strip()
        print(printStr)

if __name__ == "__main__":
    app = CommandLineEstimate()
    directory_in_str = "Results\\"+str(n)+ "-"+ str(origDist)
    directory = os.fsencode(directory_in_str)
    print("NUmber of Files: ", len(os.listdir(directory)))
    print("Partial ", Partial)
    start = 0
    for file in os.listdir(directory):
        try:
            filename = directory_in_str+"\\"+os.fsdecode(file)
            Estimate = True
            print("FILENAME: ", filename)
            for x in range(start, Parts):
                writeNumToFile(x)
                hardStacks = []
                oneStepData = loadGetEstimatesNextStates(filename, Partial, x, Parts, nPlus)
                rows = len(oneStepData)
                print("X: ", x)
                if(Estimate):
                    print("Getting Estimates ")
                    with Pool(12) as pool: 
                        r = list(tqdm(pool.imap_unordered(getTheEstimatesNext, oneStepData), total=rows, miniters=100))
                    print("arrs: ", r)
                    for arrs in r:
                        if(not arrs is None):
                            hardStacks.append(arrs)
                else:
                    oneLen = len(oneStepData)
                    for i in tqdm(range(oneLen)):
                        hardStacks.append(oneStepData[i][1])
                print("Processing Hard Stacks ")
                with Pool(12) as pool:
                    r = list(tqdm(pool.imap_unordered(processHardStacks, hardStacks), total=len(hardStacks), miniters=100))
                
                rLen = len(r)
                nMinus2Arr = []
                nMinus1Arr = []
                nMinus0Arr = []
                nPLus1Arr = []
                nPLus2Arr = []
                printArrs = []
                printArrs.append(nMinus2Arr)
                printArrs.append(nMinus1Arr)
                printArrs.append(nMinus0Arr)
                printArrs.append(nPLus1Arr)
                printArrs.append(nPLus2Arr)
                print("appending arrays ")
                for i in tqdm(range(rLen)):
                    arr = r[i]
                    if(not arr is None):
                        n = len(arr)-1
                        dist = arr[-1]
                        del arr[-1]
                        printArrs[dist-origDist+2].append(arr)
                pbar = tqdm(total=rLen)
                print("writing to files ")
                for i in range(5):
                    dist = origDist-2+i
                    if(len(printArrs[i])!=0):
                        path = "Results/"+str(n)+ "-"+ str(dist)
                        if(not os.path.exists(path)):
                            os.makedirs(path)
                        fileName = path + "/Out-" + str(n-1) + "-" + str(origDist) + ".txt"
                        with open(fileName, "a") as myfile:
                            for arr in printArrs[i]:
                                printStr = ""
                                for ele in arr:
                                    ele = int(ele)
                                    printStr = printStr + str(ele) + " "
                                printStr = printStr.strip()  
                                myfile.write(printStr + "\n")
                                print(printStr)
                                pbar.update(1)
                            myfile.close()
        except Exception as e:
            print("exception: ", e) 

    #clean duplicates
    setPrintDiff(PrintDiff)
    for i in range(5):
        dist = origDist-2+i
        path = "Results/"+str(n+1)+ "-"+ str(dist)
        if(os.path.exists(path)):
            cleanDuplicatesFolder(path)

    input("Enter any key to quit.") 
     
    sys.exit()         
        