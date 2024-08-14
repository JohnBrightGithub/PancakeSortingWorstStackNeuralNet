#File with methods to load in values from 
from multiprocessing import Pool
import pandas as pd
import numpy as np
import random
import os
import os.path
from tqdm import tqdm
from ASearch.Ctype import getDistRange
from Common.WriteToResults import writeToResults

maxColumns=35
verbose = True
fileSizeWorthLoading = 1000000
write = False
def isFileTooLarge(n, dist, dist2):
    #if the file is too large it is probably better to just get the distance
    file = getResultsPath(n, dist, dist2)
    print("isFile Too Large: ", file)
    if(os.path.isfile(file)):
        fileSize = os.path.getsize(file)
        print("filesize: ", fileSize, " fileSizeWorthLoading ", fileSizeWorthLoading)
        if(fileSize<(fileSizeWorthLoading)):
            return False
    else:
        logIfVerbose("file not found " + file)
        return True
   
    return True

def getDistAndAddToResults(state, oldDist):
    n =  len(state)
    dist = getDistRange(state, oldDist-2, oldDist+2)
    if(not isFileTooLarge(n, dist, oldDist)):
        if(write):
            writeToResults(dist,state, oldDist)
    return dist
def getDataPackage(tempState, action, dist, n, dat):
    #perform flip
    tempState[0:action+1] = tempState[0:action+1][::-1]
    newDat=(dat, tempState)
    distState = np.array(tempState[0:(n+1)]).astype(int)  
    #the label to be trained on will be 0-4, 0 if the new dist is 2 less than the origin, 4 if it is two more
    #since this new state was made with 2 flips from the original, it must be within dist 2 of the original
    nATemp = getDistAndAddToResults(distState, dist) - dist +2
    return (newDat,nATemp)

loadedArrs = {}
def loadAllNeededFiles(n, dist):
    for testDist in range(dist-2, dist+3, 1):
        if not isFileTooLarge(n+1, dist, testDist):
            path = getResultsPath(n+1, dist, testDist)
            print("loading ", path)
            loadResultsIfNotLoaded(path)
            df = np.array(loaded[path])
            newDict = {}
            for arr in df:
                newDict[stateToString(arr)] = dist
            loadedArrs[str(path)] = newDict
            
def getTuplesIteration(datagen):
    dat = datagen[0]
    n = datagen[1]
    dist = datagen[2]
    dataPackage = []
    #make copy of state with n+1 element on top
    state =np.append(dat, (n+1))
    action = n
    #flip original state over completely, so the new n+1 element is on top
    state[0:action+1] = state[0:action+1][::-1]
    actions = []

    #bring 1 up to the front for 1 iteration, since these are usually the hardes stacks:
    tempState = state.copy()
    action = list(state).index(1) 
    dataPackage.append(getDataPackage(tempState, action, dist, n, dat))
    actions.append(action)
    #transform original state (datagen[0]) for 10 random moves
    for i in range(9):
        tempState = state.copy()
        action = random.randint(1, n-1)   
        while action in actions:
            action = random.randint(1, n-1)
        actions.append(action)   
        dataPackage.append(getDataPackage(tempState, action, dist, n, dat))
    
    
    return dataPackage

def getPrev(inputState):
    #returns the previous state that created this state
    #By fliping where the n occurs and flipping the n to the last position
    state = inputState.copy()
    action=-1
    n=len(state)
    for i in range(n):
        if(state[i]==n):
            action = i
    state[0:action+1] = state[0:action+1][::-1]
    action=n-1
    state[0:action+1] = state[0:action+1][::-1]
    return state[0:n-1]

loaded = {} #dictionary that holds file data loaded from Results folder
def getResultsFile(n, dist, dist2):
    path = getResultsPath(n, dist, dist2)
    fileExists = loadResultsIfNotLoaded(path)
    return fileExists
def logIfVerbose(logStr):
    if(verbose):
        print(logStr)
def getResultsPath(n, dist, dist2):
    return "Results/"+str(n)+ "-"+ str(dist)+"/Out-" + str(n-1) + "-" + str(dist2) + ".txt"
def isResultsFile(n, dist, dist2):
    path = getResultsPath(n, dist, dist2)
    return os.path.isfile(path)
def loadResultsIfNotLoaded(path):
    if not path in loaded:
        if(os.path.isfile(path)):
            print("reading ", path)
            loaded[path] =  pd.read_csv(path, sep=" ")
        else:
            if(verbose):
                logIfVerbose("file not found " + path)
            return False
        print("finished reading ", path)
    return True
def stateStringToState(stateStr):
    stateStr = stateStr.strip()
    arr = stateStr.split(',')
    state = []
    for el in arr:
        if(el==''):
            continue
        state.append(int(el))
    return state
def getResultsDataRandom(n, dist, dist2):
    fileExists = getResultsFile(n, dist, dist2)
    if(not fileExists):
        logIfVerbose("results File not found, cannot get random row ")
        return []
    path = getResultsPath(n, dist, dist2)
    x, dist = random.choice(list(loadedArrs[path].items()))
    x = stateStringToState(x)
    retArr = x.copy()
    return np.array(retArr), dist
def getNandDistFromFilename(filename):
    tempString = filename.split('\\')
    tempString = tempString[1]
    tempString = tempString.partition('-')
    n = int(tempString[0])
    tempString = tempString[2]
    dist = int(tempString)
    return n, dist

def getNumRowsFile(filename, n):
    #for now this only works with n>9, all stacks can be brute forced with n<9, so there's no need to change that
    #get rowsize by: 
    #9 single character digits
    #(n-9) double character digits
    #n spaces
    #1 newline
    rowSize = (n-9)*2 + 9 + n + 1
    if(os.path.isfile(filename)):
        fileSize = os.path.getsize(filename)
    return int(fileSize/rowSize)
def loadGetEstimatesNextStates(filename, Partial, x, parts, nPlus):
    dict = {}
    n, dist = getNandDistFromFilename(filename)
    numRows = getNumRowsFile(filename, n)
    rowSlice = numRows/parts
    rowsToTake = int(rowSlice)
    if(not isinstance(rowSlice, int) and x==parts-1):
        rowsToTake += numRows % parts
    if(Partial):
        df = pd.read_csv(filename, sep=" ", header=None, skiprows=x*int(rowSlice), nrows=rowsToTake)
    else:
        df = pd.read_csv(filename, sep=" ", header=None)
    df =np.array(df)
    rows = len(df)
    newDf = np.zeros((rows*(n-1), 3, n+1))
    j=0
    for i in tqdm(range(rows)):
        state = df[i]
        state = np.append(state, 0)
        tempState = state.copy()
        state[n]=n+1
        action = n
        state[0:action+1] = state[0:action+1][::-1]
        for action in range(1, n):
            newState = state.copy()
            newState[0:action+1] = newState[0:action+1][::-1]
            newStateStr = stateToString(newState)
            dict[newStateStr] = tempState
            newDf[j][0] = tempState
            newDf[j][1] = newState
            newDf[j][2][0] = nPlus
            j+=1
    return np.array(newDf)

def stateToString( state):
    retString=""
    for element in state:
        retString = retString + str(element) + ","
    return retString

def getResultsDataTraining(nS, dists, sizes):
    numToLoad = len(nS)
    dat = np.zeros((2, 0 ,maxColumns))
    labels = []
    for i in range(numToLoad):
        n = nS[i]
        dist = dists[i]
        size = sizes[i]
        directory_in_str = "Results/"+str(n)+ "-"+ str(dist)
        directory = os.fsencode(directory_in_str)
        for file in os.listdir(directory):
            fileString = os.fsdecode(file)
            filename = directory_in_str+"/"+fileString
            print("fileString ", fileString)
            tempLabel = fileString.partition('Out-' + str(n-1))
            tempLabel = tempLabel[2]
            tempLabel = tempLabel.partition('-')
            tempLabel = tempLabel[2]
            tempLabel = tempLabel.partition('.')
            tempLabel = tempLabel[0]
            tempLabel = int(tempLabel)
            tempLabel = dist-tempLabel+2
            if(tempLabel>4):
                print("Problem, label should not be greater than 4 ", fileString)
            oneStepData =   pd.read_csv(filename, header=None, sep=" ")
            rows = len(oneStepData)
            tempDat = oneStepData.sample(n=min(size, rows))
            tempDat = np.array(tempDat)
            tempLen = tempDat.shape[0]
            newDat = np.zeros((2, tempLen, maxColumns))
            for i in range(tempLen):
                prevState = getPrev(tempDat[i])
                newDat[0][i] = np.pad(tempDat[i], (0, maxColumns-n), mode='constant', constant_values=0)             
                newDat[1][i] = np.pad(prevState, (0, maxColumns-n+1), mode='constant', constant_values=0)
                print("getResultsDataTraining arr1: ", newDat[0][i] , " arr2: ", newDat[1][i])
                labels.append(tempLabel)
            dat = np.append(dat, newDat, axis=1)
            print("datShape ", dat.shape)
            
    return dat, labels

