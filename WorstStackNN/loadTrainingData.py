import os
import pandas as pd
import numpy as np
from Common.commonFunc import distsToLabel

def loadTrainingData(maxColumns):
    directory_in_str = "Training"
    directory = os.fsencode(directory_in_str)
    print("NUmber of Files: ", len(os.listdir(directory)))
    dat = np.zeros((2, 0 ,maxColumns))
    labels = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filePath = os.path.join(subdir, file).decode('utf-8')
            df = np.array(pd.read_csv(filePath, sep=" "))
            print("file ", filePath)
            for row in df:
                divider = list(row).index('|')
                newDist = row[divider-1]
                newArr = row[0:divider-1]
                origDist = row[-1]
                orig = row[divider+1:-1]
                newDat = np.zeros((2, 1, maxColumns))
                n = len(newArr)
                newDat[0][0]= np.pad(newArr, (0, maxColumns-n), mode='constant', constant_values=0)             
                newDat[1][0] = np.pad(orig, (0, maxColumns-n+1), mode='constant', constant_values=0)
                label = distsToLabel(origDist, newDist)
                labels.append(label)
                dat = np.append(dat, newDat,axis=1)
    return dat, labels