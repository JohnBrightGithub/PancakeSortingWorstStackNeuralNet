from Common.DataLoader import getTuplesIteration
from Common.DataLoader import getResultsPath
from Common.WriteToResults import writeToTraining
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import numpy as np
import os
#TODO Make this configurable or maybe just have it iterate over all files in results?
def makeTrainingData(
                nums = [18, 19, 20, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27,27],
                dists = [20, 22, 22, 23, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30],
                totalSampleds = [120, 120, 120, 120, 120, 120, 60, 60, 60, 60, 12, 12, 12, 12, 12, 60]):
    numSampled = len(nums)
    for i in range(numSampled):
        num = nums[i]
        dist = dists[i]
        for secondDist in range(dist-2, dist+3):
            totalSampled = totalSampleds[i]
            nameOfFile = getResultsPath(num,dist, secondDist)
            if not os.path.isfile(nameOfFile):
                continue
            temp = pd.read_csv(nameOfFile, sep=" ")
            sampleSize = min(totalSampled, temp.shape[0])
            dat = temp.sample(n=sampleSize)
            dat = np.array(dat)
            rows = len(dat)
            datagen = [] #List of data packages from getTuplesIteration
            numProcessors = 12 #Number of processors on current machine TODO: use function to get # of processors
            r = []
            pbar = tqdm(total=len(dat))
            for rows in dat:
                datagen.append((rows, num, dist))
                # r.append(getTuplesIteration((rows, num, dist)))
                # pbar.update(1)
            with Pool(numProcessors) as pool:
                r = list(tqdm(pool.imap_unordered(getTuplesIteration, datagen), total=totalSampled, miniters=1))

            #getTuplesIteration's output is list of "packages" there are 10 packages per stack, each package has 2 arrays (orig and transformed state)
            #and "ans" which is the label used for training (0-4) which gives the difference in distance between orig and transformed state
            hashMap = {}
            index=0
            writeToTraining(r, dist)
            
    return True

makeTrainingData()