from OneStepData.WriteToResults import makeTrainingString
from OneStepData.OneStepDataLoader import getPrev
from ASearch.Ctype import getDistRange
def addToTrainingData(newArrs, origs, newDists, origDists):
    dataLen = len(newArrs)
    for i in range(dataLen):
        newDist = newDists[i]
        newArr = newArrs[i]
        orig = origs[i]
        origDist = origDists[i]
        n = len(newArr)
        printStr = makeTrainingString(newArr, newDist, orig, origDist)
        path = "Training/"+str(n)+ "-"+ str(newDist)
        fileName = path + "/Out-" + str(n-1) + "-" + str(origDist) + ".txt"
        with open(fileName, "a") as myfile:
            myfile.write(printStr + "\n")
            myfile.close()

newArrs = [
[25,23,1,6,4,7,3,5,2,8,13,11,14,10,12,9,15,20,18,21,17,19,16,22,24],
[24,1,6,4,7,3,5,2,8,13,11,14,10,12,9,15,20,18,21,17,19,16,22,25,23],
[25,23,1,7,5,3,6,2,4,8,14,12,10,13,9,11,15,21,19,17,20,16,18,22,24],
[24,1,7,5,3,6,2,4,8,14,12,10,13,9,11,15,21,19,17,20,16,18,22,25,23],]
newDists = [28,28,28,28]
origs = []
origDists = []

for i in range(len(newArrs)):
    newArr = newArrs[i]
    newDist = newDists[i]
    orig = getPrev(newArr)
    origs.append(orig)
    origDist = getDistRange(orig, newDist-2, newDist+2)
    origDists.append(origDist)
addToTrainingData(newArrs, origs, newDists, origDists)