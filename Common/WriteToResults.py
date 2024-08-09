import os
from Common.commonFunc import labelToOrigDist
from Common.commonFunc import labelToNewDist
from Common.commonFunc import distsToLabel
def writeToTraining(r, origDist):
    fileArrs = []
    for i in range(origDist-2, origDist+3):
        fileArrs.append([])
    for package in r:
        for arrs, ans in package:
            print("ans: ", ans)
            fileArrs[ans].append(arrs)
    
    for dist in range(origDist-2, origDist+3):
        index = distsToLabel(origDist, dist)
        if(len(fileArrs[index])==0):
            continue
        n=len(fileArrs[index][0][1])
        print("n: ", n, " fileArrs[distsToLabel(origDist, dist)][0][0] ", fileArrs[index][0][0])
        path = "Training/"+str(n)+ "-"+ str(dist)
        if(not os.path.exists(path)):
            os.makedirs(path)
        fileName = path + "/Out-" + str(n-1) + "-" + str(origDist) + ".txt"
        print("writing to file ", fileName)
        if(os.path.exists(fileName)):
            os.remove(fileName)
        with open(fileName, "a") as myfile:
            del fileArrs[index][0] #get rid of the first empty list
            for arrs in fileArrs[index]:
                if(not arrs is None):
                    state = arrs[1]
                    origState = arrs[0]
                    printStr = makeTrainingString(state, dist, origState, origDist)
                    myfile.write(printStr + "\n")
            myfile.close()
def makeTrainingString(state, dist, origState, origDist):
    printStr = ""
    for ele in state:
        ele = int(ele)
        printStr = printStr + str(ele) + " "
    printStr = printStr.strip()  
    printStr += " " + str(dist)
    printStr +=' | '
    for ele in origState:
        ele = int(ele)
        printStr = printStr + str(ele) + " "
    printStr = printStr.strip()  
    printStr += " " + str(origDist)
    return printStr
def writeToResults(dist, state, origDist):
    n=len(state)
    path = "Results/"+str(n)+ "-"+ str(dist)
    if(not os.path.exists(path)):
        os.makedirs(path)
    fileName = path + "/Out-" + str(n-1) + "-" + str(origDist) + ".txt"
    print("writing ", state, " to ", fileName)
    with open(fileName, "a") as myfile:

        printStr = ""
        for ele in state:
            ele = int(ele)
            printStr = printStr + str(ele) + " "
        printStr = printStr.strip()  
        myfile.write(printStr + "\n")
        myfile.close()