def labelToNewDist(origDist, label):
    return origDist+(label-2)
def labelToOrigDist(newDist, label):
    return newDist-(label-2)

def distsToLabel(origDist, newDist):
    return newDist - origDist + 2