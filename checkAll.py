from BruteForce.testChunks import testChunks
from BruteForce.testChunks import testChunksMixed
import argparse
from tqdm import tqdm
import itertools
def permutations(n):
    start = list(range(1,n+1))
    print(start)
    return list(itertools.permutations(start, n))
def getAdjacencies(state):
    adj = 0
    n=len(state)
    for i in range(n-1):
        if(state[i]!='a' and state[i+1]!= 'a'):
            if(state[i]!='R' and state[i+1]!= 'R'):
                if(abs(state[i]-state[i+1])==1):
                    adj=adj+1
    if(state[n-1]==n):
        adj = adj+1
    return adj

class CommandLineCheck:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-k", "--k", help = " number of chunks to test ", required = False, default = 4)
        parser.add_argument("-mc", "--makeChunksSelected", help = " to make regular chunks set to True, to make mixed chunks set to False ", required = False, default = True)
        
        argument = parser.parse_args()
        
        if argument.k:
            global max
            max = int(argument.k)
        if argument.makeChunksSelected:
            global makeChunksSelected
            makeChunksSelected = bool(argument.makeChunksSelected)

app = CommandLineCheck()
n=7
stacksToCheck = permutations(n)
statesNoAdj = []
for stack in stacksToCheck:
    if getAdjacencies(stack)==0:
        statesNoAdj.append(stack)
statesNoAdjLen = len(statesNoAdj)
pbar = tqdm(total=statesNoAdjLen)
if(makeChunksSelected):
    for stack in statesNoAdj:
        testChunks(stack, max)
        pbar.update(1)
else:
    for stack in statesNoAdj:
        testChunksMixed(stack, max)
        pbar.update(1)