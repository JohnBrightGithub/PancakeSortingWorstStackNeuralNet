
def makeChunks(chunk, k):
    n = len(chunk)
    retArr = []
    for i in range(k):
        for j in chunk:
            retArr.append(j+(n*i))
    return retArr

def makeMixedStack(chunk1, chunk2, k1):
    set1 = makeChunks(chunk1, k1)
    set2 = makeChunks(chunk2, 1)
    retSet = []
    for el in set1:
        retSet.append(el)
    for el in set2:
        el = el+(k1*(len(chunk1)))
        retSet.append(el)
    return retSet

def makeMixedStackStaggered(chunk1, chunk2):
    retSet = []
    k = 0
    n = len(chunk1)
    for i in range(k+2):
        for el in chunk1:
            el = el+(n*i)
            retSet.append(el)
    k+=2
    for el in chunk2:
        el = el+(n*k)
        retSet.append(el)
    k+=1
    for el in chunk1:
        el = el+(n*(k-1)) + 5
        retSet.append(el)
    return retSet

def makeAll32Stacks():
    retStacks = []
    stacks = [
    [1, 3, 7, 5, 2, 6, 4],
    [1, 5, 2, 7, 4, 6, 3],
    [1, 5, 3, 7, 4, 2, 6],
    [1, 5, 7, 3, 6, 4, 2],
    [1, 5, 7, 4, 2, 6, 3],
    [1, 6, 3, 5, 2, 7, 4],
    [1, 6, 3, 7, 5, 2, 4],
    [1, 6, 4, 7, 3, 5, 2],
    [1, 7, 4, 6, 2, 5, 3],
    [1, 7, 5, 3, 6, 2, 4],
    ]
    for stack in stacks:
        retStacks.append(makeChunks(stack,4))
    return retStacks

def makeAll31Stacks():
    retStacks = []
    stacks = [
    [1, 6, 4, 2, 7, 5, 3],
    [1, 7, 4, 6, 3, 5, 2],
    [1, 7, 5, 3, 6, 4, 2],
    [1, 4, 7, 3, 6, 2, 5],
    [1, 5, 3, 7, 2, 6, 4],
    ]
    for stack in stacks:
        retStacks.append(makeChunks(stack,4))
    return retStacks

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