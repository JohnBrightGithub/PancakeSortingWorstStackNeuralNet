from WorstStackNeuralNet.loadTrainingData import loadTrainingData
from WorstStackNeuralNet.TrainWorstStack import TrainWorstStack


def testLoadTrainingData():
    dat, labels = loadTrainingData(35)
    length = dat.shape[1]
    for i in range(length):
        print("arr1 ", dat[0][i])
        print("arr2 ", dat[1][i])
        print("label ", labels[i])

def testTrainWorstStack():
    TrainWorstStack(35)
testTrainWorstStack()