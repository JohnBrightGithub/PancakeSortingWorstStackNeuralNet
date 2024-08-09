from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np
import tensorflow as tf




def get_modelPrevState(input_shape):
    print(input_shape)
    model = Sequential([
        Conv2D(8, (5,5), activation = 'relu', input_shape = input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(8, (5,5), activation = 'relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation = 'relu'),
        Dropout(0.2),
        Dense(5, activation = 'softmax') #from 2 less to 2 more
    ])
    return model

def compile_modelPrevState(model):
    opt = tf.keras.optimizers.Adam()
    acc = 'accuracy'
    loss = 'sparse_categorical_crossentropy'
    model.compile(optimizer = opt,
                 loss = loss,
                 metrics =[acc])

def train_modelPrevState(model, scaled_train_images, trainLabels):
    print(scaled_train_images.shape, " trainLabel Shape: ", trainLabels.shape)
    history = model.fit(scaled_train_images,trainLabels,epochs=50, batch_size=256)
    return history


def TrainRandomPrevState(trainingData, trainLabels, columns):
    global maxColumns
    maxColumns = columns
    n= len(trainingData[0][0])
    rows = len(trainingData)
    newTrainingData = np.zeros(shape=(1, rows, 2*n,n))
    for row in range(rows):
        temp1 = getInputPrevState(trainingData[row][0:n-1][0:n-1][0], trainingData[row][0:n-1][0:n-1][1])
        temp1 = temp1.reshape(1, 2*n, n)
        newTrainingData[0][row] = temp1[0]
    newTrainingData= newTrainingData.reshape((rows,2*n,n,1))
    model = get_modelPrevState(newTrainingData[0].shape)
    compile_modelPrevState(model)
    train_modelPrevState(model, newTrainingData, trainLabels)
    model.save("Models\\worstPancakeStackV2.keras")
    return model

def getInputPrevState(state, prevState):
    #convert training data into arrays for training
    retArr = np.zeros(shape=(2*maxColumns, maxColumns))
    retArr[0:maxColumns] = makeMatrix(state)
    retArr[(maxColumns):(2*maxColumns)] = makeMatrix(prevState)
    return retArr

maxColumns=35
def makeMatrix(state):

    n = len(state)
    retArr = np.zeros(shape=(maxColumns,maxColumns))
    for i in range(n):
        if(int)(state[i])!=0:
            retArr[i][(int)(state[i])-1]=1
    retArr = retArr.astype(int)
    return  retArr

