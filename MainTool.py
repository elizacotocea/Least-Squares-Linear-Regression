import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
from sklearn import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from FileReader import loadData


def plotData3D(inputs, outputs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i1,i2,o in zip(inputs["GIB"],inputs["LIB"], outputs):
        xs = i1
        ys = i2
        zs = o
        ax.scatter(xs, ys, zs)

    ax.set_xlabel('GDP/capita')
    ax.set_ylabel('Freedom')
    ax.set_zlabel('Happiness')

    plt.title('GDP/capita & freedom vs. happiness')
    plt.show()


def plotModel(inputsTrain, outputsTrain, xref1, xref2, yref):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i1, i2, o in zip(inputsTrain["GIB"], inputsTrain["LIB"], outputsTrain):
        xs = i1
        ys = i2
        zs = o
        ax.scatter(xs, ys, zs)

    ax.set_xlabel('GDP/capita')
    ax.set_ylabel('Freedom')
    ax.set_zlabel('Happiness')

    ax.plot(xref1, xref2, yref, label='parametric curve')
    plt.title('GDP/capita & freedom vs. happiness')
    plt.show()

def getCoord(noOfPoints,inputs,w0,w1,w2):
    xref = []
    xref2 = []
    val2 = min(inputs["LIB"])
    val1 = min(inputs["GIB"])
    step2 = (max(inputs["LIB"]) - min(inputs["LIB"])) / noOfPoints
    step1 = (max(inputs["GIB"]) - min(inputs["GIB"])) / noOfPoints
    for i in range(1, noOfPoints):
        xref.append(val1)
        val1 += step1
        xref2.append(val2)
        val2 += step2
    yref = [w0 + w1 * x1 + w2 * x2 for x1, x2 in zip(xref, xref2)]
    return xref,xref2,yref

def run():
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, '2017.csv')
    data=loadData(filePath, 'Economy..GDP.per.Capita.', 'Freedom','Happiness.Score')

    inputs = {"GIB":data[0],"LIB":data[1]}
    outputs=data[2]
    plotData3D(inputs,outputs)
    np.random.seed(5)
    indexes = [i for i in range(len(inputs["GIB"]))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs["GIB"])), replace=False)
    testSample = [i for i in indexes if not i in trainSample]
    trainInputs={"GIB":[],"LIB":[]}
    for i in trainSample:
        trainInputs["GIB"].append(inputs["GIB"][i])
        trainInputs["LIB"].append(inputs["LIB"][i])

    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = {"GIB": [], "LIB": []}
    for i in testSample:
        testInputs["GIB"].append(inputs["GIB"][i])
        testInputs["LIB"].append(inputs["LIB"][i])
    testOutputs = [outputs[i] for i in testSample]

    X = []
    for i1, i2 in zip(trainInputs["GIB"], trainInputs["LIB"]):
        X.append([i1, i2])

    lm = linear_model.LinearRegression()
    lm.fit(X,trainOutputs)

    w0, w1, w2= lm.intercept_, lm.coef_[0], lm.coef_[1]
    print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x1', ' + ', w2, ' * x2')
    # plot the model
    noOfPoints = 1000

    xref, xref2, yref=getCoord(noOfPoints,trainInputs,w0,w1,w2)
    plotModel(trainInputs,trainOutputs,xref,xref2,yref)

    X=[]
    for i1, i2 in zip(testInputs["GIB"], testInputs["LIB"]):
        X.append([i1,i2])
    computedTestOutputs = lm.predict(X)


    xref, xref2, yref = getCoord(noOfPoints, testInputs, w0, w1,w2)
    plotModel(testInputs, computedTestOutputs, xref, xref2, yref)

    error = mean_squared_error(testOutputs, computedTestOutputs)
    print("prediction error (tool): ", error)

run()