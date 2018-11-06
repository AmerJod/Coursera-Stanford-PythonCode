# Machine Learning Online Class
# Exercise 2: Logistic Regression


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


## ==================== Part 1: Basic Function ====================

def plotData(X, y):

    accepted = np.where(y == 1) # find all the index for accepted microchips
    rejected = np.where(y == 0) # find all the index for rejected microchips

    accepted_X_0 = X[:, 0].take(accepted)  # map between the index for accepted microchips and data for  test 1
    accepted_X_1 = X[:, 1].take(accepted)  # map between the index for accepted microchips and data for  test 2

    rejected_X_0 = X[:, 0].take(rejected)  # map between the index for rejected microchips and data for  test 1
    rejected_X_1 = X[:, 1].take(rejected)  # map between the index for rejected microchips and data for  test 2

    fig, ax = plt.subplots()
    ax.scatter(accepted_X_0, accepted_X_1, marker='+', c="red", label='Accepted', alpha=0.7) # 1
    ax.scatter(rejected_X_0, rejected_X_1, marker='^', c="green", label='Rejected', alpha=0.7) # 0

    ax.legend(loc='lower right',)

    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    #plt.show(block=False)
    plt.show()
    return plt



def loaddata():
    data = np.loadtxt('ex2data2.txt', delimiter=',')
    # help to handle the missing values
    # file  =np.genfromtxt('test.csv', delimiter=';')[:, :-1]
    return data



def hypothesis(X, theta):
    return X.dot(theta)

def sigmoid(hypothesis):
    return 1.0 / (1.0 + np.exp((-hypothesis)))

def mapFeature(X1, X2):
    degrees = 6
    out = np.ones((np.shape(X1)[0], 1))

    for i in range(1, degrees + 1):
        for j in range(0, i + 1):
            term1 = X1 ** (i - j)
            term2 = X2 ** (j)
            term = (term1 * term2).reshape(np.shape(term1)[0], 1)
            out = np.hstack((out, term))
    return out

def mapFeature2(x1, x2):
    # TODO: HOW ???????????
    pass

def costFunction(X, y, initial_theta, lamda):
    m = X.shape[0]
    hypo = sigmoid(hypothesis(X, initial_theta))
    term1 = -y.T.dot(np.log(hypo))
    term2 = (1 - y).T.dot(np.log(1 - hypo))
    left_term = (term1 - term2) / m
    regularization_term = (lamda/2*m) * (np.power(initial_theta, 2))
    return left_term + regularization_term


def gradientCost(X, y, theta, lamda):
    m = X.shape[0]
    grad = X.T.dot(sigmoid(X.dot(theta)) - y) / m
    grad[1:] = grad[1:] + ((theta[1:] * lamda) / m)
    return grad

def findMinTheta(theta, X, y, lamda):
    result = op.minimize(costFunctionOpt, theta, args=(X, y, lamda), method='BFGS',
                                     options={"maxiter": 500, "disp": True})
    return result.x, result.fun

def costFunctionOpt(theta, X, y, lamda):
    cost = costFunction(X, y, theta, lamda)
    return cost

def exucuteCostFucntion(X,y):
    #  Setup the data matrix appropriately
    [m, n] = X.shape

    # add ones for the intercept term
    # X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # Initialize fitting parameters
    initial_theta = np.zeros((n, 1))

    # Compute and display initial cost and gradient
    lamda = 1.0
    cost = costFunction(X, y, initial_theta, lamda)

    # TODO: need to be fixed :(
    theta, cost = findMinTheta(initial_theta, X, y, lamda)





if __name__ == '__main__':
    # load the data
    data = loaddata()

    # get the prediction
    # X = data[:, [0, 1]]
    X = data[:, :-1]
    y = data[:, -1:]

    # We start the exercise by first plotting the data to understand the problem we are working with.
    plt = plotData(X, y)
    X = mapFeature(data[:, 0], data[:, 1])
    exucuteCostFucntion(X, y)