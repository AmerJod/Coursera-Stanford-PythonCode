# Machine Learning Online Class -
#  Exercise 3 | Part 1: One-vs-all

## ==================== Part 1: Basic Function ====================


import math
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
#from scipy import *
import scipy.misc, scipy.optimize, scipy.io, scipy.special
import pandas



input_layer_size  = 400  # 20x20 Input Images of Digits
num_labels = 10          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)


def loaddata():
    data = scipy.io.loadmat('ex3data1.mat')
    # help to handle the missing values
    # file  =np.genfromtxt('test.csv', delimiter=';')[:, :-1]
    return data


def displayData(X, theta=None):
    width = 20
    rows, cols = 10, 10
    out = np.zeros((width * rows, width * cols))

    rand_indices = np.random.permutation(5000)[0:rows * cols]

    counter = 0
    for y in range(0, rows):
        for x in range(0, cols):
            start_x = x * width
            start_y = y * width
            out[start_x:start_x + width, start_y:start_y + width] = X[rand_indices[counter]].reshape(width, width).T
            counter += 1

    img = scipy.misc.toimage(out)
    figure = plt.figure()
    axes = figure.add_subplot(111)
    axes.imshow(img)

    if theta is not None:
        result_matrix = []
        X_biased = np.c_[np.ones(X.shape[0]), X]

        for idx in rand_indices:
            result = (np.argmax(theta.T.dot(X_biased[idx])) + 1) % 10
            result_matrix.append(result)

        result_matrix = np.array(result_matrix).reshape(rows, cols).transpose()
        print
        result_matrix

    plt.show()

def hypothesis(X, theta):
    return X.dot(theta)

def sigmoid(hypothesis):
    return 1.0 / (1.0 + np.exp((-hypothesis)))

def costFunction(theta, X, y, lamda):
    h  = sigmoid(hypothesis(X, theta))
    term1 = y.T.dot(np.log(h))
    term2 = (1.0 - y).T.dot((np.log(1.0 - h)))
    m  = X.shape[0]
    cost = - (term1 + term2) / m

    # regularization term
    theta = theta[1:]  # don't sum the theta_0
    #right_hand =  sum(np.power(theta, 2)) * lamda / (2 * m)
    right_hand = theta.T.dot(theta) * lamda / (2 * m) # victorized version

    return (cost + right_hand)

def gradientDescent(theta, X, y,lamda):
    m = X.shape[0]
    h = sigmoid(hypothesis(X, theta))
    theta[0] = 0
    gradient = (1.0 / m) * (((h - y).T.dot(X)).T + (theta * lamda))
    return gradient



def oneVsAll(X, y, number_classes, lamda):
    m, n = X.shape
    # Add a column of ones to x
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    # initialize a theta for each class
    all_thetas = np.zeros((n + 1,number_classes))
    all_costs  = [None] * 10

    print('Not Optimized: ')
    for k in range(0, number_classes):
        theta = np.zeros((1, 1+n))[0]
        y_k = ((y == (k + 1)) + 0).T[0]

        #cost_k = costFunction(theta, X, y_k, lamda)
        #grad_k = gradientDescent(theta, X, y_k, lamda)
        cost, grad = lrCostFunction(theta, X, y_k, lamda)
        all_costs[k] = cost
        all_thetas[:, k] = grad

        print("%d Cost: %.10f" % (k + 1, cost))

    print('Done\n')
    return all_thetas,all_costs


def oneVsAllOptimized(X, y, number_classes, lamda):
    m, n = X.shape
    # Add a column of ones to x
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    # initialize a theta for each class
    all_thetas = np.zeros((n + 1,number_classes))
    all_costs  = [None] * 10

    print('Optimized: ')
    for k in range(0, number_classes):
        theta = np.zeros((1, 1+n))[0]
        y_k = ((y == (k + 1)) + 0).T[0]

        result = scipy.optimize.fmin_cg(costFunction, fprime=gradientDescent, x0=theta,
                                        args=(X, y_k, lamda), maxiter=50, disp=False, full_output=True)
        all_costs[k] = result[1]
        all_thetas[:, k] = result[0]

        print("%d Cost: %.10f" % (k + 1, result[1]))

    print('Done - Optimized \n')

    return all_thetas, all_costs


def lrCostFunction(theta_t, X_t, y_t, lambda_t):
    # LRCOSTFUNCTION Compute cost and gradient for logistic regression with
    # regularization
    cost = costFunction(theta_t, X_t, y_t, lambda_t)
    grad = gradientDescent(theta_t, X_t, y_t, lambda_t)
    return cost, grad


def TestCostFunction():
    # Test case for lrCostFunction
    print('\nTesting lrCostFunction() with regularization')

    theta_t =np.array([[-2], [-1], [1], [2]])
    #X_t = [np.ones(5, 1).reshape(1:15, 5, 3) / 10]
    X_t = buildTestData()
    y_t =  np.array([[1] ,[0], [1], [0] ,[1]])
    #y_t = ([1;0;1;0;1] >= 0.5)

    lambda_t = 3

    cost, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)
    # cost = costFunction(theta_t, X_t, y_t, lambda_t)
    # grad = gradientDescent(theta_t, X_t, y_t, lambda_t)

    print('\nCost: %s' % cost)
    print('Expected cost: 2.534819\n')
    print('Gradients:')
    print(' %s ' % grad)
    print('Expected gradients:')
    print(' 0.146561 , -0.548558 ,  0.724722 , 1.398003\n')



def buildTestData():
    index = 0
    theta = np.ones((5, 3))

    for j in range(0, 3):
        for i in range(0, 5):
            index += 1
            theta[i, j] = float(index / 10)

    theta = np.concatenate((np.ones((5, 1)), theta), axis=1)
    return theta



def predictOneVsAll(theta, X, y):
    m, n = X.shape
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    correct = 0
    for i in range(0, m):
        prediction = np.argmax(theta.T.dot(X[i])) + 1  # get the largest value index
        actual = y[i]
        # print "prediction = %d actual = %d" % (prediction, actual)
        if actual == prediction:
            correct += 1

    print ("Accuracy: %.2f%%" % (correct * 100.0 / m))




if __name__ == '__main__':
    data = loaddata()
    X = data['X']
    y = data['y']

    # =========== Part 1: Loading and Visualizing Data =============
    #displayData(X)

    # ============ Part 2a: Vectorize Logistic Regression ============
    TestCostFunction()

    # ============ Part 2b: One-vs-All Training ============
    number_classes = 10
    lamda = 0.1

    oneVsAll(X, y, number_classes, lamda)
    all_thetas, cost = oneVsAllOptimized(X, y, number_classes, lamda)

    # ================ Part 3: Predict for One-Vs-All ================

    pred = predictOneVsAll(all_thetas, X, y)


    displayData(X, all_thetas)



