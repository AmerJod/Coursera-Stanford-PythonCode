# Machine Learning Online Class -
# Exercise 1:Linear regression with multiple variables
## ==================== Part 1: Basic Function ====================

import numpy as np


def loaddata(filename):
    '''
        load data from text file
    '''
    data = np.loadtxt(filename, delimiter=',')
    # help to handle the missing values
    # data  =np.genfromtxt('test.csv', delimiter=';')[:, :-1]
    return data


# vectorization
def featureNormalizeWithoutLoop(X):
    '''
        Without for-loop
    '''
    mu = np.mean(X,0)              # returns a row vector
    sigma = np.std(X,0)            # returns a row vector
    m = np.size(X, 0)            # returns the number of rows in
    mu_matrix = np.ones((m, 1)) * mu
    sigma_matrix = np.ones((m, 1)) * sigma
    # subtract the mu matrix from X, and divide element-wise by the sigma matrix,
    # and arrive at X_normalized
    subtract_matrix = np.subtract(X , mu_matrix)
    X_norm = subtract_matrix / sigma_matrix
    return X_norm, mu, sigma


    # mu = mean(data, axis=0)
    # data_norm = data - mu
    # sigma = std(data_norm, axis=0, ddof=1)
    # data_norm = data_norm / sigma
    # return data_norm, mu, sigma


def featureNormalize(X):
    '''
    Normalizes the features in X
    :param X:
    :return: X
    '''
    X_norm = X;
    mu = np.zeros((1, np.size(X, 1)))
    sigma = np.zeros((1, np.size(X, 1)))

    for i in range (0 , np.size(X, 1)):
        mean_ = np.mean(X[:, i])
        sigma_ = np.std(X[:, i]) # standard diviation
        mu[0, i] = mean_
        sigma[0, i] = sigma_
        for j in range(np.size((X, 1))):
            X_norm[j, i] = (X[j, i] - mean_) / sigma_

    print('mu with loop')
    print(mean_)
    return  X_norm ,mu, sigma




def gradientDescentMulti(X, y, theta, alpha, num_iters):
    '''
        Performs gradient descent to learn theta
    '''

    m = np.size(X,0) # number of training examples
    J_history = []
    #theta = np.zeros(1, np.size(X[0,:]))
    n = np.size(X, 1)  # number of features (+ 1)

    for iter in range(1,num_iters):
        thetanew = theta
        for j in range(m):
            thetanew[j] = theta[j]  - (alpha / m) * sum(np.dot((((X * theta) - y )), X[:,j]))
        theta = thetanew
        J_history.append(computeCostMulti(X, y, theta))

    return  theta, J_history


def normalEqn(X, y):
    pass

def computeCostMulti(X, y, theta):
    '''
     computes the cost of using theta as the parameter for linear regression to fit the data points in X and y
    '''
    m = y.size
    #value = (1/2 * m) * sum((X*theta) - y)

    m = 2 * m
    prediction = X.dot(theta)
    sum_ = sum((prediction - y) ** 2)
    return sum_



if __name__ == '__main__':

    print('Plotting Data ...\n')
    data = loaddata('ex1data2.txt')
    X = data[:, [0,1]]
    y = data[:, [-1]]
    m = y.size # number of training examples

    # Scale features and set them to zero mean
    print('Normalizing Features ...\n')

    X, mu, sigma = featureNormalize(X)
    #X, mu, sigma = featureNormalizeWithoutLoop(X)

    # Add intercept term to X
    X  = np.concatenate((np.ones((m, 1)), X),axis=1)

    print('Running gradient descent ...\n')

    # Choose some alpha  value
    alpha = 0.01
    num_iters = 400

    # Init Theta and Run Gradient Descent
    theta = np.zeros((3, 1))
    print(theta)

    theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
