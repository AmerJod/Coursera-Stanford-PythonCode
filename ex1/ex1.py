# Machine Learning Online Class -
# Exercise 1: Linear Regression

## ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.m

import numpy as np
import matplotlib.pyplot as plt

def warmUpExercise():
    '''
        function that returns the 5x5 identity matrix
    '''
    matrix = np.eye(5)
    print(matrix)

def loaddata(filename):
    '''
        load data from text file
    '''
    data = np.loadtxt(filename, delimiter=',')
    # help to handle the missing values
    # data  =np.genfromtxt('test.csv', delimiter=';')[:, :-1]
    return data

def plotData(X,y):
    fig, ax = plt.subplots()
    ax.scatter(X, y, marker='x', c="red", label='Admitted', alpha=0.7)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.title('Figure 1: Scatter plot of training data')
    plt.pause(0.05)
    plt.show(block=False)
    return plt

def computeCost(X, y, theta):
    '''
        computes the cost of using theta as the parameter for linear regression to fit the data points in X and y
    '''

    # number of training examples
    m = y.size
    # J = np.matrix.sum(np.multiply((np.dot(X,theta) - y),2)) / (2*m)
    # J =  sum(np.multiply((np.dot(X, theta) - y)و 2))[0] / (2 * m)

    m = 2 * m
    prediction = np.dot(X, theta)
    sum_ = sum((prediction - y) ** 2)

    J = sum_ / m

    return J

def gradientDescent(X, y, theta, alpha, iterations):
    '''
        Performs gradient descent to learn theta.
    '''
    # number of training examples
    m = y.size
    J_history = np.zeros((iterations, 1))

    #alpha_m = alpha / m
    for iter in range(1,iterations):
        for k in range(m):
            d_0 = (alpha / m) * sum((theta[0] + (np.dot(theta[1], X[k , 1]))) - y[k])
            d_1 = (alpha / m) * sum((theta[0] + (np.dot(theta[1], X[k , 1]))) - y[k]) * X[k , 1]

            theta[0]  = theta[0] - d_0
            theta[1]  = theta[1] - d_1

    return theta


def visualizing(X,y,theta):
    print('Visualizing J(theta_0, theta_1) ...\n')

    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

    # Fill out J_vals
    for i in range(theta0_vals):
        for j in range(theta1_vals):
            t = [theta0_vals[i] , theta1_vals[j]]
            J_vals[i, j] = computeCost(X, y, t)




    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals

    # # Surface plot
    # figure;
    # surf(theta0_vals, theta1_vals, J_vals)
    # xlabel('\theta_0');
    # ylabel('\theta_1');
    #
    # # Contour plot
    # figure;
    # # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    # contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
    # xlabel('\theta_0');
    # ylabel('\theta_1');
    #
    #
    # plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

if __name__ == '__main__':
    print('Running warmUpExercise ... \n');
    print('5x5 Identity Matrix: \n');
    warmUpExercise()

    print('Plotting Data ...\n')
    data = loaddata('ex1data1.txt')
    X = data[:, [0]]
    y = data[:, [-1]]
    m = y.size # number of training examples

    # Plot Data
    # Note: You have to complete the code in plotData.m
    plt = plotData(X, y);
    ii = 0
    # Add a column of ones to x
    X = np.concatenate((np.ones((m, 1)), X),axis=1)


    # initialize fitting parameters
    theta = np.zeros((2,1))

    # Some gradient descent settings
    iterations = 1500

    # learning rate
    alpha = 0.01

    print('\nTesting the cost function ...\n')
    # compute and display initial cost
    J = computeCost(X, y, theta)

    print('With theta = [0 ; 0] \nCost computed = %s' % J)
    print('Expected cost value (approx) 32.07\n')

    print('\nRunning Gradient Descent ...\n')
    # run gradient descent
    theta = gradientDescent(X, y, theta, alpha, iterations)

    # print theta to screen
    print('Theta found by gradient descent:\n')
    print(theta)
    # np.savetxt(sys.stdout, theta, fmt="%i")

    print('Expected theta values (approx)\n')
    print(' -3.6303\n  1.1664\n\n')


    #--------
    # Plot the linear fit

    plt.plot(X[:, 1], np.dot(X , theta), c="blue", linestyle='-')
    #plt.legend('Training data', 'Linear regression')

    plt.show(block=False)

    #Predict values for population sizes of 35,000 and 70,000
    predict1 = np.dot([1, 3.5] , theta)
    print('For population = 35,000, we predict a profit of %s' % (predict1 * 10000))
    predict2 = np.dot([1, 7] , theta)
    print('For population = 70,000, we predict a profit of %s' % (predict2 * 10000))


