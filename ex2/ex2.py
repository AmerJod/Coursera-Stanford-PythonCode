# Machine Learning Online Class
# Exercise 2: Logistic Regression


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

## ==================== Part 1: Basic Function ====================

def loaddata():
    data = np.loadtxt('ex2data1.txt', delimiter=',')
    # help to handle the missing values
    # file  =np.genfromtxt('test.csv', delimiter=';')[:, :-1]
    return data

def plotData(X,y):
    postive = np.where(y == 1) # find all the index that have 1
    negative = np.where(y == 0)  # find all the index that have 0

    fig, ax = plt.subplots()
    #ax.set_aspect(1)
    ax.scatter(X[:, 0].take(postive) , X[:, 1].take(postive),  marker= '+', c="red", label='Admitted',alpha=0.7 )
    ax.scatter(X[:, 0].take(negative) , X[:, 1].take(negative),  marker='^', c="green", label='Not Admitted',alpha=0.7)
    ax.legend(loc='lower right',)

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show(block=False)
    return plt

def hypothesis(X, theta):
    return X.dot(theta)

def sigmoid(hypothesis):
    return 1.0 / (1.0 + np.exp((-hypothesis)))


def costFunctionOpt(initial_theta,X,y):
    cost,gradient = costFunction(X,y,initial_theta)
    return cost

def costFunction(X, y,initial_theta):
    h  = sigmoid(hypothesis(X, initial_theta))
    term1 = y.T.dot(np.log(h))
    term2 = (1.0 - y).T.dot((np.log(1.0 - h)))
    m  = X.shape[0]
    cost = -(1.0 / m) * (term1 + term2)

    #gradient = (1.0 / m ) * (h - y).T.dot(X)
    gradient_ = gradient(X, y,initial_theta)
    return cost , gradient_

def gradient(X, y,initial_theta):
    m = X.shape[0]
    h = sigmoid(hypothesis(X, initial_theta))
    gradient = (1.0 / m) * (h - y).T.dot(X)
    return gradient

def exucuteCostFucntion(X,y):
    '''

    '''
    #  Setup the data matrix appropriately
    [m,n] = X.shape

    # Add intercept term to x and X_test
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # Initialize fitting parameters
    initial_theta = np.zeros((n+1, 1))

    # Compute and display initial cost and gradient
    cost, grad = costFunction(X, y, initial_theta)

    print('Cost at initial theta (zeros): %s\n' % cost)
    print('Expected cost (approx): 0.693\n')
    print('Gradient at initial theta (zeros): \n')
    print(' %s \n' % grad)
    print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

def testCostFunction(X, y):
    '''
        Compute and display cost and gradient with non-zero theta
    '''

    #  Setup the data matrix appropriately, and add ones for the intercept term
    [m,n] = X.shape

    # Add intercept term to x and X_test
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    test_theta = np.array([[-24] ,[ 0.2], [0.2]])

    cost, grad = costFunction(X, y, test_theta)

    print('\nCost at test theta: %s'% cost)
    print('Expected cost (approx): 0.218\n')
    print('Gradient at test theta: ')
    print(' %s \n' % grad)
    print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

def optimizeTheta(X ,y):
    '''
        fminunc to obtain the optimal theta
    '''

    [m,n] = X.shape
    initial_theta = np.zeros((n+1,1))
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    result = op.optimize.fmin(costFunctionOpt, x0=initial_theta, args=(X, y), maxiter=400, full_output=True)
    #print(result)
    theta = result[0]
    cost = result[1]

    # Print theta  to screen
    print('Cost at theta found by fminunc: %s\n' % cost)
    print('Expected cost (approx): 0.203\n')
    print('theta: \n')
    print(' %s \n' % theta)
    print('Expected theta (approx):\n')
    print(' -25.161\n 0.206\n 0.201\n')

    return  theta, cost

def plotBoundary(data, X, theta, plt):
    plot_x = np.array([min(X[:, 1]), max(X[:, 1])])
    plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])
    plt.plot(plot_x, plot_y, c="blue", linestyle='-')
    plt.show()


def predict(new_X,theta):
    pass



def predictForNewData(theta):
    '''
          Predict whether the label is 0 or 1 using learned logistic.
    '''

    new_Data = np.array([1, 45 ,85])
    h = sigmoid(hypothesis(new_Data, theta))
    print('For a student with scores 45 and 85, we predict an admission probability of %s' % h)
    print('Expected value: 0.775 +/- 0.002\n\n')

    if h > 0.5:
        predictiton = 1
    else:
        predictiton = 0

    # TODO: fix it
    print('Train Accuracy: %s' % (np.mean(predictiton == y) * 100))
    print('Expected accuracy (approx): 89.0\n')
    print('\n')


if __name__ == '__main__':
    print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')

    # load the data
    data = loaddata()

    # get the prediction
    X = data[:, :-1]
    y = data[:, -1:]

    # We start the exercise by first plotting the data to understand the problem we are working with.
    plt = plotData(X,y)
    #initial_theta = np.zeros((n+1,1))

    # we implement the cost and gradient for logistic regression
    exucuteCostFucntion(X,y)
    testCostFunction(X,y)

    # we use a built-in function (fminunc) to find the optimal parameters theta
    theta, cost = optimizeTheta(X,y)

    plotBoundary(data,X,theta,plt)

    predictForNewData(theta)
