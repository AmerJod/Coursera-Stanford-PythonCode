# Machine Learning Online Class
# Exercise 5 | Regularized Linear Regression and Bias-Variance

import scipy.io
import scipy.misc
import scipy.optimize
import scipy.special
import matplotlib.pyplot as plt
import numpy as np



def loaddata(filename, op=1):
    if op == 1:
        print('Loading and Visualizing Data from %s ...' % filename)
    else:
        print('\nLoading Saved Neural Network Parameters from %s ...' % filename)
    data = scipy.io.loadmat(filename)
    # help to handle the missing values
    # file  =np.genfromtxt('test.csv', delimiter=';')[:, :-1]
    return data

def displayData(X, y, theta = None, addLine = False):

    plt.xlabel('Change in water level(x)')
    plt.ylabel('Water flowing out of the dam(y)')

    if addLine:
        X_bias = X
        X = X[:,1:]
        m, n = X.shape
        plt.scatter(X, y, marker='x', c='r', s=30, linewidth=2)
        plt.plot(X, X_bias.dot(theta), linewidth=2)
    else:
        plt.scatter(X, y, marker='x', c='r', s=30, linewidth=2)

    plt.show()

def addBais(x):
    m , n = x.shape
    new_row = np.ones((m, 1))
    x = np.concatenate((new_row, x), axis=1)  # add bias
    return x

def linearRegCostFunction(X, y, theta, lamda):
    m, n = X.shape
    new_row = np.ones((m, 1))
    X = np.concatenate((new_row, X), axis=1)  # add bias
    theta = np.array([[1],[1]])

    cost = computeCost(theta , X,  y, lamda)

    gradient = gradientDescent(theta, X, y, lamda)

    print('Cost at theta = [1 ; 1]: %s this value should be about 303.993192)\n' % cost)
    print('Gradient at theta = [1 ; 1]:  [%f; %f] (this value should be about [-15.303016; 598.250744])\n' % (
    gradient[0], gradient[1]))

    return  cost, gradient

def computeCost(theta, X, y, lemda):
    m, n  = X.shape
    theta = theta.reshape(X.shape[1], 1) # we have to do it like that for the optimization methods
    cost_term = np.sum(np.power(X.dot(theta) - y, 2)) / (2 * m)

    new_theta = theta[1:, :]     #removing theta_0
    regularization_term = np.sum(np.power(new_theta, 2)) * lemda / (2 * m)

    cost = cost_term + regularization_term
    return cost

def gradientDescent(theta, X, y, lamda):
    m, n  = X.shape
    theta = theta.reshape(X.shape[1], 1) # we have to do it like that for the optimization methods
    gradient = ((X.dot(theta) - y).T.dot(X)) / m  # for j = 0
    gradient[:, 1:] = gradient[:, 1:] + theta[1:] * lamda / m  # for j >= 1
    #d_1 = (((X.dot(theta) - y).T.dot(X)) / m)[:,[1]]  + (theta[1:] * lamda / m)
    return gradient[0]

def learningCurve(X, y, Xval, yval, lamda):
    m, n = X.shape
    #mVal, nVal = X.shape

    X    = addBais(X)
    Xval = addBais(Xval)
    error_train = []
    error_val   = []

    for i in range(0, m):
        cost, theta = trainLinearReg(X[0:i + 1, :], y[0:i + 1, :], lamda) # we start with one traning set, than two and so on.
        error_train.append(computeCost(theta, X[0:i + 1, :], y[0:i + 1, :], lamda))
        error_val.append(computeCost(theta, Xval, yval, lamda))

    error_train = np.array(error_train)
    error_val = np.array(error_val)
    displayLaringCurve(m,error_train,error_val)


def displayLaringCurve(numberOfTrainingExample, error_train, error_val):
    temp = np.array([x for x in range(1, numberOfTrainingExample + 1)])   # number of training examples
    plt.ylabel('Error')
    plt.xlabel('Number of training examples')
    plt.plot(temp, error_train, color='red', linewidth=2, label='Train')
    plt.plot(temp, error_val, color='blue', linewidth=2, label='Cross Validation')
    plt.legend()
    plt.show()

def trainLinearReg(X, y, lamda, disp = False):
    theta = np.zeros((X.shape[1], 1))
    result = scipy.optimize.fmin_cg(computeCost, fprime=gradientDescent, x0=theta,
                                        args= (X, y, lamda), maxiter=200, disp=False, full_output=True)
    cost = result[1]
    new_theta = result[0]

    if disp:
        displayData(X, y, new_theta, True)

    return cost, new_theta

def polyFeatures(X, p):
    out = np.copy(X)
    for i in range(1, p):
        out = np.concatenate((out, X ** (i+1)), axis=1)
    return out

def featureNormalize(data):
    mu = np.mean(data, axis=0)
    data_norm = data - mu
    sigma = np.std(data_norm, axis=0, ddof=1)
    data_norm = data_norm / sigma
    return data_norm, mu, sigma


if __name__ == "__main__":
    # ====================
    # Part 1: Basic Function &&
    # Part 2: Regularized Linear Regression Cost &&
    # Part 3: Regularized Linear Regression Gradient
    # =============
    data = loaddata('ex5data1.mat', op=1)
    X = data['X']
    y = data['y']

    #displayData(X,y)
    lamda = 1
    theta = None
    linearRegCostFunction(X, y, theta, lamda)

    # =========== Part 4: Train Linear Regression  =============
    lamda = 0
    theta = np.array([[0], [0]])  # initialize theta with 0s
    X_bias = addBais(X)
    #trainLinearReg(X_bias, y, lamda, disp=True)

    # =========== Part 5: Learning Curve for Linear Regression  =============
    lamda = 0.0

    Xval  = data['Xval']
    yval  = data['yval']

    learningCurve(X, y, Xval, yval, lamda)
    # =========== Part 6: Feature Mapping for Polynomial Regression =============
    p = 8
    X = data['X']
    y = data['y']
    Xtest = data['Xtest']
    ytest = data['ytest']

    #  Map X onto Polynomial Features and Normalize
    X_poly = polyFeatures(X, p)
    X_poly, mu, sigma = featureNormalize(X_poly)
    X_poly = addBais(X_poly)


    #  Map X_poly_test and normalize (using mu and sigma)
    X_poly_test = polyFeatures(Xtest, p)
    X_poly_test = X_poly_test - mu
    X_poly_test = X_poly_test / sigma
    X_poly_test = addBais(X_poly_test)

    #  Map X_poly_val and normalize (using mu and sigma)
    X_poly_val = polyFeatures(Xval, p)
    X_poly_val = X_poly_val - mu
    X_poly_val = X_poly_val / sigma
    X_poly_val = addBais(X_poly_val)

    print('Normalized Training Example 1:\n')
    print('  %s  ' % X_poly[0, :])

    #learningCurve(X_poly, y, X_poly_val, yval, lamda)

    # =========== Part 7: Learning Curve for Polynomial Regression ============= OPTIONAL



