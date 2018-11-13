# Machine Learning Online Class
# Exercise 4 Neural Network Learning

import scipy.io
import scipy.misc
import scipy.optimize
import scipy.special
import matplotlib.pyplot as plt
import numpy as np



# Setup the parameters you will use for this exercise
input_layer_size  = 400   # 20x20 Input Images of Digits
hidden_layer_size = 25    # 25 hidden units
num_labels = 10           # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

def loaddata(filename, op=1):
    if op == 1:
        print('Loading and Visualizing Data from %s ...' % filename)
    else:
        print('\nLoading Saved Neural Network Parameters from %s ...' % filename)

    data = scipy.io.loadmat(filename)
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

        result_matrix

    plt.show()

def nnCostFunction(X, y, theta_param, lamda, number_labels):
    m, n = X.shape
    theta_data = theta_param

    param_theta = unroll(theta_data)
    cost = computeCostFunction(X, y, param_theta, lamda, number_labels )
    return cost

def nnCostFunction_backprop(X, y, number_labels=10):
    m, n = X.shape
    lamda = 1
    number_labels = 10
    # Initializing Pameters
    theta1 = randInitializeWeights(400, 25)
    theta2 = randInitializeWeights(25, 10)
    theta_data = []
    theta_data.append(theta1)
    theta_data.append(theta2)

    cost = computeCostFunction(X, y, theta_data, lamda, number_labels )


   # Backpropagation algorithm
    y_labels = recodeYLabel(y, num_labels)

    # for i in range(1,m):
    #     a_1 = X[i]
    a1, a2, a3, z2, z3 = forwardPropagation(X, theta1, theta2)
    delta3 = a3 - y_labels
    delta2 = theta2.T.dot(delta3) * (sigmoidGradient(a2))
    delta2 = delta2[1:, :]  # removing theta2_0

    accum1 = delta2.dot(a1.T) / m
    accum2 = delta3.dot(a2.T) / m

    n_theta1 = accum1[:, 1:] + (theta1[:, 1:] * lamda / m)
    n_theta2 = accum2[:, 1:] + (theta2[:, 1:] * lamda / m)

    theta_data = []
    theta_data.append(n_theta1)
    theta_data.append(n_theta2)

    ## cost = computeCostFunction(X, y, theta_data, lamda, number_labels)

    return cost, theta_data

def computeCostFunction(X, y, param_theta, lamda, number_labels):
    # get theta
    theta1 = param_theta[0]
    theta2 = param_theta[1]

    # calculate Forward propagation
    a1, a2, a3, z2, z3 = forwardPropagation(X, theta1, theta2)

    y_labels = recodeYLabel(y, number_labels)

    # wihtout regularization
    cost = costFunction(a3, X, y_labels)

    # with regularization
    cost_r = costFunction_r(a3, X, y_labels, theta1, theta2, lamda)
    return cost_r

def regualizeTheta(theta):
    theta = theta[:, 1:]  # don't sum the theta_0
    sum = np.sum(np.power(theta, 2))
    return sum

def forwardPropagation_test(X, theta1, theta2):
    m, n = X.shape
    new_row = np.ones((m, 1))
    a1 = np.concatenate((new_row, X), axis=1)  # add bias

    z2 = hypothesis(a1, theta1)
    a2 = sigmoid(z2)
    #m, n = a2.shape
    a2 = np.concatenate((new_row, a2.T), axis=1).T  # add bias

    z3 = hypothesis(a2, theta2)
    a3 = sigmoid(z3)

    return a1, a2, a3, z2, z3

def forwardPropagation(X, theta1, theta2):
    m, n = X.shape
    new_row = np.ones((m, 1))
    a1 = np.concatenate((new_row, X), axis=1).T  # add bias

    z2 = hypothesis(a1, theta1)
    a2 = sigmoid(z2)
    #m, n = a2.shape
    a2 = np.concatenate((new_row, a2.T), axis=1).T  # add bias

    z3 = hypothesis(a2, theta2)
    a3 = sigmoid(z3)

    return a1, a2, a3, z2, z3

def hypothesis(X, theta):
    #return X.dot(theta)
    return theta.dot(X)

def sigmoid(hypothesis):
    return 1.0 / (1.0 + np.exp((-hypothesis)))

def costFunction_r(hypo, X, y, theta1, theta2, lamda ):
    term1 = - y * (np.log(hypo))
    term2 = (1.0 - y) * (np.log(1.0 - hypo))
    m  = X.shape[0]
    cost = np.sum(term1 - term2) / m

    # regularization term
    regularization_term = (regualizeTheta(theta1) + regualizeTheta(theta2)) * lamda / (2 * m)

    f_cost = cost + regularization_term

    return f_cost

def costFunction(hypo, X, y):
    term1 = - y * (np.log(hypo))
    term2 = (1.0 - y) * (np.log(1.0 - hypo))
    m  = X.shape[0]
    cost = np.sum(term1 - term2) / m
    return cost

def unroll(theta_data):
    #input_layer_size = test_Theta_size[0][1]     # size of first/input layer
    #number_labels = test_Theta_size[-1][0]      # we get the last layer to get how many labels do we have

    #theta1_data = (input_layer_size) * hidden_layer_size
    #theta1_size = (input_layer_size, hidden_layer_size)
    #theta2_size = (hidden_layer_size, num_labels)
    # theta1 = nn_params[:theta1_data].T.reshape(theta1_size).T
    # theta2 = nn_params[theta1_data:].T.reshape(theta2_size).T

    theta1 = theta_data[0]
    theta2 = theta_data[1]

    return (theta1, theta2)

def recodeYLabel(y, k):
    m = y.shape[0]
    y_out = np.zeros((k, m))

    for i in range(0, m):
        y_out[y[i] - 1, i] = 1

    return y_out

def part1_3_Compute_Cost(X, y, theta_param):
    # Weight regularization parameter (we set this to 0 here).
    lamda = 0
    num_labels = 10
    cost = nnCostFunction(X, y, theta_param, lamda, num_labels)
    print('Cost at parameters (loaded from ex4weights): %s (this value should be about 0.287629)\n' % cost)

    lamda = 1
    cost = nnCostFunction(X, y, theta_param, lamda, num_labels)
    print('Cost at parameters (loaded from ex4weights): %s (this value should be about 0.383770)\n' % cost)

def part2_Compute_Sigmoid_Gradient(X, y):
    sig = sigmoidGradient(0)
    print('Sigmoid gradient evaluated at : ')
    print('Value: 0.25')
    sig = sigmoidGradient(np.array([1, -0.5, 0, 0.5, 1]));
    print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]: ')
    print('Value: %s \n' % sig)

    cost, theta_data = nnCostFunction_backprop(X,y)
    print("cost : %s"  % cost)

    return cost, theta_data

def sigmoidGradient(z):
    a = sigmoid(z)
    return a * (1 - a)

def randInitializeWeights(L_in, L_out):
    '''
        randomly initializes the weights of a layer with L_in incoming connections and L_out outgoing connections.
    '''
    e = 0.12
    w = np.random.random((L_out, L_in + 1)) * 2 * e - e

    return w

def debugInitializeWeights(fan_out, fan_in):
    '''
    Initialize the weights of a layer with fan_in incoming connections and fan_out outgoing connections using a fixed
    strategy, this will help you later in debugging
    '''

    num_elements = fan_out * (1 + fan_in)
    w = np.array([np.sin(x) / 10 for x in range(1, num_elements + 1)])
    w = w.reshape(1 + fan_in, fan_out).T
    return w


if __name__  == '__main__':
    data = loaddata('ex4data1.mat', op=1)
    X = data['X']
    y = data['y']

    # ==================== Part 1: Basic Function ====================
    displayData(X)

    # ================ Part 2: Loading Parameters ================
    data_with_weights = loaddata('ex4weights.mat', op=2)

    # Unroll parameters, we can work with any size

    m, n = X.shape
    test_Theta_data_rolled = np.ones((0)) #
    test_Theta_data = []
    # test_Theta_size = []
    # test_Theta_param = []
    for key in data_with_weights:
        if 'theta' in key.lower(): # only theta keys
            #test_Theta_data_rolled = np.concatenate((test_Theta_data_rolled, data_with_weights[key].T.flatten()), axis=0)
            test_Theta_data.append(data_with_weights[key])
            #test_Theta_size.append(data_with_weights[key].shape)


    # ================ Part 3: Compute Cost (Feedforward)  +  Part 4: Implement Regularization ================
    print('\nFeedforward Using Neural Network ...\n')
    part1_3_Compute_Cost(X, y, test_Theta_data)


    # ================ Part 5: Sigmoid Gradient + Part 6: Initializing Pameters ================
    part2_Compute_Sigmoid_Gradient(X, y)
    # Before start implementing the neural network, we will first
    # implement the gradient for the sigmoid function.
    #TODO: we have to implement fmincg
