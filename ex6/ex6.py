# Machine Learning Online Class
# Exercise 6 | Support Vector Machines

import scipy.io
import scipy.misc
import scipy.optimize
import scipy.special
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm


#
#
# import sys
# import scipy.misc, scipy.io, scipy.optimize
# from sklearn import svm, grid_search
# from numpy import *
#
# import pylab
# from matplotlib import pyplot, cm
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.mlab as mlaba
#
#






def loaddata(filename):

    print('Loading and Visualizing Data from %s ...' % filename)
    data = scipy.io.loadmat(filename)
    # help to handle the missing values
    # file  =np.genfromtxt('test.csv', delimiter=';')[:, :-1]
    return data

def plotData(X, y):
    postive = np.where(y == 1)  # find all the index that have 1
    negative = np.where(y == 0)  # find all the index that have 0

    fig, ax = plt.subplots()
    # ax.set_aspect(1)
    ax.scatter(X[:, 0].take(postive), X[:, 1].take(postive), marker='+', c="red", label='positive', alpha=0.7)
    ax.scatter(X[:, 0].take(negative), X[:, 1].take(negative), marker='o', c="green", label='negative',alpha=0.7)
    #plt.show(block=False)
    return plt


def plotBoundary(data, X, theta, plt):
    pass


def visualizeBoundary(X, trained_svm, plt = None):
    kernel = trained_svm.get_params()['kernel']
    if kernel == 'linear':
        w = trained_svm.dual_coef_.dot(trained_svm.support_vectors_).flatten()
        xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
        yp = (-w[0] * xp + trained_svm.intercept_) / w[1]
        plt.plot(xp, yp, 'b-')
        return plt


    elif kernel == 'rbf':
        x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
        x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)

        X1, X2 = np.meshgrid(x1plot, x2plot)
        vals = np.zeros(X2.shape)

        for i in range(0, X1.shape[1]):
            this_X = np.c_[X1[:, i], X2[:, i]]
            vals[:, i] = trained_svm.predict(this_X)

        plt.contour(X1, X2, vals, colors='green')
        return plt



def gaussianKernel(x1, x2, sigma):
    return np.exp(-sum((x1 - x2) ** 2.0) / (2 * sigma ** 2.0))


def dataset3Params(X, y , X_val , y_val):
    '''
        returns your choice of C and sigma for Part 3 of the exercise
        where you select the optimal (C, sigma) learning parameters to use for SVM
        with RBF kernel
    '''
    C = 1
    sigma = 0.3





if __name__ == '__main__':
    # =============== Part 1: Loading and Visualizing Data ================
    print('Loading and Visualizing Data ...\n')
    data = loaddata('ex6data1.mat')
    X = data['X']
    y = data['y']
    '''

    plt = plotData(X,y)

    # linear SVM with C = 1
    linear_svm = svm.SVC(C=1, kernel='linear')
    linear_svm.fit(X, y.ravel())

    #plt = plotData(X, y)
    plt = visualizeBoundary(X, linear_svm , plt)
    plt.show(block=True)

    # C = 100
    linear_svm.set_params(C=100)
    linear_svm.fit(X, y.ravel())

    #plt = plotData(X, y)
    plt = visualizeBoundary(X, linear_svm, plt)
    plt.show(block=True)


    '''

    # ===============
    # Part 3: Implementing Gaussian Kernel
    # Part 4: Visualizing Dataset 2
    # Part 5: Training SVM with RBF Kernel (Dataset 2)
    # ===============

    print('Evaluating the Gaussian Kernel ...')

    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2
    sim = gaussianKernel(x1, x2, sigma);

    print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f : %s \n (for sigma = 2, this value should be about 0.324652)'% ( sigma, sim))
    print('Loading and Visualizing Data ...\n')
    data = loaddata('ex6data2.mat')
    X = data['X']
    y = data['y']
    plt = plotData(X, y)
    plt.show(block=True)

    sigma = 0.01
    rbf_svm = svm.SVC(C=1, kernel='rbf', gamma=1.0 / sigma)  # gamma is actually inverse of sigma
    rbf_svm.fit(X, y.ravel())

    plt = plotData(X, y)
    plt = visualizeBoundary(X, rbf_svm,plt)

    plt.show(block=True)


    # ===============
    # Part 6: Visualizing Dataset 3
    # Part 7: Training SVM with RBF Kernel (Dataset 3)
    # ================

    data = loaddata("ex6data3.mat")
    X, y = data['X'], data['y']
    X_val, y_val = data['Xval'], data['yval']

    rbf_svm = svm.SVC(kernel='rbf')

    best = dataset3Params(X, y, X_val, y_val)
    rbf_svm.set_params(C=best['C'])
    rbf_svm.set_params(gamma=best['gamma'])
    rbf_svm.fit(X, y)

    plotData(X, y)
    visualizeBoundary(X, rbf_svm)
    plt.show(block=True)





