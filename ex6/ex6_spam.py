# Machine Learning Online Class
# Exercise 6 | Spam Classification with SVMs
#





import sys
import string
import csv
import re
import pickle

from numpy import *
import nltk, nltk.stem.porter
import scipy.misc, scipy.io, scipy.optimize
from sklearn import svm, model_selection

import pylab
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlaba


def readFile(filename):
    email_contents = ''
    print('Loading Data from %s ...' % filename)
    with open('filename', 'r') as file:
        email_contents = file.read()
    return email_contents

def getVocabList():
    vocab_list = {}
    with open('vocab.txt', 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            vocab_list[row[1]] = int(row[0])

    return vocab_list

def processEmail(email_contents):
    vocab_list = getVocabList()

    word_indices = []

    email_contents = email_contents.lower()
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    email_contents = re.sub('[$]+', 'dollar', email_contents)

    stemmer = nltk.stem.porter.PorterStemmer()
    tokens = re.split('[ ' + re.escape("@$/#.-:&*+=[]?!(){},'\">_<;%") + ']', email_contents)

    for token in tokens:
        token = re.sub('[^a-zA-Z0-9]', '', token)
        token = stemmer.stem(token.strip())

        if len(token) == 0:
            continue

        if token in vocab_list:
            word_indices.append(vocab_list[token])

    return word_indices


if __name__  == "__main__":
    # ====================
    # Part 1: Email Preprocessing
    # Part 2: Feature Extraction
    # Part 2: Feature Extraction
    # ====================

    print('Preprocessing sample email (emailSample1.txt)')

    #  Extract Features
    file_contents = readFile('emailSample1.txt')
    word_indices  = processEmail(file_contents)

    # Print Stats
    print('Word Indices: \n')
    print(' %s', word_indices)


