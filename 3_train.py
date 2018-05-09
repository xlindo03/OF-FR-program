#!/usr/bin/python3
from os import walk
import csv
import numpy as np
#import cv2
import os
#import face_recognition
import datetime
import time
from numpy import genfromtxt

from sklearn import tree
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

from sklearn.externals import joblib
from sklearn import preprocessing
import csv

def transformValues(predY, threshold):
    values = []
    for val in predY:
        if val <= threshold:
            values.append(0)
        else:
            values.append(1)
    return values


np.random.seed(2004)

#dir
fileDir = os.path.dirname(os.path.realpath(__file__))
dirLog = os.path.join(fileDir, 'Log')
dirModels = os.path.join(fileDir, 'models')

dirDatasets = os.path.join(fileDir, 'datasets')
dirDatasetsFaceRecognition = os.path.join(dirDatasets, 'FaceRecognition')
dirDatasetsOpenFace = os.path.join(dirDatasets, 'OpenFace')



#file
fileLogError = os.path.join(dirLog, 'log_error.txt')


datasetNameX = 'dataset5X.csv'
datasetNameY = 'dataset5Y.csv'

b_acc = 0
threshold = 0

CreateFRdataset = True
CreateOFdataset = False
normalize = False
showAcc = True
save = False

inp = input('Zvolit OpenFace? [ne]: ')
if inp == 'a' or inp == 'ano':
    print('Zvoleny OpenFace dataset')
    CreateFRdataset = False
    CreateOFdataset = True
else:
    print('Zvoleny FaceRecognition dataset')

inp = input('Provadet normalizeci dat? [ne]: ')
if inp == 'a' or inp == 'ano':
    print('Zvolena normalizace')
    normalize = True

inp = input('Ulozit model? [ne]: ')
if inp == 'a' or inp == 'ano':
    print('Model bude ulozen do {}'.format(dirModels))
    save = True

inp = input('Zobrazit presnost pro vsechny prahy? [ano]: ')
if inp == 'n' or inp == 'ne':
    print('Model bude ulozen do {}'.format(dirModels))
    showAcc = False


if CreateFRdataset:
    trainX = os.path.join(dirDatasetsFaceRecognition, datasetNameX)
    trainY = os.path.join(dirDatasetsFaceRecognition, datasetNameY)

if CreateOFdataset:
    trainX = os.path.join(dirDatasetsOpenFace, datasetNameX)
    trainY = os.path.join(dirDatasetsOpenFace, datasetNameY)


print(trainX)
print(trainY)

datasetX = genfromtxt(trainX, delimiter=',')
datasetY = genfromtxt(trainY, delimiter=',')

rows = datasetX.shape[0]
#print(rows)

end = int(round(rows*0.7, 0)) # vyber kolik % dat bude trenovaci
trainX = datasetX[0:end]
trainY = datasetY[0:end]
testX = datasetX[end:rows]
testY = datasetY[end:rows]

print("_______________________\n SVM - SVR \n_______________________")
clf = svm.SVR(C=100000, degree=3, kernel='rbf', gamma=1.9, shrinking=True, tol=1e-9, cache_size=500, verbose=True, max_iter=-1)

    #normalizace

    #X_normalized = preprocessing.normalize(trainX, norm='l2')
    #X_test_normalized = preprocessing.normalize(testX, norm='l2')

    # normalize
if normalize:
    trainX = preprocessing.normalize(trainX)
    testX = preprocessing.normalize(testX)

clf = clf.fit(trainX, trainY)
predY = clf.predict(testX)

for i in range(0, 100):
    i  = i / 100
    predY = transformValues(predY, i)
    acc = accuracy_score(testY, predY)
    if acc > b_acc:
        b_acc = acc
        threshold = i

    if showAcc:
        print('Prah: {} | Presnost: {}'.format(i, acc))

print('Nejlepsi presnost je {} pri prahu {}'.format(b_acc, threshold))



if save:
    tim = "{:.0f}".format(time.time())
    name = 'model_' + str(tim) + '.pkl'
    print('Natrenovany model je ulozeny. Nazev modelu: {}'.format(name))
    joblib.dump(clf, os.path.join(dirModels, name))

