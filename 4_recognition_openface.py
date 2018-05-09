from os import walk
import csv
import numpy as np
import cv2
import os
import face_recognition
import datetime
import time
from numpy import genfromtxt
import itertools
from sklearn import svm

from sklearn.externals import joblib
from sklearn import preprocessing
import csv


#dir
fileDir = os.path.dirname(os.path.realpath(__file__))
dirLog = os.path.join(fileDir, 'Log')

dirDatasets = os.path.join(fileDir, 'datasets')
dirDatasetsOpenFace = os.path.join(dirDatasets, 'OpenFace')
dirModels = os.path.join(fileDir, 'models')
modelDir = os.path.join(fileDir, 'of_dlib')

align = openface.AlignDlib(os.path.join(modelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(os.path.join(modelDir, 'nn4.small2.v1.t7'), 96) # imgDim 96


tIMG = os.path.join(fileDir, 'tIMG')
defaultModelOF = 'default_svr_of.pkl'
defaultModelOFnor = 'default_svr_of_normalize.pkl'
dir1 = 'none'
dir2 = 'none'
files = []
dirs = []


print('Defaultni slozka: {}'.format(tIMG))


normalize = False

#file
fileLogError = os.path.join(dirLog, 'log_error.txt')

for (dirPath, dirnames, filenames) in walk(tIMG):
    dirs.extend(dirnames)
np.random.seed(2004)
dirs.sort()

def OFencoding(imgPath):
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        with open(fileLogError, 'a') as file:
            msg = str(datetime.datetime.now()) + ' || recognition - OpenFace || Fotografii {} nelze nahrat.'.format(os.path.basename(imgPath)) + '\n'
            # print(msg)
            file.write(msg)
        return [1, 2, 3]
        #raise Exception("Nelze nahrat img: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        with open(fileLogError, 'a') as file:
            msg = str(datetime.datetime.now()) + ' || recognition - OpenFace || Na fotografii {} nebynalezeny oblicej.'.format(os.path.basename(imgPath)) + '\n'
            # print(msg)
            file.write(msg)
        return [1, 2, 3]
        #raise Exception("Nenalezeny oblicej: {}".format(imgPath))

    alignedFace = align.align(96, rgbImg, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        with open(fileLogError, 'a') as file:
            msg = str(datetime.datetime.now()) + ' || recognition - OpenFace || Fotografii {} nelze zarovnat.'.format(os.path.basename(imgPath)) + '\n'
            # print(msg)
            file.write(msg)
        return [1, 2, 3]
        #raise Exception("IMG nelze zarovnat: {}".format(imgPath))

    encoding = net.forward(alignedFace)

    return encoding

def transformValues(predY, threshold):
    values = []
    for val in predY:
        if val <= threshold:
            values.append(0)
        else:
            values.append(1)
    return values




def Openface(files):
    clf = joblib.load(model)

    filesEncoding = {}
    filesPred = {}
    for i, file in enumerate(files):
        img1_encoding = OFencoding(os.path.join(tIMG, file))
        #img1_encoding = [1, 2, 3, i]

        # TODO vztvorit dict
        filesEncoding[file] = img1_encoding

    for (img1, img2) in itertools.combinations(filesEncoding, 2):
        img1_encoding = filesEncoding[img1]
        img2_encoding = filesEncoding[img2]

        if normalize:
            img1_encoding = preprocessing.normalize(img1_encoding)
            img2_encoding = preprocessing.normalize(img2_encoding)

        encodings = np.array([img1_encoding.tolist() + img2_encoding.tolist()])
        pred = clf.predict(encodings)
        pred = transformValues(pred)
        #pred = 1

        filesPred[img1 + ' - ' + img2] = pred

    return filesPred


def dirTest(name):
    if name is '':
        print('Pouzije se defaultni slozka.')
        return False
    if not os.path.isdir(os.path.join(tIMG, name)):
        print('{} neexistuje, zadej prosim znovu'.format(name))
        return True
    else:
        print('Pouzije se {}'.format(name))
        return False






model_inp = input('Zadej nazev modelu: [default_svr_.pkl]: ')
try:
    model = os.path.join(dirModels, model_inp)
except:
    print('Vyskytla se chyba, pouzije se defaulutni model.')
    model = os.path.join(dirModels, defaultModelOF)

inp = input('Provadet normalizeci dat? [ne]: ')
if inp == 'a' or inp == 'ano':
    print('Zvolena normalizace')
    normalize = True
    model = os.path.join(dirModels, defaultModelOFnor)


#start programu
while True:
    print('Slozky: {}'.format(dirs))
    dir1 = input('[1] Vyber slozku, ktera se ma pouzit k rozpoznavani: [tIMG/]: ')

    while dirTest(dir1):
        dir1 = input('[1] Vyber slozku, ktera se ma pouzit k rozpoznavani: [tIMG/]: ')

    dir1 = os.path.join(tIMG, dir1)

    for name in os.listdir(dir1):
        if os.path.isfile(os.path.join(dir1, name)):
            print(name)
            files.append(name)

    #print('')

    pred = Openface(files)

    for name, p in pred.items():
        print('{}: {}'.format(name, p[0]))  #p vraci list takze return je [0] nebo [1] kdyz bude p[0] tak bude jenom cislo
