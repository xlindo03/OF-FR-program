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
dirDatasetsFaceRecognition = os.path.join(dirDatasets, 'FaceRecognition')
dirModels = os.path.join(fileDir, 'models')

tIMG = os.path.join(fileDir, 'tIMG')
defaultModelFR = 'default_svr_fr.pkl'
defaultModelFRnor = 'default_svr_fr_normalize.pkl'

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





def Facerecognition(files):
    clf = joblib.load(model)

    filesEncoding = {}
    filesPred = {}
    for i, file in enumerate(files):
        img1_load = face_recognition.load_image_file(os.path.join(tIMG, file))
        img1_encoding = face_recognition.face_encodings(img1_load)[0]

        if img1_encoding.size is 128:
        #img1_encoding = [1, 2, 3, i]

            #TODO vztvorit dict
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






inp = input('Zadej nazev modelu: [{}]: '.format(defaultModelFR))
if inp is '':
    model = os.path.join(dirModels, defaultModelFR)
else:
    model = os.path.join(dirModels, inp+'.pkl')




inp = input('Provadet normalizeci dat? [ne]: ')
if inp == 'a' or inp == 'ano':
    print('Zvolena normalizace')
    normalize = True
    model = os.path.join(dirModels, defaultModelFRnor)

inp = input('Nastaveni prahu presnosti: [0.5]: ')
try:
   threshold = float(inp)
except ValueError:
   print("Neni cislo!")
   threshold = 0.5


def transformValues(predY):
    values = []
    for val in predY:
        if val <= threshold:
            values.append(0)
        else:
            values.append(1)
    return values

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
    files.sort()

    #print('')
    pred = Facerecognition(files)

    for name, p in pred.items():
        print('{}: {}'.format(name, p[0]))  #p vraci list takze return je [0] nebo [1] kdyz bude p[0] tak bude jenom cislo

