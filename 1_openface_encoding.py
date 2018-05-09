#!/usr/bin/python2
from os import walk
import csv
import numpy as np
import cv2
import os
import face_recognition
import datetime
import openface

#funkcni

#dir
fileDir = os.path.dirname(os.path.realpath(__file__))
dirPhoto = os.path.join(fileDir, 'photo')
dirLog = os.path.join(fileDir, 'Log')
dirEncodigns = os.path.join(fileDir, 'encodings')
dirEncodignsOpenFace = os.path.join(dirEncodigns, 'OpenFace')


modelDir = os.path.join(fileDir, 'of_dlib')


align = openface.AlignDlib(os.path.join(modelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(os.path.join(modelDir, 'nn4.small2.v1.t7'), 96) # imgDim 96

#file
fileLogError = os.path.join(dirLog, 'log_error.txt')

FaceEncodings = 'face_encoding.csv'
FaceNames = 'face_name.csv'

print(fileDir)
print(dirPhoto)
print(dirEncodigns)
print(dirEncodignsOpenFace)

dir_list = []
dir_list_count = []


def encodingOF(imgPath):
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        with open(fileLogError, 'a') as file:
            msg = str(datetime.datetime.now()) + ' || recognition - OpenFace || Fotografii {} nelze nahrat.'.format(os.path.basename(imgPath)) + '\n'
            # print(msg)
            file.write(msg)
        print("Nelze nahrat img: {}".format(imgPath))
        return [1, 2, 3]
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        with open(fileLogError, 'a') as file:
            msg = str(datetime.datetime.now()) + ' || recognition - OpenFace || Na fotografii {} nebynalezeny oblicej.'.format(os.path.basename(imgPath)) + '\n'
            # print(msg)
            file.write(msg)
        print("Nenalezeny oblicej: {}".format(imgPath))
        return [1, 2, 3]

    alignedFace = align.align(96, rgbImg, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        with open(fileLogError, 'a') as file:
            msg = str(datetime.datetime.now()) + ' || recognition - OpenFace || Fotografii {} nelze zarovnat.'.format(os.path.basename(imgPath)) + '\n'
            # print(msg)
            file.write(msg)
        print("IMG nelze zarovnat: {}".format(imgPath))
        return [1, 2, 3]

    encoding = net.forward(alignedFace)

    return encoding

#TODO dodelat OpenFace encoding

for (dirPath, dirnames, filenames) in walk(dirPhoto):
    print(filenames)
    if len(filenames) is not 0:
        filenames.sort()
        print(filenames)

    for i,filePhoto in enumerate(filenames):
        fullPathPhoto = os.path.join(dirPath, filePhoto)
        print(fullPathPhoto)

        """
        if os.path.isfile(dirpath + '/encodings.csv') == True:
            os.remove(dirpath + '/encodings.csv')
        """

        print(os.path.basename(dirPath))
        newDir= os.path.join(dirEncodignsOpenFace, os.path.basename(dirPath))
        print(newDir)
        if not os.path.exists(newDir):
            os.makedirs(newDir)

        image = face_recognition.load_image_file(fullPathPhoto)
        face_location = face_recognition.face_locations(image, 1, "hog")
        #face_location = [[1,2,3],[1,2,3]]
        #face_location = [[1, 2, 3]]

        if len(face_location) < 2:
            encoding = np.array(encodingOF(fullPathPhoto))
            #print('jeden')

        else:
            encoding = np.array([1,2,3])
            print('Nalezeno vice obliceju nez jeden, pocet obliceju: {}. Vice informaci v logu.'.format(len(face_location)))
            with open(fileLogError, 'a') as file:
                msg = str(datetime.datetime.now()) + ' || encoding - OpenFace || Na fotografii {} bylo nalezeno vice nez jeden obliceju. ' \
                                                     'Fotografie nebude pro zakodovani obliceje pouzita.  ' \
                                                     'Pocet obliceju: {}'.format(filePhoto, len(face_location)) + '\n'
                #print(msg)
                file.write(msg)


        if encoding.size < 128:
            encoding = [1,2,3]

        with open(os.path.join(newDir, FaceEncodings), 'a') as file:
            writer = csv.writer(file)
            writer.writerow(encoding)

        with open(os.path.join(newDir, FaceNames), 'a') as file:
            writer = csv.writer(file)
            writer.writerow([filePhoto])
