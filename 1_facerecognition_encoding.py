#!/usr/bin/python3
from os import walk
import csv
import numpy as np
import cv2
import os
import face_recognition
import datetime

#funkcni

#dir
fileDir = os.path.dirname(os.path.realpath(__file__))
dirPhoto = os.path.join(fileDir, 'photo')
dirLog = os.path.join(fileDir, 'Log')
dirEncodigns = os.path.join(fileDir, 'encodings')
dirEncodignsFaceRecognition = os.path.join(dirEncodigns, 'FaceRecognition')

#file
fileLogError = os.path.join(dirLog, 'log_error.txt')

FaceEncodings = 'face_encoding.csv'
FaceNames = 'face_name.csv'

print(fileDir)
print(dirPhoto)
print(dirEncodigns)
print(dirEncodignsFaceRecognition)

dir_list = []
dir_list_count = []

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
        newDir= os.path.join(dirEncodignsFaceRecognition, os.path.basename(dirPath))
        print(newDir)
        if not os.path.exists(newDir):
            os.makedirs(newDir)

        image = face_recognition.load_image_file(fullPathPhoto)
        face_location = face_recognition.face_locations(image, 1, "hog")
            #face_location = [[1,2,3],[1,2,3]]

        if len(face_location) < 2:
            encoding = face_recognition.face_encodings(image)[0]
            #print('jeden')

        else:
            encoding = np.array([1,2,3])
            print('Nalezeno vice obliceju nez jeden, pocet obliceju: {}. Vice informaci v logu.'.format(len(face_location)))
            with open(fileLogError, 'a', newline='') as file:
                msg = str(datetime.datetime.now()) + ' || encoding - FaceRecognition || Na fotografii {} bylo nalezeno vice nez jeden obliceju. ' \
                                                         'Fotografie nebude pro zakodovani obliceje pouzita.  ' \
                                                         'Pocet obliceju: {}'.format(filePhoto, len(face_location)) + '\n'
                #print(msg)
                file.write(msg)


        if encoding.size < 128:
            encoding = [1,2,3]

        with open(os.path.join(newDir, FaceEncodings), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(encoding)

            with open(os.path.join(newDir, FaceNames), 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([filePhoto])
