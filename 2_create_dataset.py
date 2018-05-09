#!/usr/bin/python3
from os import walk
import csv
import numpy as np
import os
import datetime

#dir
fileDir = os.path.dirname(os.path.realpath(__file__))
dirLog = os.path.join(fileDir, 'Log')

dirEncodigns = os.path.join(fileDir, 'encodings')
dirEncodignsFaceRecognition = os.path.join(dirEncodigns, 'FaceRecognition')
dirEncodignsOpenFace = os.path.join(dirEncodigns, 'OpenFace')

dirDatasets = os.path.join(fileDir, 'datasets')
dirDatasetsFaceRecognition = os.path.join(dirDatasets, 'FaceRecognition')
dirDatasetsOpenFace = os.path.join(dirDatasets, 'OpenFace')

#file
fileLogError = os.path.join(dirLog, 'log_error.txt')

FaceEncodings = 'face_encoding.csv'
FaceNames = 'face_name.csv'

datasetName = 'dataset.csv'
datasetNameX = 'datasetX.csv'
datasetNameY = 'datasetY.csv'




raw_list = []
dir_list = []
encodings = ''
x = 0
y = 0

vzor_2 = False
vzor_3 = False
vzor_4 = False
vzor_5 = True

CreateFRdataset = True
CreateOFdataset = False

inp = input('Zvolit OpenFace? [no]: ')
if inp == 'y' or inp == 'yes':
    print('Zvoleny OpenFace dataset')
    CreateFRdataset = False
    CreateOFdataset = True
else:
    print('Zvoleny FaceRecognition dataset')


if CreateFRdataset:
    encodings = dirEncodignsFaceRecognition
    datasets = dirDatasetsFaceRecognition

if CreateOFdataset:
    encodings = dirEncodignsOpenFace
    datasets = dirDatasetsOpenFace


print(fileDir)
print(dirEncodigns)
print(dirEncodignsFaceRecognition)
print(dirEncodignsOpenFace)



for (dirpath, dirnames, filenames) in walk(encodings):
    dir_list.extend(dirnames)

#print(dir_list)

for i, dname in enumerate(dir_list):
    try:
        full_A = os.path.join(os.path.join(encodings, dname), FaceEncodings)
        full_B = os.path.join(os.path.join(encodings, dir_list[i+1]), FaceEncodings)
    except:
        print('Konec seznamu!')
        quit()

    print('{} | {}'.format(dname, dir_list[i+1]))

    with open(full_A) as my_csv:
        csvReader = csv.reader(my_csv, delimiter=',')
        A_list = []

        for row in csvReader:
            # print(', '.join(row))
            A_list.append(row)

    with open(full_B) as my_csv:
        csvReader = csv.reader(my_csv, delimiter=',')
        B_list = []

        for row in csvReader:
            # print(', '.join(row))
            B_list.append(row)

    a_set = []
    b_set = []
    c_set = []
    d_set = []
    e_set = []
    f_set = []

    if vzor_4 or vzor_5:
        g_set = []
        h_set = []
        i_set = []
        j_set = []
        k_set = []
        l_set = []

    # vzor 2
    if vzor_2:
        datasetName = 'dataset2.csv'
        datasetNameX = 'dataset2X.csv'
        datasetNameY = 'dataset2Y.csv'
        a_set = A_list[0] + B_list[0] + ['0']
        b_set = A_list[1] + A_list[2] + ['1']
        c_set = A_list[1] + B_list[1] + ['0']
        d_set = A_list[0] + A_list[1] + ['1']
        e_set = A_list[0] + B_list[1] + ['0']
        f_set = A_list[0] + A_list[2] + ['1']

    # vzor 3
    if vzor_3:
        datasetName = 'dataset3.csv'
        datasetNameX = 'dataset3X.csv'
        datasetNameY = 'dataset3Y.csv'
        a_set = A_list[0] + B_list[0] + ['0']
        b_set = A_list[1] + B_list[1] + ['0']
        c_set = A_list[2] + B_list[2] + ['0']
        d_set = A_list[0] + A_list[1] + ['1']
        e_set = A_list[0] + A_list[2] + ['1']
        f_set = A_list[1] + A_list[2] + ['1']

    # vzor 4
    if vzor_4:
        datasetName = 'dataset4.csv'
        datasetNameX = 'dataset4X.csv'
        datasetNameY = 'dataset4Y.csv'
        a_set = A_list[0] + B_list[0] + ['0']
        b_set = A_list[1] + B_list[1] + ['0']
        c_set = A_list[2] + B_list[2] + ['0']
        d_set = A_list[0] + A_list[1] + ['1']
        e_set = A_list[0] + A_list[2] + ['1']
        f_set = A_list[1] + A_list[2] + ['1']
        g_set = A_list[0] + B_list[1] + ['0']
        h_set = A_list[0] + B_list[2] + ['0']
        i_set = A_list[1] + B_list[0] + ['0']
        j_set = A_list[1] + B_list[2] + ['0']
        k_set = A_list[2] + B_list[0] + ['0']
        l_set = A_list[2] + B_list[1] + ['0']

    # vzor 5
    if vzor_5:
        datasetName = 'dataset5.csv'
        datasetNameX = 'dataset5X.csv'
        datasetNameY = 'dataset5Y.csv'
        a_set = A_list[0] + A_list[1] + ['1']
        b_set = A_list[0] + A_list[2] + ['1']
        c_set = A_list[1] + A_list[0] + ['1']
        d_set = A_list[1] + A_list[2] + ['1']
        e_set = A_list[2] + A_list[0] + ['1']
        f_set = A_list[2] + A_list[1] + ['1']
        g_set = A_list[0] + B_list[0] + ['0']
        h_set = A_list[0] + B_list[1] + ['0']
        i_set = A_list[0] + B_list[2] + ['0']
        j_set = A_list[1] + B_list[0] + ['0']
        k_set = A_list[1] + B_list[1] + ['0']
        l_set = A_list[1] + B_list[2] + ['0']

        #######

    # if len(a_set) == 257 and len(b_set) == 257 and len(c_set) == 257 and len(d_set) == 257 and len(e_set) == 257:
    if len(a_set) == 257 and len(b_set) == 257 and len(c_set) == 257 and len(d_set) == 257 and len(
            e_set) == 257 and len(f_set) == 257 and len(g_set) == 257 and len(h_set) == 257 \
            and len(i_set) == 257 and len(j_set) == 257 and len(k_set) == 257 and len(l_set) == 257:
        x += 1
        # print(x)

        os.path.join(dirDatasetsFaceRecognition)
        with open(os.path.join(datasets, datasetName), "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(a_set)
            writer.writerow(b_set)
            writer.writerow(c_set)
            writer.writerow(d_set)
            writer.writerow(e_set)
            writer.writerow(f_set)

            if vzor_4 or vzor_5:
                writer.writerow(g_set)
                writer.writerow(h_set)
                writer.writerow(i_set)
                writer.writerow(j_set)
                writer.writerow(k_set)
                writer.writerow(l_set)

        with open(os.path.join(datasets, datasetNameX), "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(a_set[0:256])
            writer.writerow(b_set[0:256])
            writer.writerow(c_set[0:256])
            writer.writerow(d_set[0:256])
            writer.writerow(e_set[0:256])
            writer.writerow(f_set[0:256])

            if vzor_4 or vzor_5:
                writer.writerow(g_set[0:256])
                writer.writerow(h_set[0:256])
                writer.writerow(i_set[0:256])
                writer.writerow(j_set[0:256])
                writer.writerow(k_set[0:256])
                writer.writerow(l_set[0:256])

        with open(os.path.join(datasets, datasetNameY), "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(a_set[256:])
            writer.writerow(b_set[256:])
            writer.writerow(c_set[256:])
            writer.writerow(d_set[256:])
            writer.writerow(e_set[256:])
            writer.writerow(f_set[256:])

            if vzor_4 or vzor_5:
                writer.writerow(g_set[256:])
                writer.writerow(h_set[256:])
                writer.writerow(i_set[256:])
                writer.writerow(j_set[256:])
                writer.writerow(k_set[256:])
                writer.writerow(l_set[256:])

    else:
        y += 1
        print('Problem s encodingem u osob {} a {}'.format(dname, dir_list[i + 1]))





