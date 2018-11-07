
import os
import cv2
import csv
import numpy as np
import pandas as pd
import matplotlib as plt
import tensorflow as tf
import keras 
import pdb

def detect(img, cascade):
	rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
	if len(rects) == 0:
		return []
	rects[:,2:] += rects[:, :2]
	return rects

cascade = cv2.CascadeClassifier('/home/cxh/Documents/Platform/opencv-3.3.1/data/haarcascades/haarcascade_frontalface_default.xml')

f = '/home/cxh/Documents/Datasets/JAFFE/jaffe/'
fs = os.listdir(f)
data = np.zeros([213, 64*64], dtype=np.uint8)
label = np.zeros([213], dtype=int)
i = 0
for f1 in fs:
    tmp_path = os.path.join(f, f1)
    if(os.path.splitext(tmp_path)[1] != '.tiff'):
        continue
    if not os.path.isdir(tmp_path):
        # print tmp_path[len(f):]
        img = cv2.imread(tmp_path,1)
        #pdb.set_trace()
        dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #pdb.set_trace()
        dst = np.expand_dims(dst, -1)
        # print dst
        rects = detect(dst, cascade)
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img,(x1+10,y1+20),(x2-10,y2),(0,255,255),2)
            # face area
            img_roi = np.uint8([y2-(y1+20), (x2-10)-(x1+10)])
            roi = dst[y1+20:y2, x1+10:x2-10]
            img_roi = roi
            re_roi = cv2.resize(img_roi, (64,64))
            # label
            img_label = tmp_path[len(f)+3:len(f)+5]
            # print(img_label)
            if img_label == 'AN':
                label[i] = 0
            elif img_label == 'DI':
                label[i] = 1
            elif img_label == 'FE':
                label[i] = 2
            elif img_label == 'HA':
                label[i] = 3
            elif img_label == 'SA':
                label[i] = 4
            elif img_label == 'SU':
                label[i] = 5
            elif img_label == 'NE':
                label[i] = 6
            else:
                print("get label error.......")

            data[i][0:64*64] = np.ndarray.flatten(re_roi)
           #  pdb.set_trace()
            i = i + 1

			# cv2.imshow('src',dst)
			# cv2.imshow('img', img)
			# if cv2.waitKey() ==32:
			# 	continue



with open('./JAFFE/jaffe.csv', 'w') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(['emotion', 'pixels'])
	dataInfo = []
	for i in range(len(label)):
		data_list = list(data[i])
		print data_list
		b = ' '.join(str(x) for x in data_list)
		l = np.hstack([label[i], b])
		#dataInfo.append([label[i],b])
		writer.writerow(l)



