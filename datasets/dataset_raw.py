#!/usr/bin/python
# coding:utf8
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


emotion ={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}

dataset = 'jaffe'
datasets_path = dataset+'/'

file_csv = os.path.join(datasets_path, dataset+'.csv')
data = pd.read_csv(file_csv, dtype='a')
label = np.array(data['emotion'])
img_data = np.array(data['pixels'])

N_sample = label.size
Face_data = np.zeros((N_sample, 64*64))
Face_label = np.zeros((N_sample, 7), dtype=int)


for i in range(10):
    x = img_data[i]
    x = np.fromstring(x, dtype=float, sep=' ')
    x = x/x.max()
    img_x = np.reshape(x, (64, 64))
    plt.subplot(5,5,i+1)
    plt.axis('off')
    plt.title(emotion[int(label[i])])
    plt.imshow(img_x, plt.cm.gray)

plt.show()
