import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


emotion ={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}

data = pd.read_csv('fer2013.csv', dtype='a')
label = np.array(data['emotion'])
img_data = np.array(data['pixels'])

N_sample = label.size
Face_data = np.zeros((N_sample, 48*48))
Face_label = np.zeros((N_sample, 7), dtype=int)


for i in range(25):
    x = img_data[i]
    x = np.fromstring(x, dtype=float, sep=' ')
    x = x/x.max()
    img_x = np.reshape(x, (48, 48))
    plt.subplot(5,5,i+1)
    plt.axis('off')
    plt.title(emotion[int(label[i])])
    plt.imshow(img_x, plt.cm.gray)
plt.show()
