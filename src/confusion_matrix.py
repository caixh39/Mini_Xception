import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import sys
import cv2
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import os
from models.cnn import mini_XCEPTION
from utils.datasets import get_labels
from utils.datasets import DataManager
from utils.datasets import load_fer2013_test
from utils.preprocessor import preprocess_input



# parameters for loading data and images
input_shape = (64, 64, 1)
verbose = 1
num_label = 7
# emotion_prediction = []
# emotion_label_arg = []
emotion_model_path = '../trained_models/emotion_models/fer2013GAP_concate390k_XCEPTION.137-0.6886.hdf5'
emotion_labels = get_labels('fer2013')

# Load Model
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]


# model.summary()

# loading test dataset, PrivateData
test_data_loader = load_fer2013_test('../datasets/fer2013/test.csv', image_size=input_shape[:2])
test_faces, test_emotions = test_data_loader[0] ,test_data_loader[1] 


print '[+] Loading Data'
data = np.zeros((len(emotion_labels), len(emotion_labels)))
test_faces = np.reshape(test_faces,(test_faces.shape[1],test_faces.shape[2],test_faces.shape[3],test_faces.shape[4]))

for i in xrange(test_faces.shape[0]):
    print test_faces.shape
    # test_faces[i] = preprocess_input(test_faces[i], True)
    test_ft = preprocess_input(test_faces[i], True)
    test_ft = np.reshape(test_ft,(1,test_ft.shape[0],test_ft.shape[1],test_ft.shape[2]))
    # print(test_ft.shape)
    # os._exit()

    emotion_prediction = emotion_classifier.predict(test_ft)
    emotion_label_arg = np.argmax(emotion_prediction)
    print test_ft.shape
    print np.argmax(emotion_prediction)
    print test_emotions[i]
    data[test_emotions[i], np.argmax(emotion_prediction)] += 1

# Take % by column
for i in range(len(data)):
    total = np.sum(data[i])
    for x in range(len(data[0])):
        data[i][x] = data[i][x] / total


print(data)

print('[+] Generating graph')
c = plt.pcolor(data, edgecolors='k', linewidths=4,
               cmap='Blues', vmin=0.0, vmax=1.0)

EMOTIONS = ['angry', 'disgusted', 'fear',
            'happy', 'sad', 'surprised', 'neutral']

def show_values(pc, fmt="%.2f", **kw):
    pc.update_scalarmappable()
    # ax = pc.get_axes()
    ax = pc.axes
    ax.set_yticks(np.arange(len(emotion_labels)) + 0.5, minor=False)
    ax.set_xticks(np.arange(len(emotion_labels)) + 0.5, minor=False)
    ax.set_xticklabels(EMOTIONS, minor=False)
    ax.set_yticklabels(EMOTIONS, minor=False)
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


show_values(c)
plt.xlabel('Predicted Emotion')
plt.ylabel('Real Emotion')
plt.show()
plt.savefig('../images/confusion_matrix.png', format='png')





