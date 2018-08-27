import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import os
import sys
import cv2
import time
import itertools
import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report

# import seaborn as sn

from keras.models import load_model
from models.cnn import mini_XCEPTION
from utils.datasets import get_labels
from utils.datasets import DataManager
from utils.datasets import load_fer2013_test
from utils.datasets import appendDFToCSV_void
from utils.preprocessor import preprocess_input



# parameters for loading data and images
input_shape = (64, 64, 1)

filename = '../datasets/data/confusion_matrix.csv'
file_dir = os.path.split(filename)[0]
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)
if os.path.exists(filename):
    os.remove(filename)

# loading test dataset, PrivateData
test_data_loader = load_fer2013_test('../datasets/fer2013/test.csv', image_size=input_shape[:2])
test_faces, test_emotions = test_data_loader[0] ,test_data_loader[1] 

# Load Model
emotion_model_path = '../trained_models/test_models/fer2013_mini_XCEPTION.107-0.66.hdf5'
emotion_labels = get_labels('fer2013')
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]

print '[+] Loading Data'
data = np.zeros((len(emotion_labels), len(emotion_labels)))
test_faces = np.reshape(test_faces,(test_faces.shape[1],test_faces.shape[2],test_faces.shape[3],test_faces.shape[4]))

for i in xrange(test_faces.shape[0]):
    # reshape image 
    test_ft = preprocess_input(test_faces[i], True)
    test_ft = np.reshape(test_ft,(1,test_ft.shape[0],test_ft.shape[1],test_ft.shape[2]))

    # inference a image emotion
    emotion_prediction = emotion_classifier.predict(test_ft)
    emotion_label_arg = np.argmax(emotion_prediction)
    emotion_probability = np.max(emotion_prediction)

    print test_ft.shape
    print np.argmax(emotion_prediction)
    print test_emotions[i]

    # data matrix recording the number of image
    data[test_emotions[i], emotion_label_arg] += 1

    clos = pd.DataFrame({'image':i,
                         'True': test_emotions[i],
                        'Predicted': emotion_label_arg,
                        'Proability':emotion_probability},
                        columns=['image', 'True', 'Predicted', 'Proability'],
                        index=np.arange(1))

    clos = appendDFToCSV_void(clos, filename)

# print data
# for i in range(len(data)):
#     total = np.sum(data[i])
#     # print total, len(data)
#     # print data[i] # 7-label total of inference image 
#     # print data[0]
#     # print data[1]
#     # print data[2]
#     # print data[3]
#     # print data[4]
#     # print data[5]
#     # print data[6]
#     for x in range(len(data[0])):
#         data[i][x] = data[i][x] / total  # inference accuracy instead of image numbers
#         print data[i][x]
# # inference proability for test dataset images
# print data

print('[+] Generating graph')
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class_names = ['angry','disgusted','fear','happy','sad','surprised','neutral']
label_file = pd.read_csv(filename)
y_true = label_file['True']
y_pred = label_file['Predicted']
y_true = np.array(y_true)
y_pred = np.array(y_pred)


#  the fraction of correct predictions
# mini_Xception_author = 0.6597938144329897, author_disgusted=0.6514349400947339, 
# 48x48=0.6539426023962106

accuracy = 0
accuracy = accuracy_score(y_true, y_pred)
print accuracy


# cohen_kappa_score: Scores above .8 are generally considered good agreement
# mini_Xception_author: 0.5884103182361573, author_disgusted=0.5789161870609528
# 48x48=0.5817710986824143
ck_score = 0
ck_score = cohen_kappa_score(y_true, y_pred)
print ck_score

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
print cnf_matrix
np.set_printoptions(precision=2)

# classification_report: precision, recall, f1-score, support
''' precision: 
    recall:  
    f1-score: 
    support: 
    detail: http://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix '''
classification_metric = classification_report(y_true, y_pred, target_names=class_names)
print classification_metric

# hanming loss
# mini_Xception_author = 0.3402061855670103
distance_loss = 0
distance_loss = hamming_loss(y_true, y_pred)
print distance_loss

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


plt.savefig('../images/confusion_matrix/'+ emotion_model_path[38:-5] + 'confusion_matrix.png', format='png')
plt.show()



# Cohen's kappa: a statistic that measures inter-annotator agreement
# cohen_kappa_score(y1, y2[, labels, weights, ... ]) 

# Compute confusion matrix to evaluate the accuracy of a classification
# confusion_matrix(y_true, y_pred[, labels,..])   

# Average hinge loss (non-regularized)
# hinge_loss(y_true, pred_decision[, labels, ..])  

# Compute the Matthews correlation coefficient (MCC)
# matthews_corrcoef(y_true, y_pred[, ..])  