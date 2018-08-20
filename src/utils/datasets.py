import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from scipy.io import loadmat
import pandas as pd
import numpy as np
from random import shuffle
import os
import cv2

class DataManager(object):
    """Class for loading fer2013 emotion classification dataset or
        imdb gender classification dataset."""
    def __init__(self, dataset_mode=None, dataset_path=None, image_size=(48, 48)):

        self.dataset_path = dataset_path
        self.image_size = image_size
        self.dataset_mode = dataset_mode

        if self.dataset_path != None:
            self.dataset_path = dataset_path
        elif self.dataset_mode == 'train':
            self.dataset_path = '../datasets/fer2013/train.csv'
        elif self.dataset_mode == 'val':   # fer2013 'PublicTest'
            self.dataset_path = '../datasets/fer2013/val.csv'
        elif self.dataset_mode == 'test':  # fer2013 'PrivateTest'
            self.dataset_path = '../datasets/fer2013/test.csv'
        else:
            raise Exception('Incorrect dataset name, please input fer2013 coresponding')



    # def get_data(self):
    #     if self.dataset_mode == 'fer2013':
    #         ground_truth_data = self._load_fer2013()
    #     return ground_truth_data

    def load_fer2013(self):
        data = pd.read_csv(self.dataset_path)
        pixels = data['pixels'].tolist() # nested list of array elements. pixels is a list 
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))

        faces = np.asarray(faces)          
        faces = np.expand_dims(faces, -1) # inserting a new axis, -1 equal to 1 ?
        emotions = pd.get_dummies(data['emotion']).as_matrix()

        return faces, emotions

    
def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0:'angry',1:'disgust',2:'fear',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}
    elif dataset_name == 'imdb':
        return {0:'woman', 1:'man'}
    elif dataset_name == 'KDEF':
        return {0:'AN', 1:'DI', 2:'AF', 3:'HA', 4:'SA', 5:'SU', 6:'NE'}
    else:
        raise Exception('Invalid dataset name')

def get_class_to_arg(dataset_name='fer2013'):
    if dataset_name == 'fer2013':
        return {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'sad':4,
                'surprise':5, 'neutral':6}
    elif dataset_name == 'imdb':
        return {'woman':0, 'man':1}
    elif dataset_name == 'KDEF':
        return {'AN':0, 'DI':1, 'AF':2, 'HA':3, 'SA':4, 'SU':5, 'NE':6}
    else:
        raise Exception('Invalid dataset name')

def split_imdb_data(ground_truth_data, validation_split=.2, do_shuffle=False):
    ground_truth_keys = sorted(ground_truth_data.keys())
    if do_shuffle == True:
        shuffle(ground_truth_keys)
    training_split = 1 - validation_split
    num_train = int(training_split * len(ground_truth_keys))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys

def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data


def appendDFToCSV_void(df, csvFilePath, sep=","):    
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + 
            " columns. CSV file has " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)