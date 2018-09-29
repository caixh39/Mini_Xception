"""
File: train_emotion_classifier.py
Author: Octavio Arriaga
Email: arriaga.camargo@gmail.com
Github: https://github.com/oarriaga
Description: Train emotion classification model
"""
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# from keras.models import load_model
from tensorflow.python.keras._impl.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.keras._impl.keras.applications import mobilenet
from tensorflow.python.keras._impl.keras.models import load_model
from utils.datasets import DataManager
from utils.preprocessor import preprocess_input
from keras import backend as K

# parameters
batch_size = 32
input_shape = (64, 64, 1)
verbose = 1
num_classes = 7
base_path = '../trained_models/emotion_models/'
model_path = base_path + 'fer2013_MobileNet_0929.01-0.2948.hdf5'


## test MobileNet, there have custom function layer
with CustomObjectScope({'relu6': mobilenet.relu6,'DepthwiseConv2D': mobilenet.DepthwiseConv2D}):
    model = load_model(model_path)

# model = load_model(model_path)
model.summary()


# loading test dataset, PrivateData
test_data_loader = DataManager(dataset_mode='test', image_size=input_shape[:2])
test_faces, test_emotions = test_data_loader.load_fer2013()
test_faces = preprocess_input(test_faces)
num_samples, num_classes = test_emotions.shape 


# model evaluate
test_loss, test_accuracy = model.evaluate(test_faces, test_emotions, batch_size, verbose=1)
print('test loss:', test_loss)
print('accuracy:', test_accuracy)



