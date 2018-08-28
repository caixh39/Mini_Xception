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

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau,TensorBoard
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from models.cnn import mini_XCEPTION, mini_unsampling_Xception, GAP_concate_XCEPTION
from models.cnn import mini_FPN_Net, mini_scale_XCEPTION, Multi_XCEPTION
from utils.datasets import DataManager
from utils.preprocessor import preprocess_input

# parameters
batch_size = 32
input_shape = (64, 64, 1)
verbose = 1
num_classes = 7
base_path = '../trained_models/test_models/'
model_path = base_path + 'fer2013_mini_Xception_author.137-0.6556.hdf5'

model = load_model(model_path)
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


# result:
# author_github: ('test loss:', 0.954571045119306) + ('accuracy:', 0.6597938144495973)
# author_disgust_train: ('test loss:', 0.9779383508266928) + ('accuracy:', 0.6692672053662871)