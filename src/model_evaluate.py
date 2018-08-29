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

from keras.models import load_model
from utils.datasets import DataManager
from utils.preprocessor import preprocess_input

# parameters
batch_size = 16
input_shape = (64, 64, 1)
verbose = 1
num_classes = 7
base_path = '../trained_models/test_models/'
model_path = base_path + 'fer2013_0828_GAP_concate_XCEPTION.04-0.6637.hdf5'

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
# computers: GeForce GT 730, i7-7700 CPU @3.60GHz × 8 
# 1.author_github: ('test loss:', 0.954571045119306) + ('accuracy:', 0.6597938144495973) 65.98%, time=3.98~4.2ms/step
#   fer2013_mini_XCEPTION.107-0.66.hdf5
# 2. author_aug_disgust_training: ('test loss:', 0.9779383508266928) + ('accuracy:', 0.6692672053662871) 66.93%, time=4.27ms/step
#   fer2013_mini_Xception_author_traindata_disgust.198-0.6559.hdf5
# 3. InveptionV3 + Sep network
#  'fer2013_0828_GAP_concate_XCEPTION.04-0.6637.hdf5' = ('test loss:', 1.0027647953024812)+('accuracy:', 0.6767901922707174) 67.68%, time=10ms/step
# 
# 
# #  author + unsampling: ('test loss:', 0.9773935891353642) + ('accuracy:', 0.6609083310280314) 66.09%, time = 11ms/step