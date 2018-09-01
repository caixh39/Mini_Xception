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


from keras.callbacks import ReduceLROnPlateau,TensorBoard
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from keras.models import load_model
import tensorflow as tf
import argparse

from utils.datasets import DataManager
from utils.preprocessor import preprocess_input
# from utils.visual_callbacks import AccLossPlotter
from models.cnn import mini_XCEPTION, GAP_concate_XCEPTION, mini_concate_V3_XCEPTION
from models.cnn import parameters_mini_XCEPTION, mini_concate_XCEPTION



# parameters
num_epochs = 10000
input_shape = (64, 64, 1)
verbose = 1
num_classes = 7
patience = 80
gpu_count = 4
batch_size = 32* gpu_count
base_path = '../trained_models/emotion_models/'
# models_path = base_path + 'fer2013_0831_mini_concate_V3_XCEPTION.127-0.6609.hdf5'

# retrain
# model = load_model(models_path)
# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

# model parameters/compilation; Configures the model for training
# optimizer: SGD suitable small datasets 
# check to see if we are compiling using just a single GPU

# Instantiate the base model
# (here, we do it on CPU, which is optional).
with tf.device('/cpu:0' if gpu_count > 1 else '/gpu:0'):
    model =  mini_concate_V3_XCEPTION(input_shape, num_classes)

# Replicates the model on N GPUs.
# This assumes that your machine has N available GPUs.
if gpu_count > 1:
    model = multi_gpu_model(model, gpus=gpu_count)
else:
    model = model


model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


datasets = ['fer2013']
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    # saving model after one epoch finishing
    trained_models_path = base_path + dataset_name + '_0901__mini_concate_V3_XCEPTION'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.4f}.hdf5' 
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1,
                                      save_best_only=True, mode='auto',period=1)

    # view on internal states and statistics of the model during training
    # a set callbacks functions, and visualization by tensorboard
    log_file_path = base_path + dataset_name + trained_models_path[40:] + '_emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_acc', factor=0.1,
                                  patience=int(patience/4), verbose=1)
    tensor_board = TensorBoard(log_dir='../log_dir',
                             histogram_freq=1,
                             write_graph=True,
                             write_images=True)
#    plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr, tensor_board]

    # loading train dataset
    train_data_loader = DataManager(dataset_mode='train', image_size=input_shape[:2])
    train_faces, train_emotions = train_data_loader.load_fer2013()
    train_faces = preprocess_input(train_faces)
    num_samples, num_classes = train_emotions.shape

    # loading val dataset, PublicData
    val_data_loader = DataManager(dataset_mode='val', image_size=input_shape[:2])
    val_faces, val_emotions = val_data_loader.load_fer2013()
    val_faces = preprocess_input(val_faces)
    num_samples, num_classes = val_emotions.shape   

    
    # Efficiency: generator run by paralle
    # Trains the model on data generated batch-by-batch by a Python generator
    model.fit_generator(data_generator.flow(train_faces, train_emotions,
                                            batch_size),
                        steps_per_epoch=len(train_faces) / batch_size,
                        shuffle=True,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=(val_faces, val_emotions))

