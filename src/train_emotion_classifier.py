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
from keras.preprocessing.image import ImageDataGenerator

from models.cnn import mini_XCEPTION,SE_XCEPTION, GAP_concate_XCEPTION
from utils.datasets import DataManager
from utils.datasets import split_data
from utils.preprocessor import preprocess_input

# parameters
batch_size = 16
num_epochs = 10000
input_shape = (64, 64, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 80
base_path = '../trained_models/emotion_models/'

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
model = GAP_concate_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


datasets = ['fer2013']
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    # saving model after one epoch finishing
    trained_models_path = base_path + dataset_name + '_GAP_concate_117k_XCEPTION'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.4f}.hdf5' 
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1,
                                      save_best_only=True,mode='auto',period=1)

    # view on internal states and statistics of the model during training
    # a set callbacks functions, and visualization by tensorboard
    log_file_path = base_path + dataset_name + '_emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_acc', factor=0.5,
                                  patience=int(patience/4), verbose=1)
    tensor_board = TensorBoard(log_dir='../log_dir',
                             histogram_freq=1,
                             write_graph=True,
                             write_images=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr, tensor_board]

    # loading dataset
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    faces, emotions = data_loader.get_data()
    faces = preprocess_input(faces)
    num_samples, num_classes = emotions.shape
    train_data, val_data = split_data(faces, emotions, validation_split)
    train_faces, train_emotions = train_data
    
    # Efficiency: generator run by paralle
    # Trains the model on data generated batch-by-batch by a Python generator
    model.fit_generator(data_generator.flow(train_faces, train_emotions,
                                            batch_size),
                        steps_per_epoch=len(train_faces) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=val_data)
