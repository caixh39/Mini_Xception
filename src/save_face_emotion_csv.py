import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import cv2
import csv
import time
import numpy as np
import pandas as pd
from statistics import mode
from keras.models import load_model
from pandas.io.formats.csvs import CSVFormatter

from utils.datasets import get_labels
from utils.datasets import appendDFToCSV_void
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import make_face_coordinates
from utils.preprocessor import preprocess_input

import Config as cfg
# parameters for loading data and images
# emotion_model_path = '../trained_models/emotion_models/fer2013_SE_medium_XCEPTION.101-0.67.hdf5'
emotion_model_path = '../trained_models/fer2013_mini_XCEPTION.107-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape 
frame_window = 10
emotion_offsets = (20, 40)
detect_fps_flag = 0
frame_count = 0
clos = {}
time = 0

# loading models 
face_detection = load_detection_model()
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture('/media/cuhksz/Database/Emotion_TD/child/MVI_1137.MOV')
# video_capture = cv2.VideoCapture(0)
# video_capture = cv2.VideoCapture('/home/medical/Users/Caixh/Database/HuYou/forward_1_a.mp4')

# getting the numbers of total image in video, int()
frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
# getting the image number in one second
fps = video_capture.get(cv2.CAP_PROP_FPS)

# ms
frame_time = 1/fps  

# Number of frames in the video filestart_time = time.time() 
# start_time = time.time() 
while True:
    ret, bgr_image = video_capture.read()
    if ret:
        time = time + frame_time
        frame_count += 1
        # print time/1000
    else:
        break

    # frame_time = time.time() - start_time  # Capture frame-by-frame
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    detected_faces, score, idx = detect_faces(face_detection, gray_image)

    detect_fps_flag = 0

    cfg.frame_face_count = -1

    for detected_face in detected_faces:
        if detect_fps_flag == 0:
            detect_fps_flag = 1

        cfg.frame_face_time = time
        cfg.frame_face_count = frame_count

        face_coordinates = make_face_coordinates(detected_face)
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        # print(y1, y2, x1, x2)

        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, emotion_target_size)
            # print gray_face # saving single frame, image size
        except:
            continue

        # reshape data type , fitting the model's input 
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        # model predict
        cfg.emotion_prediction = emotion_classifier.predict(gray_face)
        cfg.emotion_probability = np.max(cfg.emotion_prediction)
        emotion_label_arg = np.argmax(cfg.emotion_prediction) # max index 
        cfg.emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(cfg.emotion_text)
  
        if len(emotion_window) > frame_window:
            emotion_window.pop(0)

        try:
            emotion_mode = mode(emotion_window)
        except:
            continue
        
        # drawing box's color
        if cfg.emotion_text == 'angry':
            color = cfg.emotion_probability * np.asarray((255, 0, 0))
        elif cfg.emotion_text == 'sad':
            color = cfg.emotion_probability * np.asarray((0, 0, 255))
        elif cfg.emotion_text == 'happy':
            color = cfg.emotion_probability * np.asarray((255, 255, 0))
        elif cfg.emotion_text == 'surprise':
            color = cfg.emotion_probability * np.asarray((153, 17, 238))
        elif cfg.emotion_text == 'fear':
            color = cfg.emotion_probability * np.asarray((0, 255, 255))
        else:
            color = cfg.emotion_probability * np.asarray((0, 392255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)


    # saving data    
    if cfg.frame_face_count != -1:
        clos = pd.DataFrame({'Frame':cfg.frame_face_count,
                    'Time':round(cfg.frame_face_time,2),
                    'Prediction':cfg.emotion_text,
                    'Proability':round(cfg.emotion_probability, 2),
                     'angry':round(cfg.emotion_prediction[0][0], 2),
                     'disgust':round(cfg.emotion_prediction[0][1], 2),
                     'fear':round(cfg.emotion_prediction[0][2], 2),
                     'happy':round(cfg.emotion_prediction[0][3], 2),
                     'sad':round(cfg.emotion_prediction[0][4], 2),
                     'surprise':round(cfg.emotion_prediction[0][5], 2),
                     'neutral':round(cfg.emotion_prediction[0][6], 2)},
                      columns=cfg.EmotionLabels,
                      index=np.arange(1)) 
    else:
        cfg.frame_face_time = time
        clos = pd.DataFrame({'Frame':cfg.frame_face_count,
            'Time':round(cfg.frame_face_time,2),
            'Prediction':'-1',
            'Proability':-1,
             'angry':-1,
             'disgust':-1,
             'fear':-1,
             'happy':-1,
             'sad':-1,
             'surprise':-1,
             'neutral':-1},
              columns=cfg.EmotionLabels,
              index=np.arange(1)) 

    clos = appendDFToCSV_void(clos,'Emotion_MVI_1137.csv')            

    # pop window and close window
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

