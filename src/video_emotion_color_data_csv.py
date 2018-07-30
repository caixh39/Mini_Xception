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

# parameters for loading data and images
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model()
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []


# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture('/media/cuhksz/Database/Emotion_Samples/child/MVI_1138.MOV')
# video_capture = cv2.VideoCapture(0)
#video_capture = cv2.VideoCapture('/home/medical/Users/Caixh/Database/HuYou/forward_1_a.mp4')

# Number of frames in the video file
fps = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) 
print fps

detect_fps_flag = 0
EmotionData = []
clos = {}
# feelings_faces = []
# for index, emotion in enumerate(emotion_labels):
#     feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))

while True:
    bgr_image = video_capture.read()[1]   # Capture frame-by-frame
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    detected_faces, score, idx = detect_faces(face_detection, gray_image)

    detect_fps_flag = 0
    for detected_face in detected_faces:
        if detect_fps_flag == 0:
            detect_fps_flag = 1

        face_coordinates = make_face_coordinates(detected_face)
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        # print(y1, y2, x1, x2)

        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, emotion_target_size)
            # print gray_face # saving single frame, image size
        except:
            continue


        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        # saving multiple emotion_prediction (7 emotion) base on one image
        # emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
        # print emotion_prediction[0][1]

        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction) # max index 
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        # # print emotion_probability, emotion_text # type(emotion_probability), type(emotion_text)
        EmotionLabels = ['Prediction', 'Proability', 'angry', 'disgust', 'fear',
                        'happy', 'sad', 'surprise', 'neutral' ]
        # clos = pd.DataFrame(columns = EmotionLabels)
        # EmotionData.append(emotion_text)
        # EmotionData.append(emotion_probability)
        # EmotionData.append(emotion_prediction[0][0])
        # EmotionData.append(emotion_prediction[0][1])
        # EmotionData.append(emotion_prediction[0][2])
        # EmotionData.append(emotion_prediction[0][3])
        # EmotionData.append(emotion_prediction[0][4])
        # EmotionData.append(emotion_prediction[0][5])
        # EmotionData.append(emotion_prediction[0][6])
        # EmotionData.append('/n')
        # print EmotionData

        print emotion_text, emotion_probability, emotion_prediction[0][0],emotion_prediction[0][1]
        print emotion_prediction[0][2],emotion_prediction[0][3],emotion_prediction[0][4]
        print emotion_prediction[0][5],emotion_prediction[0][6]

        if detect_fps_flag == 1:
            clos = pd.DataFrame({'Prediction':emotion_text,
                            'Proability':emotion_probability,
                             'angry':emotion_prediction[0][0], 
                             'disgust':emotion_prediction[0][1], 
                             'fear':emotion_prediction[0][2],
                             'happy':emotion_prediction[0][3], 
                             'sad':emotion_prediction[0][4], 
                             'surprise':emotion_prediction[0][5], 
                             'neutral':emotion_prediction[0][6]},
                              columns=EmotionLabels,
                              index=np.arange(1)) 
            # clos.to_csv('Emotion.csv')
        # with open('Emotion.csv','a') as f:
        #     clos.to_csv(f,header=False)
        # clos.to_csv('Emotion_1.csv', mode='a', header=False)                 
        # for i in range(0,fps):
        # for id in range(0,6):
        #     clos.iloc[i,[id]] = EmotionData[0][id]

        clos = appendDFToCSV_void(clos,'Emotion_2.csv')

	
        if len(emotion_window) > frame_window:
            emotion_window.pop(0)

        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((153, 17, 238))
        elif emotion_text == 'fear':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 392255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)


#    if detect_fps_flag == 1:
#        print('time: ' + str(1. / ((time.time() - start))) + ' fps')        
        
    # if detect_fps_flag == 1:         
	   # EmotionData = {'Emotion': emotion_text, 
			 #         'Proability': emotion_probability}
	   # VideoTime = np.arange(fps)
	   # EmotionData = pd.DataFrame(EmotionData, columns=['Emotion', 'Proability'],index=VideoTime)
	   # print EmotionData
	   # EmotionData.to_csv('emotion.csv', sep='\t', encoding='utf-8', index=False)
    # clos.append(clos)  
    # clos.to_csv('Emotion.csv')

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

