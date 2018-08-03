import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from statistics import mode
import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text,draw_text_top3
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import make_face_coordinates
from utils.preprocessor import preprocess_input
import time
# parameters for loading data and images
# detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
# face_detection = load_detection_model(detection_model_path)
face_detection = load_detection_model()
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
# video_capture = cv2.VideoCapture(0)
video_capture = cv2.VideoCapture('/media/cuhksz/Database/Emotion_TD/child/MVI_1138.MOV')

detect_fps_flag = 0
start = 0.
while True:
    bgr_image = video_capture.read()[1]

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    # faces = detect_faces(face_detection, gray_image)
    detected_faces, score, idx = detect_faces(face_detection, gray_image)
    # for face_coordinates in faces:
    detect_fps_flag = 0
    for detected_face in detected_faces:
        # print the algorithms frame/second
        if detect_fps_flag == 0:
            start = time.time()
            detect_fps_flag = 1
        face_coordinates = make_face_coordinates(detected_face)
        # print face_coordinates[:2]
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        # print(y1, y2, x1, x2)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, emotion_target_size)
        except:
            # continue
            # print('error on resize')
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        probability_order = sorted(np.squeeze(emotion_prediction))
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        clos = {'angry':emotion_prediction[0][0], 
            'disgust':emotion_prediction[0][1], 
            'fear':emotion_prediction[0][2],
            'happy':emotion_prediction[0][3], 
            'sad':emotion_prediction[0][4], 
            'surprise':emotion_prediction[0][5], 
            'neutral':emotion_prediction[0][6]}
        # python's values and key convert
        clos_conv = {v:k for k,v in clos.items()} 
        
        # print clos
        # print emotion_text, emotion_probability
        # print clos_conv[probability_order[-2]], probability_order[-2]
        # print clos_conv[probability_order[-3]], probability_order[-3]


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
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 392255, 0))

        color = color.astype(int)
        color = color.tolist()

        emotion_top1 = emotion_text + '=' + str(emotion_probability)
        emotion_top2 = clos_conv[probability_order[-2]] + '=' + str(probability_order[-2])
        emotion_top3 = clos_conv[probability_order[-3]] + '=' + str(probability_order[-3])

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text_top3(face_coordinates, rgb_image, emotion_top1, 
                        emotion_top2, emotion_top3, color, 0, -50, 0.8, 2)

        # draw_text is max(emotion_probability)
        # draw_text(face_coordinates, rgb_image, emotion_mode,
                  # color, 0, -45, 1, 1)

    # print the algorithms frame/second
    # if detect_fps_flag == 1:
    #     print('time: ' + str(1. / ((time.time() - start))) + ' fps')

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
