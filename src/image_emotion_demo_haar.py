import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import sys
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model

from utils.datasets import get_labels
from utils.inference import draw_text_top3
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets 
from utils.inference import load_image
from utils.preprocessor import preprocess_input


def load_detection_model(detection_model_path):
    detection_model = cv2.CascadeClassifier(detection_model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

# parameters for loading data and images
# image_path = sys.argv[1]
image_path = '/home/cuhksz/Pictures/image/fear/fear9.png'
image_name = image_path[33:-4]

detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]


# loading images
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')

faces = detect_faces(face_detection, gray_image)
for face_coordinates in faces:

    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    gray_face = gray_image[y1:y2, x1:x2]

    try:
        gray_face = cv2.resize(gray_face, (emotion_target_size))
    except:
        continue

    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_prediction = emotion_classifier.predict(gray_face)
    emotion_probability = np.max(emotion_prediction)  
    emotion_label_arg = np.argmax(emotion_prediction)
    probability_order = sorted(np.squeeze(emotion_prediction))
    emotion_text = emotion_labels[emotion_label_arg]

    # print np.squeeze(emotion_prediction), probability_order

    EmotionLabels = [ 'angry', 'disgust', 'fear',
                    'happy', 'sad', 'surprise', 'neutral' ]

    clos = {'angry':emotion_prediction[0][0], 
            'disgust':emotion_prediction[0][1], 
            'fear':emotion_prediction[0][2],
            'happy':emotion_prediction[0][3], 
            'sad':emotion_prediction[0][4], 
            'surprise':emotion_prediction[0][5], 
            'neutral':emotion_prediction[0][6]}
    # python's values and key convert
    clos_conv = {v:k for k,v in clos.items()} 
    print clos
    print emotion_text, emotion_probability
    print clos_conv[probability_order[-2]], probability_order[-2]
    print clos_conv[probability_order[-3]], probability_order[-3]

    # print 'result:', emotion_text, emotion_probability, probability_order[-2]
    # print EmotionLabels[0], ':', emotion_prediction[0][0]
    # print EmotionLabels[1], ':',emotion_prediction[0][1]
    # print EmotionLabels[2], ':',emotion_prediction[0][2]
    # print EmotionLabels[3], ':',emotion_prediction[0][3]
    # print EmotionLabels[4], ':',emotion_prediction[0][4]
    # print EmotionLabels[5], ':',emotion_prediction[0][5]
    # print EmotionLabels[6], ':',emotion_prediction[0][6]

    
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


    emotion_top1 = emotion_text + '=' + str(emotion_probability)
    emotion_top2 = clos_conv[probability_order[-2]] + '=' + str(probability_order[-2])
    emotion_top3 = clos_conv[probability_order[-3]] + '=' + str(probability_order[-3])
    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text_top3(face_coordinates, rgb_image, emotion_top1, emotion_top2, emotion_top3, color, 0, -50, 0.8, 2)



bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('../images/fear/'+ image_name + '_' + 'haar.png', bgr_image)
