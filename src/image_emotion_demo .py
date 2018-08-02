import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import sys
import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import load_detection_model
from utils.inference import make_face_coordinates
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_image
from utils.preprocessor import preprocess_input

# parameters for loading data and images
# image_path = sys.argv[1]
image_path = '/home/cuhksz/Pictures/image/fear_change.png'
image_name = image_path[28:-4]
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

# loading models
face_detection = load_detection_model()
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# loading images
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')

# faces = detect_faces(face_detection, gray_image)
# for face_coordinates in faces:
detect_faces, score, idx = detect_faces(face_detection, gray_image)
for detect_face in detect_faces:
    face_coordinates = make_face_coordinates(detect_face)

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
    emotion_text = emotion_labels[emotion_label_arg]


    EmotionLabels = [ 'angry', 'disgust', 'fear',
                    'happy', 'sad', 'surprise', 'neutral' ]
    print 'result:', emotion_text, emotion_probability
    print EmotionLabels[0], ':', emotion_prediction[0][0]
    print EmotionLabels[1], ':',emotion_prediction[0][1]
    print EmotionLabels[2], ':',emotion_prediction[0][2]
    print EmotionLabels[3], ':',emotion_prediction[0][3]
    print EmotionLabels[4], ':',emotion_prediction[0][4]
    print EmotionLabels[5], ':',emotion_prediction[0][5]
    print EmotionLabels[6], ':',emotion_prediction[0][6]

    
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


    emotion_text = emotion_text + '=' + str(emotion_probability)

    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2) 

bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('../images/'+image_name+'.png', bgr_image)
