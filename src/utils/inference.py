import cv2
import dlib
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.preprocessing import image



def load_image(image_path, grayscale=False, target_size=None):

    # image = Image.open(image_path)
    # image_array = np.asarray(image, dtype=np.float32)
    # image = img_to_array(image_array) 

    # pre-process the image for classification
    pil_image = image.load_img(image_path, grayscale,target_size)

    return image.img_to_array(pil_image)  # image


# def load_detection_model(model_path):
#     detection_model = cv2.CascadeClassifier(model_path)
#     return detection_model

def load_detection_model():
    return dlib.get_frontal_face_detector()


def detect_faces(detection_model, gray_image_array):
    # return detection_model.detectMultiScale(gray_image_array, 1.3, 5)
    return detection_model.run(gray_image_array, 0, 0)


def make_face_coordinates(detected_face):
    x = detected_face.left()
    y = detected_face.top()
    width = detected_face.right() - detected_face.left()
    height = detected_face.bottom() - detected_face.top()
    return [x, y, width, height]


def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    new_x = x - x_off
    if new_x < 0:
        new_x = 5
    new_y = y - y_off
    if new_y < 0:
        new_y = 5
    return (new_x, x + width + x_off, new_y, y + height + y_off)


def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
              font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def draw_text_top3(coordinates, image_array, text_1, text_2, text_3, color, x_offset=0, y_offset=0,
              font_scale=0.5, thickness=2):
    x, y = coordinates[:2]
    image_text = cv2.putText(image_array, text_1, (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    image_text = cv2.putText(image_text, text_2, (5, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    image_text = cv2.putText(image_text, text_3, (5, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    return image_text


def get_colors(num_classes):
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = np.asarray(colors) * 255
    return colors
