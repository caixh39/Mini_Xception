import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import cv2
import csv
import time
import numpy as np
import pandas as pd

# file path
video_path = '/media/cuhksz/Database/Emotion_TD/child/MVI_1133.MOV'

# parameters
time = 0

# loading video
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(video_path)

# getting the numbers of total image in video, int()
frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
# print frames

# getting the image number in one second
fps = video_capture.get(cv2.CAP_PROP_FPS)
# print fps

# ms
frame_time = 1/fps

# getting video's time: second
# print frames/fps

while True:
	ret, bgr_image = video_capture.read()
	if ret:
		time = time + frame_time
		print time
	else:
		break

# frame_count = 0
# while True:
#     bgr_image = video_capture.read()[1]
#     frame_count += 1
#     frame_time = time.time() - start_time 


#    if detect_fps_flag == 1:
#        print('time: ' + str(1. / ((time.time() - start))) + ' fps')  
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break
