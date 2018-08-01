#author:David Wong
import numpy as np
import cv2

class KalmanFilter:
    def __init__(self):
        self.last_measurment = np.array((2,1),np.float32)
        self.current_measurment = np.array((2, 1), np.float32)

        self.last_prediction = np.zeros((2,1),np.float32)
        self.current_prediction = np.zeros((2, 1), np.float32)

        self.kalman = cv2.KalmanFilter(4,2)

        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)*0.03
        self.kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.5

    def predict(self,x,y):

        self.last_prediction = self.current_prediction
        self.last_measurment = self.current_measurment

        self.current_measurment = np.array([[np.float32(x)],[np.float32(y)]]) #prediction

        self.kalman.correct(self.current_measurment)

        self.current_prediction = self.kalman.predict()

        return np.int(self.current_prediction[0]),np.int(self.current_prediction[1])