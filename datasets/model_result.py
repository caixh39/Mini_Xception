""" saving model result
	concat_Xception net
"""

import os
import numpy as np
import pandas as pd
import csv

# concat_Xception result: name, time, memory, accuracy
network = ['AlexNet', 'BOVW+VGG(4x)', 
			'Xception', 'Mobile', 'Shuffle', 'Mobile_v2', 'Mobile_v2(96)','Mobile_v2(96,C)',
			'Concat_Mobile', 'Mini_xception', 'Concat_Xception', 'Concat_Xception(96)']

infer_time = [ 0, 0, 2000, 730, 597, 1000, 2000, 2000, 1000, 470, 832, 701]
accuracy = [71.20, 75.42, 68.61, 67.76, 65.67, 67.65, 69.24, 70.13, 69.57, 66.48, 70.13, 70.19]

# std=0.15
accuracy_std = [0, 0, 68.46, 67.61, 65.52, 67.50, 69.09, 69.98, 69.42, 66.33, 69.98, 70.04] 
F1_score = [0, 0, 68.53, 67.84, 65.60, 67.57, 69.11, 70.00, 69.41, 65.98, 70.04, 70.13]
memory = [20875248, 3235463, 973135, 2284615, 2284615, 
			6671671, 264679, 58423, 290719, 290719]

clos = pd.DataFrame({'network':network,
	                'Time':infer_time,
	                'Proability':accuracy,
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