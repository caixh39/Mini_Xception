Facial Emotion Recognition 
--- 
#### Neural Network: mini_Xception
- base on Xception architecture, detecting multi scale info from image 

#### Face detect using Dlib library 
- Hog for face dectection

#### using
- $ python video_emotion_color_demo.py 

#### Defict:
- facial emotion recognition's accuracy is only 65.87% when reruning the model, via dataset is Train and Public datas of fer2013.

#### Remend: 
- Modify the architectures of cnn model to improve accuracy


#### Problem:
- keras.preprocessing.image.load_img(image_path, grayscale,target_size) 
- open ../python2.7/dist_packages/keras_preprocessing.image.py 
- comment " 'load_img' function else color mode"
