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

## Detail explain:
### Facial Emotion Recognition timely and quickly
#### REQUIREMENTS.txt: 表情识别算法所需的算法环境
#### src:
 - ./train_emotion_classifier_mul_gpu.py: 模型训练
 - ./video_emotion_color_demo.py: 摄像头进行实时人脸表情识别测试
 - ./video_emotion_color_data_csv.py: 保存实时人脸表情识别测试的情绪数据信息
 - ./model_evaluate.py: 对训练好的模型进行fer2013 private datasets 进行测试
 - ./confusion_matrix.py: 生成测试集的混淆矩阵及其它量化指标
 - ./image_emotion_demo_haar.py: 对单张图像进行情绪识别（opencv-haar 人脸检测算法）
 - ./image_emotion_demo_dlib.py: 对单张图像进行情绪识别（dlib-hog 人脸检测算法）
 - ./feature_visual.py: 对模型学习特征的过程进行可视化
 - ./save_emotion_info_csv.py: 对小孩的情绪视频，进行表情识别并保存对应数据
 - ./Config.py: 全局配置参数
 - utlis:
   - ./convert_fer2013.py: 将fer2013.csv 划分训练集/验证集/测试集
   - ./data_augmentation.py:
   - ./datasets.py
   - ./inference.py
   - ./preprocessor.py
   - ./utils.py
 - models: 模型结构
   - ./cnn.py: mini_XCEPTION + mini_concate_XCEPTION_V1 + mini_concate_XCEPTION_V2 + mini_concate_XCEPTION_V3
   - ./compare_cnn.py: XceptionNet + MobileNet + InceptionV3 + InceptionResNetV2
   - ./Module_Net.py: SE model + conv2d_bn + sep_conv2d_bn + DepthwiseConv2D 常用网络模块的代码实现
#### datasets:
 - 表情识别算法进行模型训练的数据集：fer2013, CK+
 - fer2013: 将fer2013按 training dataset, public dataset划分为训练集 test.csv, private dataset为 test.csv
 - 情绪数值记录：用之前采集表情数据的视频进行模拟表情数据值的提取
#### images:
 - confusion_matrix: 训练好的模型在测试集预测结果的混淆矩阵
 - models: 模型的结构图
 - visualize_filter: 图片送入模型，在不同卷积层的特征图 (mini_Xception 模型测试）
#### trained_models:
 - detection_models: opencv-haar 人脸检测的配置文件
 - emotion_models: 模型训练过程，生成的模型参数文件
 - test_models: 模型训练得到的最好的结果
