## AI-Powered-Detection
Object detcetion & Facial emotion recognition

- Detection processing(Object detection & Facial Emotion): [detection_processing/README.md](https://github.com/jinhojang6/ai-powered-detection/blob/master/detection_processing/README.md)
    - Object detection: [COCO dataset](http://cocodataset.org/#home) and yolov3
        - download [yolov3-spp.weights](https://pjreddie.com/media/files/yolov3-spp.weights) and move the weights file to /detection_processing/yolov3-coco
        - yolov3-spp.weights is stronger in detection but slower in processing than [yolov3.weights](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg) 
    - Facial expression: follow the comments in yolo.py
        - download [yolov3-wider_16000.weights](https://drive.google.com/open?id=1n66gI61kilcsdWSHEHaSY0oNSDfWKBFp) and move the file to /detection_processing/yolov3-coco


- Facial emotion training : [emotion_training/README.md](https://github.com/jinhojang6/ai-powered-detection/blob/master/emotion_training/README.md)
    - [fer2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data), Tensorflow and Keras
    - [Colab](https://github.com/jinhojang6/ai-powered-detection/blob/master/emotion_training/emotion_training.ipynb) : Tensorflow-gpu
    - Need to mount your Google drive on Colab to run `train_emotion_classifier.py`


- Find directory structures and weights files [here](http://bit.ly/keras-detection-practice)