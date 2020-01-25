## AI-Powered-Detection
Object detcetion & Facial emotion recognition

- **Detection processing(Object detection & Facial Emotion): [detection_processing/README.md](https://github.com/jinhojang6/ai-powered-detection/blob/master/detection_processing/README.md)**
    - Object detection: [COCO dataset](http://cocodataset.org/#home) and yolov3
        - download [yolov3-spp.weights](https://pjreddie.com/media/files/yolov3-spp.weights) and move the weights file to /detection_processing/yolov3-coco
        - yolov3-spp.weights is stronger in detection but slower in processing than [yolov3.weights](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg) 
    - Facial expression: follow the comments in yolo.py
        - download [yolov3-wider_16000.weights](https://drive.google.com/open?id=1n66gI61kilcsdWSHEHaSY0oNSDfWKBFp) and move the file to /detection_processing/yolov3-coco
    - Detection result:
        - Objection detection:
            <a href='https://photos.google.com/share/AF1QipNncknCcaQAhGxIN9Nb3IoHFFfuSg3cfg4MiX1ak43wczn5aoz3PStpQIA5RtSFDA?key=MGF6VmVUVHdMUjNaVmlla25ORC1TYl9vZGJvUTVB&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/UwGqDIfjcVX-TaeUQFr1aaofN5y9843z-3Myv4NnpmJq47Gb_V8eituG4gQhOa8uibJX9UWf2d3fsAsxKfrlrnlvcFOl6TXCNPOm5-wOdtg0gxQH2OMTfMgfRN047aGBTAEkn8QyVA=w2400' /></a>
            - https://youtu.be/I6TLHhydtc8
            - https://youtu.be/X-xy68YJ35o
            
        - Facial emotion recognition
             <a href='https://photos.google.com/share/AF1QipOYcQb-R5CAKhSYnxv6VRFc4wpsEvIUce7LdfeZCjRMLyEU5A6evwulmZs1We7-Ug?key=WGd3ZkEzRmFTTkVNY0o4NmpVak93M2RHYkE4enZ3&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/LcmtBuVjV0SAZrKnmLiShKWyWH3T9B6MwrqCS1Orvc3NYGHI-3JxKIzHOxWInA2sPULIXIA5fwcCMIb_CimhBZ_atyeIz9In4kJtbkOHXLKkja47d3S51CpgtRA6BpzaYiMGO2z3YQ=w2400' /></a>
            - https://youtu.be/HKW70fkzJUI
            - https://youtu.be/nu_I-H73cpE

<br/>

- **Facial emotion training : [emotion_training/README.md](https://github.com/jinhojang6/ai-powered-detection/blob/master/emotion_training/README.md)**
    - [fer2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data), Tensorflow and Keras
    - [Colab](https://github.com/jinhojang6/ai-powered-detection/blob/master/emotion_training/emotion_training.ipynb) : Tensorflow-gpu
    - Need to mount your Google drive on Colab to run [train_emotion_classifier.py](https://github.com/jinhojang6/ai-powered-detection/blob/master/emotion_training/src/train_emotion_classifier.py)


<br/>

- You can find the directory structures and weights files [here](http://bit.ly/keras-detection-practice)
