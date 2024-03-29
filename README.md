# AI-Powered-Detection
Practice objects detcetion & Facial emotion recognition with Keras

<br/>

## Inference
- **Detection processing (Object detection & Facial Emotion): read [detection_processing/README.md](https://github.com/jinhojang6/ai-powered-detection/blob/master/detection_processing/README.md)**

    - Check out [yolo.py](https://github.com/jinhojang6/ai-powered-detection/blob/master/detection_processing/yolo.py) to learn how to run this program. You will be able to choose between `1. Object dection` and `2. Facial expression` in your terminal.
    
    - Objects detections: [COCO dataset](http://cocodataset.org/#home) + YOLOv3
        - Download [yolov3-spp.weights](https://pjreddie.com/media/files/yolov3-spp.weights) and move the weights file to /detection_processing/yolov3-coco
        - yolov3-spp.weights is stronger in detection but slower in processing than [yolov3.weights](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
        
    - Facial expression: [emotion_model.hdf5](https://github.com/jinhojang6/ai-powered-detection/blob/master/detection_processing/models/emotion_model.hdf5) + [haarcascade_frontalface_default.xml](https://github.com/jinhojang6/ai-powered-detection/blob/master/detection_processing/models/haarcascade_frontalface_default.xml)
        - Download [yolov3-wider_16000.weights](https://drive.google.com/open?id=1n66gI61kilcsdWSHEHaSY0oNSDfWKBFp) for face detection and move the file to `/detection_processing/yolov3-coco`
        
    - Detection results:
        - Objects detection:
            <a href='https://photos.google.com/share/AF1QipNncknCcaQAhGxIN9Nb3IoHFFfuSg3cfg4MiX1ak43wczn5aoz3PStpQIA5RtSFDA?key=MGF6VmVUVHdMUjNaVmlla25ORC1TYl9vZGJvUTVB&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/UwGqDIfjcVX-TaeUQFr1aaofN5y9843z-3Myv4NnpmJq47Gb_V8eituG4gQhOa8uibJX9UWf2d3fsAsxKfrlrnlvcFOl6TXCNPOm5-wOdtg0gxQH2OMTfMgfRN047aGBTAEkn8QyVA=w2400' /></a>
            
        - Facial emotion recognition
             <a href='https://photos.google.com/share/AF1QipOYcQb-R5CAKhSYnxv6VRFc4wpsEvIUce7LdfeZCjRMLyEU5A6evwulmZs1We7-Ug?key=WGd3ZkEzRmFTTkVNY0o4NmpVak93M2RHYkE4enZ3&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/LcmtBuVjV0SAZrKnmLiShKWyWH3T9B6MwrqCS1Orvc3NYGHI-3JxKIzHOxWInA2sPULIXIA5fwcCMIb_CimhBZ_atyeIz9In4kJtbkOHXLKkja47d3S51CpgtRA6BpzaYiMGO2z3YQ=w2400' /></a>

<br/>

## Training
- **Facial emotion detection training : read [emotion_training/README.md](https://github.com/jinhojang6/ai-powered-detection/blob/master/emotion_training/README.md)**

    - [fer2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data), Tensorflow and Keras
    
    - [Colab](https://github.com/jinhojang6/ai-powered-detection/blob/master/emotion_training/emotion_training.ipynb) : Tensorflow-gpu
    
    - Need to mount your Google drive on Colab to run [train_emotion_classifier.py](https://github.com/jinhojang6/ai-powered-detection/blob/master/emotion_training/src/train_emotion_classifier.py)


<br/>

## Examples on Google Drive
- You can find the directory structures and weights files [here](http://bit.ly/keras-detection-practice)
