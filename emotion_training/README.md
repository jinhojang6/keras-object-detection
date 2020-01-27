## Credit

- petercunha: https://github.com/petercunha/Emotion.git


## Installation

Clone this repository:
```
git clone https://github.com/jinhojang6/ai-powered-detection.git
cd ai-powered-detection/emotion_training
```

Install all the dependencies with `pip3 install <module name>`

```
#use tensorflow or tensorflow-gpu
pip3 install tensorflow-gpu==2.0 numpy==1.17.4 scipy==1.1.0 opencv-python==4.1.2.30 pillow==6.2.1 pandas==0.25.3 matplotlib==3.1.2 h5py==2.10.0 keras==2.3.1

#Colab
pip3 install tensorflow-gpu==2.0
pip3 install numpy==1.17.4
pip3 install scipy==1.1.0
pip3 install opencv-python==4.1.2.30
pip3 install pillow==6.2.1
pip3 install pandas==0.25.3
pip3 install matplotlib==3.1.2
pip3 install h5py==2.10.0
pip3 install keras==2.3.1
```

or use `pip3 install tensorflow-gpu==2.0 numpy==1.17.4 ...`


## Run

Once the dependencies are installed successfully, go to the directory where emotions.py is located and
`python3 emotions.py`


## Training a model with Colab

- Upload `emotion_training.ipynb` on Colab
- Find an example of a directory structure at : http://bit.ly/keras-detection-practice
- Download datasets in /datasets (There is a filder named fer2013). A cloned repository doesn't include the datasets
- Upload directory structures as the link above on your Google Drive.
- To mount your Google Drive files containing the dataset, run `emotion_training.ipynb` and follow all the instructions at
```
from google.colab import drive
drive.mount('/content/drive')
```
- `!cd drive/My\ Drive/AI/emotion_training/src && python3 train_emotion_classifier.py` will kick off a training (4 epochs by default) and it creates a brand new model in `trained_models/emotion_models` (.hdf5 file extension)
- Move the new model created to src/models/ in your cloned repository
- In `emotions.py`, replace `emotion_model_path = './models/emotion_model.hdf5` with the new model file
- Run `python3 emotions.py` and you will find a different result from the original model
- Find 2 examples in /processed_videos


## Training new models for emotion classification from scratch

- Download the fer2013.tar.gz file from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- Move the downloaded file to the datasets directory inside this repository.
- Untar the file:
`tar -xzf fer2013.tar`
- Download train_emotion_classifier.py from orriaga's repo [here](https://github.com/oarriaga/face_classification/blob/master/src/train_emotion_classifier.py)
- Run the train_emotion_classification.py file:
`python3 train_emotion_classifier.py`


## Deep Learning Model

The model used is from this [research paper](https://github.com/oarriaga/face_classification/blob/master/report.pdf) written by Octavio Arriaga, Paul G. Plöger, and Matias Valdenegro.
