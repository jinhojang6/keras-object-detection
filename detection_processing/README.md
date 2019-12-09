## Credit

- iArunava : https://github.com/iArunava/YOLOv3-Object-Detection-with-OpenCV

## Installation
```
# recommend using tensorflow=2.0 or tensorflow-gpu=2.0
pip3 install tensorflow-gpu==2.0 numpy==1.17.4 scipy==1.1.0 opencv-python==4.1.2.30 pillow==6.2.1 pandas==0.25.3 matplotlib==3.1.2 h5py==2.10.0 keras==2.3.1

```


## How to use?

1) Clone the repository and download [weights files](http://bit.ly/keras-detection-practice)

```
https://github.com/jinhojang6/ai-powered-detection.git
```

2) Move to the directory and move the weights file to /yolov3-coco
```
cd ai-powered-detection
```

3) To infer on an image that is stored on your local machine
```
python3 yolo.py --image-path='/path/to/image/'
```
4) To infer on a video that is stored on your local machine
```
python3 yolo.py --video-path='/path/to/video/'
```
5) To infer real-time on webcam
```
python3 yolo.py
```

Note: This works considering you have the `weights` and `config` files at the yolov3-coco directory.
<br/>
If the files are located somewhere else then mention the path while calling the `yolov3.py`. For more details
```
yolo.py --help
```

## Example
```
#Choose python or python3 of your choice
Mac or Linux: python3 yolo.py --video-path='./test.mp4'
Windows: python3 yolo.py --video-path='.\test.mp4'

You will find `output.avi` in the project folder (Could take 2-5 minutes to process 10 seconds video using CPU)
```


## References

1) [PyImageSearch YOLOv3 Object Detection with OpenCV Blog](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)

## License

The code in this project is distributed under the MIT License.
