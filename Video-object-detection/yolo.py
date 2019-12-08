import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from yolo_utils import infer_image, show_image
from keras.models import load_model

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from statistics import mode

FLAGS = []

#face detection labels, refer to coco-labels for objection
emotion_model_path = './models/emotion_model.hdf5'
emotion_classifier = load_model(emotion_model_path)
emotion_target_size = emotion_classifier.input_shape[1:3]
emotion_labels = get_labels('fer2013')
emotion_offsets = (20, 40)
emotion_window = []
frame_window = 10
face_cascade = cv.CascadeClassifier('./models/haarcascade_frontalface_default.xml')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-m', '--model-path',
		type=str,
		default='./yolov3-coco/',
		help='The directory where the model weights and \
			  configuration files are.')

	# 1. Object detection

	# # if you want to use a lighter version: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg &
	# # and use this cfg: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
	# # yolov3-spp is stronger than yolov3. Don't foget to comment out line 171-223
	# parser.add_argument('-w', '--weights',
	# 	type=str,	
	# 	default='./yolov3-coco/yolov3-spp.weights',
	# 	help='Path to the file which contains the weights \
	# 		 	for YOLOv3.')

	# # object detection cfg
	# parser.add_argument('-cfg', '--config',
	# 	type=str,
	# 	default='./yolov3-coco/yolov3.cfg',
	# 	help='Path to the configuration file for the YOLOv3 model.')


	# 2. Facial expressions

	# face detection weights: https://drive.google.com/file/d/1xYasjU52whXMLT5MtF7RCPQkV66993oR/view 
	parser.add_argument('-w', '--weights',
		type=str,	
		default='./yolov3-coco/yolov3-wider_16000.weights',
		help='Path to the file which contains the weights \
			 	for YOLOv3.')

	# facial expression cfg
	parser.add_argument('-cfg', '--config',
		type=str,
		default='./yolov3-coco/yolov3-face.cfg',
		help='Path to the configuration file for the YOLOv3 model.')

	parser.add_argument('-i', '--image-path',
		type=str,
		help='The path to the image file')

	parser.add_argument('-v', '--video-path',
		type=str,
		help='The path to the video file')


	parser.add_argument('-vo', '--video-output-path',
		type=str,
        default='./output.avi',
		help='The path of the output video file')

	parser.add_argument('-l', '--labels',
		type=str,
		default='./yolov3-coco/coco-labels',
		help='Path to the file having the \
					labels in a new-line seperated way.')

	parser.add_argument('-c', '--confidence',
		type=float,
		default=0.5,
		help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

	parser.add_argument('-th', '--threshold',
		type=float,
		default=0.3,
		help='The threshold to use when applying the \
				Non-Max Suppresion')

	parser.add_argument('--download-model',
		type=bool,
		default=False,
		help='Set to True, if the model weights and configurations \
				are not present on your local machine.')

	parser.add_argument('-t', '--show-time',
		type=bool,
		default=False,
		help='Show the time taken to infer each image.')

	FLAGS, unparsed = parser.parse_known_args()

	# Download the YOLOv3 models if needed
	if FLAGS.download_model:
		subprocess.call(['./yolov3-coco/get_model.sh'])

	# # Get the object detection labels
	labels = open(FLAGS.labels).read().strip().split('\n')

	# Intializing colors to represent each label uniquely
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	# Load the weights and configutation to form the pretrained YOLOv3 model
	net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

	# Get the output layer names of the model
	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
	# If both image and video files are given then raise error
	if FLAGS.image_path is None and FLAGS.video_path is None:
	    print ('Neither path to an image or path to video provided')
	    print ('Starting Inference on Webcam')

	# Do inference with given image
	if FLAGS.image_path:
		# Read the image
		try:
			img = cv.imread(FLAGS.image_path)
			height, width = img.shape[:2]
		except:
			raise 'Image cannot be loaded!\n\
                               Please check the path provided!'

		finally:
			img, _, _, _, _ = infer_image(net, layer_names, height, width, img, colors, labels, FLAGS)
			show_image(img)


	elif FLAGS.video_path:
		# Read the video
		try:
			cap = cv.VideoCapture(FLAGS.video_path)
			height, width = None, None
			writer = None
		except:
			raise 'Video cannot be loaded!\n\
                               Please check the path provided!'

		finally:
			while True:
				ret, bgr_image = cap.read()

			    # Checking if the complete video is read
				if not ret:
					break

				if width is None or height is None:
					height, width = bgr_image.shape[:2]

				# inference => face or object, refer to line 41 and 49
				bgr_image, _, _, _, _ = infer_image(net, layer_names, height, width, bgr_image, colors, labels, FLAGS)


				# Start of facial expression, comment out this section if you want object detection
				gray_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)
				rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)

				faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
						minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)

				for face_coordinates in faces:

					x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
					gray_face = gray_image[y1:y2, x1:x2]
					try:
						gray_face = cv.resize(gray_face, (emotion_target_size))
					except:
						continue

					gray_face = preprocess_input(gray_face, True)
					gray_face = np.expand_dims(gray_face, 0)
					gray_face = np.expand_dims(gray_face, -1)
					emotion_prediction = emotion_classifier.predict(gray_face)
					emotion_probability = np.max(emotion_prediction)
					emotion_label_arg = np.argmax(emotion_prediction)
					emotion_text = emotion_labels[emotion_label_arg]
					emotion_window.append(emotion_text)

					if len(emotion_window) > frame_window:
						emotion_window.pop(0)
					try:
						emotion_mode = mode(emotion_window)
					except:
						continue

					if emotion_text == 'angry':
						color = emotion_probability * np.asarray((255, 0, 0))
					elif emotion_text == 'sad':
						color = emotion_probability * np.asarray((0, 0, 255))
					elif emotion_text == 'happy':
						color = emotion_probability * np.asarray((255, 255, 0))
					elif emotion_text == 'surprise':
						color = emotion_probability * np.asarray((0, 255, 255))
					else:
						color = emotion_probability * np.asarray((0, 255, 0))

					color = color.astype(int)
					color = color.tolist()

					draw_text(face_coordinates, rgb_image, emotion_mode,
							color, 0, -45, 1, 1)

				bgr_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)

				# End of facial expression

				if writer is None:
					# Initialize the video writer
					fourcc = cv.VideoWriter_fourcc(*"MJPG")
					writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30, 
						            (bgr_image.shape[1], bgr_image.shape[0]), True)



				writer.write(bgr_image)

			print ("[INFO] Cleaning up...")
			writer.release()
			cap.release()


	else:
		# Infer real-time on webcam
		count = 0

		vid = cv.VideoCapture(0)
		while True:
			_, frame = vid.read()
			height, width = frame.shape[:2]

			if count == 0:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
		    						height, width, frame, colors, labels, FLAGS)
				count += 1
			else:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
		    						height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
				count = (count + 1) % 6

			cv.imshow('webcam', frame)

			if cv.waitKey(1) & 0xFF == ord('q'):
				break
		vid.release()
		cv.destroyAllWindows()
