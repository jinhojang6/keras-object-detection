import argparse
import cv2 as cv
import numpy as np
import analysis_perframe as pfh
from keras.models import load_model

import sys
sys.path.append('..')
from yolo_utils import infer_image
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from statistics import mode

def test_face_improved(path_in, path_out, suffix = 'face_improved'):
	parser = argparse.ArgumentParser()
	FLAGS, unparsed = parser.parse_known_args()

	FLAGS.model_path = '../yolov3-coco/'
	FLAGS.weights = '../yolov3-coco/yolov3-wider_16000.weights'
	FLAGS.config = '../yolov3-coco/yolov3-face.cfg'
	FLAGS.video_path = path_in
	FLAGS.video_output_path = path_out + '_face_improved.avi'
	FLAGS.labels = '../yolov3-coco/coco-labels'
	FLAGS.confidence = 0.5
	FLAGS.threshold = 0.3
	FLAGS.download_model = False
	FLAGS.show_time = False

	emotion_model_path = '../models/emotion_model.hdf5'
	emotion_classifier = load_model(emotion_model_path)
	emotion_target_size = emotion_classifier.input_shape[1:3]
	emotion_labels = get_labels('fer2013')
	emotion_offsets = (20, 40)
	emotion_window = []
	frame_window = 10
	face_cascade = cv.CascadeClassifier('../models/haarcascade_frontalface_default.xml')

	vid = cv.VideoCapture(FLAGS.video_path)
	height, width, writer = None, None, None

	labels = open(FLAGS.labels).read().strip().split('\n')
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)
	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	frame_number = 0
	while True:
		grabbed, frame = vid.read()

		if not grabbed:
			break
		else:
			frame_number += 1

		if width is None or height is None:
			height, width = frame.shape[:2]

		img, boxes, confidences, classids, idxs = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS)

		gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)

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

		img = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)

		output_array = []
		for index in range(len(classids)):
			output_array.append({'name' : labels[classids[index]], 'percentage_probability' : confidences[index] * 100})

		pfh.per_frame_handler(frame_number, output_array, suffix = suffix)

		if writer is None:
			fourcc = cv.VideoWriter_fourcc(*"MJPG")
			writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

		writer.write(frame)

	writer.release()
	vid.release()
