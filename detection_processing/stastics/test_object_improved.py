import argparse
import cv2 as cv
import numpy as np
import analysis_perframe as pfh

import sys
sys.path.append('..')
from yolo_utils import infer_image

def test_object_improved(path_in, path_out, suffix = 'object_improved'):
	parser = argparse.ArgumentParser()
	FLAGS, unparsed = parser.parse_known_args()

	FLAGS.model_path = '../yolov3-coco/'
	FLAGS.weights = '../yolov3-coco/yolov3-spp.weights'
	FLAGS.config = '../yolov3-coco/yolov3.cfg'
	FLAGS.video_path = path_in
	FLAGS.video_output_path = path_out + '_object_improved.avi'
	FLAGS.labels = '../yolov3-coco/coco-labels'
	FLAGS.confidence = 0.5
	FLAGS.threshold = 0.3
	FLAGS.download_model = False
	FLAGS.show_time = False

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
