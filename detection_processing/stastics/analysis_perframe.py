import os
import analysis_stastics

def per_frame_handler(frame_number, output_array, output_count = None, returned_frame = None, suffix = 'default'):
	stat = {}

	for output in output_array:
		if output['name'] not in stat:
			stat[output['name']] = {'count' : 0, 'confidence_sum' : 0, 'confidence' : 0}

		stat[output['name']]['count'] += 1
		stat[output['name']]['confidence_sum'] += output['percentage_probability']

	for item in stat:
		if stat[item]['count'] > 0:
			stat[item]['confidence'] = stat[item]['confidence_sum'] / stat[item]['count']

	analysis_stastics.stats.add_stat(suffix, stat)

	report_path = f'../results/analysis_{suffix}.txt'
	file_mode = 'w'
	if os.path.isfile(report_path):
		with open(report_path, 'r') as file:
			if int(list(file)[-1].split(' ')[0]) < frame_number:
				file_mode = 'a'

	with open(report_path, file_mode) as file:
		file.write(f'{frame_number} : {stat}\n')

	print(f'{frame_number} : {stat}')
