import os
import analysis_stastics

def per_frame_handler(frame_number, output_array, output_count = None, returned_frame = None, suffix = 'object_default'):
	stat = analysis_stastics.stats.add_stat(suffix, output_array)

	report_path = f'../results/analysis_{suffix}.txt'
	file_mode = 'w'
	if os.path.isfile(report_path):
		with open(report_path, 'r') as file:
			if int(list(file)[-1].split(' ')[0]) < frame_number:
				file_mode = 'a'

	with open(report_path, file_mode) as file:
		file.write(f'{frame_number} : {stat}\n')

	# print(f'{frame_number} : {stat}')
