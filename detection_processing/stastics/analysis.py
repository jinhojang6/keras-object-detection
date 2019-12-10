import os
import matplotlib.pyplot as plt
%matplotlib inline

from test_face_improved import test_face_improved
from test_object_default import test_object_default
from test_object_improved import test_object_improved
import analysis_stastics

execution_path = os.getcwd()
filename = 'PopeVisitToKorea.mp4'

words = filename.split('.')
filename_short = '.'.join(words[:len(words) - 1])

path_in = f'../videos/{filename}'
path_out = f'../results/{filename_short}'

analysis_stastics.stats = analysis_stastics.stastics()
analysis_stastics.emotions = analysis_stastics.emotions()
test_face_improved(path_in, path_out)
test_object_default(path_in, path_out)
test_object_improved(path_in, path_out)

avg_result = analysis_stastics.stats.get_avg()
result_out = f'face_emotion : {analysis_stastics.emotions.get_result()}\n'
for mode in avg_result:
	result_out += f'{mode} : {avg_result[mode]}\n'
with open('../results/result_avg.txt', 'w') as file:
	file.write(result_out)
