class stastics():
	def __init__(self):
		self.stastics = {}

	def add_stat(self, suffix, output_array):
		if suffix not in self.stastics:
			self.stastics[suffix] = {}

		stat = {}

		for output in output_array:
			if output['name'] not in stat:
				stat[output['name']] = {'count' : 0, 'confidence_sum' : 0, 'confidence' : 0}

			stat[output['name']]['count'] += 1
			stat[output['name']]['confidence_sum'] += output['percentage_probability']

			if output['name'] not in self.stastics[suffix]:
				self.stastics[suffix][output['name']] = {'frame_no' : 0, 'count_sum' : 0, 'conf_sum' : 0, 'conf_distribut' : [0] * 100}

			self.stastics[suffix][output['name']]['conf_distribut'][int(output['percentage_probability'] // 1)] += 1

		for item in stat:
			if stat[item]['count'] > 0:
				stat[item]['confidence'] = stat[item]['confidence_sum'] / stat[item]['count']

			self.stastics[suffix][item]['frame_no'] += 1
			self.stastics[suffix][item]['count_sum'] += stat[item]['count']
			self.stastics[suffix][item]['conf_sum'] += stat[item]['confidence_sum']

		return stat

	def get_result(self):
		return_text = ''

		for suffix in self.stastics:
			return_text += f'mode : {suffix}\n'
			return_text += f'item\tframe_no\tcount_sum\tcount_avg\tconf_avg\tconf_distribut\n'
			for item in self.stastics[suffix]:
				self.stastics[suffix][item]['count_avg'] = self.stastics[suffix][item]['count_sum'] / self.stastics[suffix][item]['frame_no']
				self.stastics[suffix][item]['conf_avg'] = self.stastics[suffix][item]['conf_sum'] / self.stastics[suffix][item]['count_sum']

				list = self.stastics[suffix][item]
				return_text += f"{item}\t{list['frame_no']}\t{list['count_sum']}\t{list['count_avg']}\t{list['conf_avg']}\t{list['conf_distribut']}\n"

			return_text += '\n'

		return return_text

class emotions():
	def __init__(self):
		self.emotions = {'frame_no' : 0, 'total_count' : 0, 'average_perframe' : 0}

	def add_frame(self):
		self.emotions['frame_no'] += 1

	def add_emotion(self, emotion):
		if emotion not in self.emotions:
			self.emotions[emotion] = 0

		self.emotions['total_count'] += 1
		self.emotions[emotion] += 1

	def get_result(self):
		self.emotions['average_perframe'] = self.emotions['total_count'] / self.emotions['frame_no']

		return str(self.emotions)
