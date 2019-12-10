class stastics():
	def __init__(self):
		self.stastics = {}

	def add_stat(self, suffix, stat):
		if suffix not in self.stastics:
			self.stastics[suffix] = {}

		for item in stat:
			if item not in self.stastics[suffix]:
				self.stastics[suffix][item] = {'frame_no' : 0, 'count_sum' : 0, 'confidence_sum' : 0, 'count_avg' : 0, 'confidence_avg' : 0}

			self.stastics[suffix][item]['frame_no'] += 1
			self.stastics[suffix][item]['count_sum'] += stat[item]['count']
			self.stastics[suffix][item]['confidence_sum'] += stat[item]['confidence_sum']

	def get_avg(self):
		for suffix in self.stastics:
			for item in self.stastics[suffix]:
				self.stastics[suffix][item]['count_avg'] = self.stastics[suffix][item]['count_sum'] / self.stastics[suffix][item]['frame_no']
				self.stastics[suffix][item]['confidence_avg'] = self.stastics[suffix][item]['confidence_sum'] / self.stastics[suffix][item]['count_sum']

		return self.stastics

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
