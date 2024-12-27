import pandas as pd
import numpy as np


class Stats:
	def __init__(self, path, model, round):
		self.path = path
		self.model = model
		self.round = round
		# self.data = pd.read_csv(path)
	
	def get_data(self, round):
		assert round >= 0, 'Round must be a positive integer'
		assert round <= self.round, 'Round must be less than or equal to the total number of rounds'
		assert type(round) == int, 'Round must be an integer'
		if round == 0:
			return pd.read_csv(self.path+f'/result/attack_{self.model}_output.csv')
		else:
			return pd.read_csv(self.path+f'/iter_{round-1}/attack_{self.model}_output.csv')
	
	def get_score_avg(self, round):
		result = self.get_data(round).groupby('Iter').mean(numeric_only=True)
		# print(result)
		return result
	
	def get_score_std(self, round):
		result = self.get_data(round).groupby('Iter').std(numeric_only=True)
		# print(result)
		return result
	



