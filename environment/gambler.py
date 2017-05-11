# -*- encoding = utf8 -*-
import random


class Environment:
	
	def __init__(self, p=0.4):
		self.capital = 1
		self.p = p
		self.reset()
		
	def reset(self):
		self.capital = 1
		return self.capital
		
	def render(self):
		print 'I have %i coins!' % self.capital
		
	def step(self, action):
		if random.random() < self.p:
			self.capital -= action
		else:
			self.capital += action
		if self.capital >= 100:
			done = True
			reward = 1
			state = -1
		elif self.capital <= 0:
			done = True
			reward = -1
			state = -1
		else:
			done = False
			reward = 0
			state = self.capital 
			
		return state, reward, done