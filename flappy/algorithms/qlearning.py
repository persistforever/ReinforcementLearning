# -*- encoding = utf8 -*-
import random
import copy
import time

import numpy
import theano

from environment.flappy import Environment
from flappy.algorithms.network import QNetwork 


class QLearning:

	def __init__(self, index=0, seed=0):
		self.env = Environment()
		self.index = index
		# init variable
		self.actions = self.env.actions
		self.trajectory_list = []
		# init q network
		rng = numpy.random.RandomState(int(random.random()*100))
		print '%s %s %s' % ('='*5, 'Compile Network Start', '='*5)
		self.q_network = QNetwork(rng=rng, n_state=5, n_action=len(self.actions))
		self.q_func = self.q_network.get_q_func()
		self.q_update = self.q_network.train_one_batch()
		print '%s %s %s' % ('='*5, 'Compile Network End', '='*5)
		# init params
		self.gamma = 0.9
		self.epsilon = 0.1
		self.yita = 0.001

	def train(self, iteration=1000):
		start_time = time.time()
		while True:
			# initial state
			state = self.env.reset()
			# initial gone
			done, trajectory = False, []
			while not done:
				sample = [state]
				# choose action
				actionid = self._sample_action(state)
				action = self.actions[actionid]
				sample.append(actionid)
				# get information from evironment
				done, reward, new_state = self.env.step(action=action)
				sample.append(reward)
				trajectory.append(sample)
				# get y
				if done:
					y = reward
				else:
					new_state = numpy.array([new_state], dtype=theano.config.floatX)
					q_value = self.q_func(new_state)[0,:]
					y = reward + self.gamma * max(q_value)
				self.q_update(numpy.array([state], dtype=theano.config.floatX), \
							numpy.array([actionid], dtype=theano.config.floatX), \
							numpy.array([y], dtype=theano.config.floatX), self.yita)
				# render and observe
				self.env.render()
			# save trajectory
			self.trajectory_list.append(trajectory)
		end_time = time.time()
		print '%s consumes %i tries' % ('QLearning', self.n_try)
		self.log.append('%s consumes %i tries' % ('QLearning', self.n_try))
		print '%s consumes %.2f seconds' % ('QLearning', end_time-start_time)
		self.log.append('%s consumes %.2f seconds' % ('QLearning', end_time-start_time))
		# save trajectory
		self._save_trajectory(self.trajectory_list, [])
		self._save_log(self.log, self._get_log_path(self.index))
	
	def _sample_action(self, state):
		if random.random() < self.epsilon:
			action = random.choice(range(len(self.actions)))
		else:
			state = numpy.array([state], dtype=theano.config.floatX)
			q_value = self.q_func(state)[0,:]
			action = max(enumerate(q_value))[0]
			
		return action
	
	def _get_image_path(self, index):
		return '../pic/env/flappy_' + str(index) + '.png'
	
	def _get_log_path(self, index):
		return '../experiments/trajectory/QLearning_' + str(index) + '.txt'


ql = QLearning(index=0, seed=0)
ql.train()