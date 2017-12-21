# -*- encoding = utf8 -*-
import random
import copy
import time

import numpy
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop


class QLearning:

	def __init__(self, index=0, seed=0):
		self.env = gym.make('CartPole-v0')
		self.index = index
		# init variable
		self.actions = ['left', 'right']
		self.trajectory_list = []
		self.memory = []
		self.memory_size = 32
		# init params
		self.gamma = 0.9
		self.epsilon = 1.0
		self.epsilon_bound = 0.01
		self.epsilon_decrease = 0.99
		# init model params
		self.state_size = 4
		self.hidden_size = 20
		self.learning_rate = 0.01
		self.model = self._build_model()

	def train(self, iteration=1000):
		start_time = time.time()
		n_iter = 0
		while True:
			n_iter += 1
			# initial state
			state = self.env.reset()
			# initial gone
			done, trajectory = False, []
			while not done:
				# render and observe
				# self.env.render()
				# choose action
				sample = [list(state)]
				actionid = self._sample_action(state)
				sample.append(actionid)
				# get information from evironment
				new_state, reward, done, _ = self.env.step(action=actionid)
				reward = reward if not done else -10
				sample.append(reward)
				trajectory.append(sample)
				# store memory
				self.memory.append((numpy.reshape(state, [1, self.state_size]), \
								actionid, reward, \
								numpy.reshape(new_state, [1, self.state_size]), done))
				# update state
				state = copy.deepcopy(new_state)
				
			# memory replay
			self._memory_replay(size=self.memory_size)
			# save trajectory
			print('@iter: %i, score: %i, epsilon: %.2f' % (n_iter, \
				int(sum([t[2] for t in trajectory[:-1]])), self.epsilon))
			self.trajectory_list.append(trajectory)
			
		end_time = time.time()
		"""
		print '%s consumes %i tries' % ('QLearning', self.n_try)
		self.log.append('%s consumes %i tries' % ('QLearning', self.n_try))
		print '%s consumes %.2f seconds' % ('QLearning', end_time-start_time)
		self.log.append('%s consumes %.2f seconds' % ('QLearning', end_time-start_time))
		# save trajectory
		self._save_trajectory(self.trajectory_list, [])
		self._save_log(self.log, self._get_log_path(self.index))
		"""
		
	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Dense(self.hidden_size, input_dim=self.state_size, activation='tanh'))
		model.add(Dense(self.hidden_size, activation='tanh', kernel_initializer='uniform'))
		model.add(Dense(len(self.actions), activation='linear'))
		model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))
		return model
		
	def _memory_replay(self, size=32):
		batch_size = min(size, len(self.memory))
		batch_data = random.sample(self.memory, batch_size)
		X = numpy.zeros((batch_size, self.state_size))
		Y = numpy.zeros((batch_size, len(self.actions)))
		for i in range(batch_size):
			state, action, reward, next_state, done = batch_data[i]
			target = self.model.predict(state)[0]
			if done:
				target[action] = reward
			else:
				target[action] = reward + self.gamma * \
					numpy.amax(self.model.predict(next_state)[0])
			X[i], Y[i] = state, target
		self.model.fit(X, Y, batch_size=batch_size, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_bound:
			self.epsilon *= self.epsilon_decrease
		
	def _sample_action(self, state):
		state = numpy.reshape(state, [1, self.state_size])
		if random.random() < self.epsilon:
			action = random.choice(range(len(self.actions)))
		else:
			q_value = self.model.predict(state)[0,:]
			action = max(enumerate(q_value), key=lambda x: x[1])[0]
		return action
	
	def _get_image_path(self, index):
		return '../pic/env/flappy_' + str(index) + '.png'
	
	def _get_log_path(self, index):
		return '../experiments/trajectory/QLearning_' + str(index) + '.txt'


ql = QLearning(index=0, seed=0)
ql.train()