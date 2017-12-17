# -*- coding: utf-8 -*-
# author: ronniecao
# time: 2017/12/17
# description: DQN of flappy bird
import os
import random
import time
import numpy
import Queue
import copy
import cv2
from environment.flappy import Environment


class QLearning:
	def __init__(self):
		self.env = Environment(is_show=False)
		self.init_image = self.env.reset()
		self.flap_prob = 0.1
		self.replay_memory = []
		self.replay_memory_maxsize = 5000
		self.init_replay_memory()
		for item in self.replay_memory:
			print(len(item['state']), item['reward'], item['is_end'])

	def init_replay_memory(self):
		image = self.init_image
		image_queue = [image]
		image_queue_maxsize = 4
		is_end = False
		while not is_end:
			rnd = random.random()
			print(rnd)
			action = 'flap' if rnd < self.flap_prob else 'noflap'
			next_image, reward, is_end = self.env.render(action)
			# 如果image_queue满，则将当前状态存入replay_memory
			if len(image_queue) >= image_queue_maxsize:
				state = copy.deepcopy(image_queue)
				image_queue.pop(0)
				image_queue.append(next_image)
				next_state = copy.deepcopy(image_queue)
				self.replay_memory.append({
					'state': state, 'action': action, 'reward': reward, 
					'is_end': is_end, 'next_state': next_state})
			else:
				image_queue.append(next_image)
			image = next_image


if __name__ == '__main__':
	qlearning = QLearning()