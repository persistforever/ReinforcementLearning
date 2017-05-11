# -*- encoding = utf8 -*-
import random


class Environment:
	
	def __init__(self):
		# we have [A,2,3,4,5,6,7,8,9,10] cards
		self.card = range(1, 11)
		self.actions = ['hit', 'stick']
		
	def reset(self):
		self.dealer_score = self._init_dealer()
		self.player_score = 0
		return self.player_score
		
	def _init_dealer(self):
		self.dealer_score = 0
		while self.dealer_score < 17:
			card = random.choice(self.card)
			self.dealer_score += card
			print self.dealer_score
		return self.dealer_score
		
	def render(self):
		print 'player score: %i, dealer score: %i' % (self.player_score, self.dealer_score)
		
	def step(self, action):
		if action == 'hit':
			card = random.choice(self.card)
			self.player_score += card
			done = True if self.player_score > 21 else False
		elif action == 'stick':
			done = True
		# judge new state
		if done:
			if self.dealer_score > 21:
				# dealer bust
				if self.player_score > 21:
					reward = 0
				else:
					reward = 1
			else:
				if self.player_score > 21:
					reward = -1
				else:
					if self.player_score > self.dealer_score:
						reward = 1
					elif self.player_score == self.dealer_score:
						reward = 0
					else:
						reward = -1
		else:
			reward = 0
		state = self.player_score
			
		return state, reward, done