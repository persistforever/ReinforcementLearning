# -*- coding: utf8 -*-
import random


class Environment:
	
	def __init__(self):
		# we have [A,2,3,4,5,6,7,8,9,10] cards
		self.card = ['A'] + [str(t) for t in range(2,11)]
		# states [1,2,3,...,20,21,bust]
		self.states = range(1, 23)
		self.actions = ['hit', 'stick']
		
	def reset(self):
		self.dealer_score, self.dealer_natural, dealer_showing = self._init_dealer()
		self.player_score, self.player_natural = self._init_player()
		if self.player_natural:
			reward = 0 if self.dealer_natural else 1
			done = True
		else:
			reward = 0
			done = False
		state = self.player_score if self.player_score <= 21 else 22
		return dealer_showing, state, reward, done
		
	def _init_dealer(self):
		faceup = random.choice(self.card)
		facedown = random.choice(self.card)
		natural = False
		if faceup == 'A' and facedown == '10' or faceup == '10' and facedown == 'A':
			score = 21
			natural = True
		elif faceup == 'A' and facedown == 'A':
			score = 12
		else:
			score = 0
			score += 11 if faceup == 'A' else int(faceup)
			score += 11 if facedown == 'A' else int(facedown)
		# print faceup, facedown, score
		while score < 17:
			card = random.choice(self.card)
			if card == 'A':
				if score + 11 <= 21:
					score += 11
				else:
					score += 1
			else:
				score += int(card)
			# print card, score
		# print score, natural, faceup
		return score, natural, faceup
		
	def _init_player(self):
		faceup = random.choice(self.card)
		facedown = random.choice(self.card)
		natural = False
		if faceup == 'A' and facedown == '10' or faceup == '10' and facedown == 'A':
			score = 21
			natural = True
		elif faceup == 'A' and facedown == 'A':
			score = 12
		else:
			score = 0
			score += 11 if faceup == 'A' else int(faceup)
			score += 11 if facedown == 'A' else int(facedown)
		# print score, natural
		return score, natural
		
	def render(self):
		print 'player score: %i, dealer score: %i' % (self.player_score, self.dealer_score)
		
	def step(self, action):
		if action == 'hit':
			card = random.choice(self.card)
			if card == 'A':
				if self.player_score + 11 <= 21:
					self.player_score += 11
				else:
					self.player_score += 1
			else:
				self.player_score += int(card)
			done = True if self.player_score > 21 else False
		elif action == 'stick':
			done = True
		# judge new state
		if done:
			if self.dealer_score == 21 and self.dealer_natural:
				if self.player_natural:
					reward = 0
				else:
					reward = -1
			elif self.dealer_score == 21 and not self.dealer_natural:
				if self.player_natural:
					reward = 1
				elif self.player_score == 21:
					reward = 0
				else:
					reward = -1
			elif self.dealer_score > 21:
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
		state = self.player_score if self.player_score <= 21 else 22
			
		return state, reward, done