# -*- encoding = utf8 -*-
import random
import copy
import time

from temporal_difference import TemporalDifference


class SARSA(TemporalDifference):

	def __init__(self, observe=True, save=False, map_size=(9,9), index=0, seed=0):
		TemporalDifference.__init__(self, observe, save, map_size, index, seed)
		# init params
		self.gamma = 0.9
		self.epsilon = 0.1
		self.alpha = 0.1

	def train(self, iteration=1000):
		start_time = time.time()
		for _ in xrange(iteration):
			# initial state
			self.env.reset()
			stateid = self.states_dict[self._point2string(self.env.start_point)]
			# initial gone
			gone, trajectory = True, []
			# choose action
			actionid = self._sample_action(stateid, self.policy)
			action = self.actions[actionid]
			while gone:
				sample = [stateid]
				sample.append(actionid)
				# get information from evironment
				gone, reward, new_state = self.env.step(action=action)
				self._add_state(new_state)
				new_stateid = self.states_dict[self._point2string(new_state)]
				self.n_try += 1
				# if hit wall, stop this traj
				if gone == False and reward == -100:
					sample.append(reward)
					trajectory.append(sample)
					self._estimate_state_action_function(stateid=stateid, actionid=actionid, \
														reward=reward, new_stateid=new_stateid, \
														new_actionid=-1, gone=gone)
					self.policy = self._soft_policy_iteration()
				elif gone == False and reward == 100:
					sample.append(reward)
					trajectory.append(sample)
					self._estimate_state_action_function(stateid=stateid, actionid=actionid, \
														reward=reward, new_stateid=new_stateid, \
														new_actionid=-1, gone=gone)
					self.policy = self._soft_policy_iteration()
				else:
					# sample action
					new_actionid = self._sample_action(new_stateid, self.policy)
					# add trajectory
					sample.append(reward)
					trajectory.append(sample)
					self._estimate_state_action_function(stateid=stateid, actionid=actionid, \
														reward=reward, new_stateid=new_stateid, \
														new_actionid=new_actionid, gone=gone)
					self.policy = self._soft_policy_iteration()
					stateid = new_stateid
					actionid = new_actionid
					action = self.actions[actionid]
				# render and observe
				if self.observe:
					if iteration % 100 == 0:
						state_action_func_dict = {}
						for state in range(len(self.state_action_func)):
							state_string = self._point2string(self.states[state])
							alpha_list = copy.deepcopy(self.state_action_func[state])
							alpha_list = [t-min(self.state_action_func[state]) for t in alpha_list]
							if sum(alpha_list) == 0:
								alpha_list = [0.0]*4
							else:
								alpha_list = [1.0*t/sum(alpha_list) for t in alpha_list]
							state_action_func_dict[state_string] = alpha_list
						self.env.render(state_action_func_dict)
			# save trajectory
			self.trajectory_list.append(trajectory)
			if trajectory[-1][2] == 100:
				self.win_trajectory = trajectory
			# exit judge
			if iteration % 100 == 0:
				if len(self.latest_rewards) == 50:
					self.latest_rewards = self.latest_rewards[1:]
				self.latest_rewards.append(trajectory[-1][2])
		if self.observe:
			self.env.close()
		end_time = time.time()
		print '%s consumes %i tries' % ('SARSA', self.n_try)
		self.log.append('%s consumes %i tries' % ('SARSA', self.n_try))
		print '%s consumes %.2f seconds' % ('SARSA', end_time-start_time)
		self.log.append('%s consumes %.2f seconds' % ('SARSA', end_time-start_time))
		# save trajectory
		self._save_trajectory(self.trajectory_list, self.win_trajectory)
		self._save_log(self.log, self._get_log_path(self.map_size, self.index))
	
	def _estimate_state_action_function(self, stateid, actionid, reward, \
									new_stateid, new_actionid, gone=False):
		if gone:
			self.state_action_func[stateid][actionid] += self.alpha * (reward + \
				self.gamma * self.state_action_func[new_stateid][new_actionid] - \
				self.state_action_func[stateid][actionid])
		else:
			self.state_action_func[stateid][actionid] += self.alpha * (reward - \
				self.state_action_func[stateid][actionid])
				
	def _soft_policy_iteration(self):
		new_policy = copy.deepcopy(self.policy)
		for stateid in range(len(self.states)):
			action_value = [[actionid, self.state_action_func[stateid][actionid]] \
				for actionid in range(len(self.actions))]
			optimal_action = max(action_value, key=lambda x: x[1])[0]
			for actionid in range(len(self.actions)):
				new_policy[stateid][actionid] = 1.0 * self.epsilon / len(self.actions)
			new_policy[stateid][optimal_action] += 1 - self.epsilon
		
		return new_policy
	
	def _sample_action(self, state, policy):
		prob_array = policy[state]
		action_list = []
		for action in range(len(prob_array)):
			action_list.extend([action] * int(prob_array[action]*1000))
			
		return random.choice(action_list)
	
	def _get_image_path(self, map_size, index):
		return '../pic/env/maze_(' + str(map_size[0]) + '_' + str(map_size[1]) + \
			')_' + str(index) + '.png' 
	
	def _get_log_path(self, map_size, index):
		return '../experiments/trajectory/SARSA_(' + \
			str(map_size[0]) + '_' + str(map_size[1]) + ')_' + str(index) + '.txt'


sarsa = SARSA(observe=False, save=False, map_size=(12,4), index=0, seed=0)
sarsa.train()