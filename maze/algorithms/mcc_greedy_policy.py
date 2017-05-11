# -*- encoding = utf8 -*-
import random
import copy
import time

from monte_carlo_control import MonteCarloControl


class MonteCarloControlGreedyPolicy(MonteCarloControl):

	def __init__(self, observe=True, save=False, map_size=(9,9), index=0, seed=0):
		MonteCarloControl.__init__(self, observe, save, map_size, index, seed)

	def train(self, iteration=100000):
		start_time = time.time()
		for _ in xrange(iteration):
			# initial state
			self.env.reset()
			stateid = self.states_dict[self._point2string(self.env.start_point)]
			# initial gone
			gone, trajectory, trajectory_dict = True, [], {}
			while gone:
				sample = [stateid]
				state_string = self._point2string(self.states[stateid])
				if state_string not in trajectory_dict:
					trajectory_dict[state_string] = None
				# sample action from policy
				actionid = self._sample_action(stateid, self.policy)
				action = self.actions[actionid]
				sample.append(actionid)
				# get information from evironment
				gone, reward, state = self.env.step(action=action)
				self.n_try += 1
				# if go back or hit wall, change reward and stop this traj
				if (gone and self._point2string(state) in trajectory_dict) or reward == -10:
					wall_string = self._point2string(self.states[stateid])
					wall_string += '#' + str(action)
					if wall_string not in self.wall_reward:
						self.wall_reward[wall_string] = 0
					self.wall_reward[wall_string] -= self.wall_punish
					reward = self.wall_reward[wall_string]
					sample.append(reward)
					trajectory.append(sample)
					break
				# add trajectory
				sample.append(reward)
				trajectory.append(sample)
				# add new state into self.states and self.policy
				if gone:
					stateid = self._add_state(state)
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
			if trajectory[-1][2] == 1000:
				self.win_trajectory = trajectory
			# policy iteration
			self._estimate_state_action_function(trajectory)
			self.policy = self._greedy_policy_iteration()
			# exit judge
			if iteration % 100 == 0:
				if len(self.latest_rewards) == 50:
					self.latest_rewards = self.latest_rewards[1:]
				self.latest_rewards.append(trajectory[-1][2])
				if self._can_exit(self.latest_rewards):
					break
		if self.observe:
			self.env.close()
		end_time = time.time()
		print '%s consumes %i tries' % ('greedy-policy', self.n_try)
		self.log.append('%s consumes %i tries' % ('greedy-policy', self.n_try))
		print '%s consumes %.2f seconds' % ('greedy-policy', end_time-start_time)
		self.log.append('%s consumes %.2f seconds' % ('greedy-policy', end_time-start_time))
		# save trajectory
		self._save_trajectory(self.trajectory_list, self.win_trajectory)
		self._save_log(self.log, self._get_log_path(self.map_size, self.index))
	
	def _add_state(self, state):
		state_string = self._point2string(state)
		if state_string not in self.states_dict:
			self.states.append(state)
			self.states_dict[state_string] = len(self.states_dict)
			self.policy.append([1.0/len(self.actions) for _ in range(len(self.actions))])
			self.state_action_func.append([0 for _ in range(len(self.actions))])
			self.state_action_count.append([0 for _ in range(len(self.actions))])
		return self.states_dict[state_string]
	
	def _estimate_state_action_function(self, trajectory):
		for trajid, info in enumerate(trajectory):
			stateid, actionid, _ = info
			accumulated_reward = sum([(self.gamma**idx)*trajectory[trajid:][idx][2] for idx in \
				range(len(trajectory[trajid:]))])
			self.state_action_count[stateid][actionid] += 1
			self.state_action_func[stateid][actionid] += 1.0 * (accumulated_reward - \
				self.state_action_func[stateid][actionid]) / \
				self.state_action_count[stateid][actionid]
				
	def _greedy_policy_iteration(self):
		new_policy = copy.deepcopy(self.policy)
		for stateid in range(len(self.states)):
			action_value = [[actionid, self.state_action_func[stateid][actionid]] \
				for actionid in range(len(self.actions))]
			optimal_action = max(action_value, key=lambda x: x[1])[0]
			for actionid in range(len(self.actions)):
				new_policy[stateid][actionid] = 0
			new_policy[stateid][optimal_action] = 1
		
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
		return '../experiments/monte_carlo_control/trajectory/greedy-policy' + \
			'_(' + str(map_size[0]) + '_' + str(map_size[1]) + ')_' + str(index) + '.txt'


for size in [15]: # map size must be odd
	print 'size is (%i, %i) %s' % (size, size, '#'*50)
	for i in range(20):
		print 'example %i %s' % (i, '='*50)
		greedy_policy = MonteCarloControlGreedyPolicy(observe=False, save=False, \
										map_size=(size,size), index=i, seed=i)
		greedy_policy.train()