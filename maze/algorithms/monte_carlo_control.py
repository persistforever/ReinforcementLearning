# -*- encoding = utf8 -*-

from environment.maze import Environment 


class MonteCarloControl:

	def __init__(self, observe=True, save=False, map_size=(9,9), index=0, seed=0):
		self.map_size, self.index = map_size, index
		self.env = Environment(map_size=self.map_size, observe=observe, seed=seed, \
							save=save, path=self._get_image_path(self.map_size, self.index))
		# init actions
		self.actions = self.env.valid_actions
		# init states and policy
		self.states, self.states_dict = [], {}
		self.policy = []
		self.state_action_func, self.state_action_count = [], []
		self._add_state(self.env.start_point)
		# init wall dict
		self.wall_reward = {}
		self.trajectory_list = []
		self.wall_punish = 100
		# init params
		self.gamma = 0.95
		self.epsilon = 0.005
		# latest 50 times reward
		self.latest_rewards = []
		self.n_try = 0
		self.observe = observe
		self.log = []

	def train(self, iteration=100000):
		raise NotImplemented
	
	def _estimate_state_action_function(self, trajectory):
		raise NotImplemented
				
	def _policy_iteration(self):
		raise NotImplemented
	
	def _point2string(self, point):
		return '#'.join([str(t) for t in point])
	
	def _sample_action(self, state, policy):
		raise NotImplemented
	
	def _add_state(self, state):
		raise NotImplemented
	
	def _can_exit(self, rewards):
		n_win = sum([1 for reward in rewards if reward == 1000])
		return True if n_win >= 40 else False
	
	def _get_image_path(self, map_size, index):
		raise NotImplemented
	
	def _get_log_path(self, map_size, index):
		raise NotImplemented
			
	def _save_trajectory(self, trajectory_list, win_trajectory):
		self.log.append('%s try path %s' % ('-'*20, '-'*20))
		for trajectory in trajectory_list:
			path_string = []
			for stateid, _, _ in trajectory:
				path_string.append('(' + str(self.states[stateid][0]) + ',' + \
								str(self.states[stateid][1]) + ')')
			path_string = ' -> '.join(path_string)
			self.log.append(path_string)
			
		self.log.append('%s win path %s' % ('-'*20, '-'*20))
		path_string = []
		for stateid, _, _ in win_trajectory:
			path_string.append('(' + str(self.states[stateid][0]) + ',' + \
							str(self.states[stateid][1]) + ')')
		path_string = ' -> '.join(path_string)
		self.log.append(path_string)
		
	def _save_log(self, log_list, path):
		with open(path, 'w') as fw:
			for log in log_list:
				fw.writelines(log + '\n')