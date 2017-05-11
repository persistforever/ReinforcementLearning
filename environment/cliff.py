# -*- encoding = utf8 -*-
import numpy
import rendering


class Environment:
	
	def __init__(self, map_size=(10, 4), observe=True, seed=0, save=False, path=None):
		self.height = map_size[0]
		self.width = map_size[1]
		self.map = numpy.ones((self.height, self.width), dtype=int)
		for i in range(1, self.height-1):
			self.map[i, 0] = 0
		
		self.valid_actions = ['up', 'down', 'left', 'right']
		self.start_point = [0, 0] # start point
		self.end_point = [self.height-1, 0] # end point
		self.block_height, self.block_width = 30, 30
		
		self.reset()
		
		if save:
			self.viewer = rendering.Viewer(self.height*self.block_height, \
										self.width*self.block_width)
			self.render(is_save=True, path=path)
			self.viewer.window.close()
		if observe:
			self.viewer = rendering.Viewer(self.height*self.block_height, \
										self.width*self.block_width)
		
	def reset(self):
		self.state = self.start_point = [0, 0] # start point
		self.life = 1.0
		
	def close(self):
		self.viewer.window.close()
		
	def render(self, state_action_func=None, is_save=False, path=None):
		block_height, block_width = self.block_height, self.block_width
		geometroies = []
		for x in range(self.height):
			for y in range(self.width):
				if self.map[x, y] == 1: # floor
					floor = rendering.Block([x*block_height,y*block_width], \
										block_height, block_width)
					floor.set_line_color(235,235,235)
					floor.set_fill_color(248,248,255)
					geometroies.append(floor)
				elif self.map[x, y] == 0: # wall
					wall = rendering.Block([x*block_height,y*block_width], \
										block_height, block_width)
					wall.set_line_color(64,64,64)
					wall.set_fill_color(13,13,13)
					geometroies.append(wall)
				if x == self.start_point[0] and y == self.start_point[1]: # start point
					start = rendering.Block([x*block_height,y*block_width], \
										block_height, block_width/2)
					start.set_line_color(124,205,124)
					start.set_fill_color(124,205,124)
					geometroies.append(start)
				if x == self.end_point[0] and y == self.end_point[1]: # end point
					end = rendering.Block([x*block_height,y*block_width], \
										block_height, block_width/2)
					end.set_line_color(238,59,59)
					end.set_fill_color(238,59,59)
					geometroies.append(end)
				if state_action_func:
					state_string = str(x) + '#' + str(y) 
					if state_string in state_action_func:
						alpha_list = state_action_func[state_string]
						# top action
						action = rendering.Triangle([(x+0.5)*block_height,(y+0.5)*block_width], \
											block_height, block_width)
						action.set_fill_color_up(205,205,0, alpha_list[0])
						action.set_fill_color_down(205,205,0, alpha_list[1])
						action.set_fill_color_left(205,205,0, alpha_list[2])
						action.set_fill_color_right(205,205,0, alpha_list[3])
						geometroies.append(action)
				if x == self.state[0] and y == self.state[1]:
					agent = rendering.Agent([(x+0.5)*block_height, (y+0.5)*block_width], \
										block_height/2, block_width/2)
					agent.set_fill_color(100,149,237, alpha=self.life)
					geometroies.append(agent)
		self.viewer.render(geometroies, is_save, path)
		
	def step(self, action):
		gone, reward = True, 0
		
		# calculate next state
		if action == 'up':
			next_state = [self.state[0], self.state[1]+1]
		elif action == 'down':
			next_state = [self.state[0], self.state[1]-1]
		elif action == 'left':
			next_state = [self.state[0]-1, self.state[1]]
		elif action == 'right':
			next_state = [self.state[0]+1, self.state[1]]
		else:
			raise Exception('Action Error!')
			
		# judge gone
		if self._is_out_off_map(next_state):
			gone = False
			reward = -100
			self.state = [self.start_point[0], self.start_point[1]]
		elif self._is_hit_wall(next_state):
			gone = False
			reward = -100
			self.state = [self.start_point[0], self.start_point[1]]
		elif self._is_end_point(next_state):
			gone = False
			reward = 100
			self.state = [self.start_point[0], self.start_point[1]]
		else:
			gone = True
			reward = -1
			self.state = [next_state[0], next_state[1]]
			
		return gone, reward, self.state
			
	def _is_hit_wall(self, state):
		return True if self.map[state[0], state[1]] == 0 else False
			
	def _is_end_point(self, state):
		return True if state[0] == self.end_point[0] and \
			state[1] == self.end_point[1] else False
			
	def _is_out_off_map(self, state):
		return True if state[0] < 0 or state[1] < 0 or \
			state[0] >= self.height or state[1] >= self.width else False