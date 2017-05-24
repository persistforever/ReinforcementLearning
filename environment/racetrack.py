# -*- coding: utf8 -*-
import numpy
import rendering
import random


class Environment:
	
	def __init__(self, observe=True, save=False, path=None):
		self.height = 30
		self.width = 15
		
		# 创建地图
		self.map = numpy.zeros((self.height, self.width), dtype='int')
		for i in range(0, self.height):
			for j in range(3, 10):
				self.map[i, j] = 1
		for i in range(0, 6):
			for j in range(3, self.width):
				self.map[i, j] = 1
		for i in range(1, self.height-3):
			self.map[i, 2] = 1
		for i in range(3, self.height-10):
			self.map[i, 1] = 1
		for i in range(4, 14):
			self.map[i, 0] = 1
		self.map[6, 10] = 1
		# 起点
		for i in range(3, 10):
			self.map[self.height-1, i] = 2
		# 终点
		for i in range(0, 6):
			self.map[i, self.width-1] = 3
		
		self.valid_actions = ['up', 'down', 'left', 'right']
		self.block_height, self.block_width = 20, 20
		
		if save:
			self.viewer = rendering.Viewer(self.height*self.block_height, \
										self.width*self.block_width)
			self.render(is_save=True, path=path)
			self.viewer.window.close()
		if observe:
			self.viewer = rendering.Viewer(self.height*self.block_height, \
										self.width*self.block_width)
		
	def reset(self):
		# 创建地图
		self.map = numpy.zeros((self.height, self.width), dtype='int')
		for i in range(0, self.height):
			for j in range(3, 10):
				self.map[i, j] = 1
		for i in range(0, 6):
			for j in range(3, self.width):
				self.map[i, j] = 1
		for i in range(1, self.height-3):
			self.map[i, 2] = 1
		for i in range(3, self.height-10):
			self.map[i, 1] = 1
		for i in range(4, 14):
			self.map[i, 0] = 1
		self.map[6, 10] = 1
		# 起点
		for i in range(3, 10):
			self.map[self.height-1, i] = 2
		# 终点
		for i in range(0, 6):
			self.map[i, self.width-1] = 3
			
		self.state = self.start_point = [self.height-1, random.randint(3,9)] # start point
		self.velx, self.vely = 0, 0
		self.life = 1.0
		
		return self.state, 0, False
		
	def close(self):
		self.viewer.window.close()
		
	def render(self, state_action_func=None, is_save=False, path=None):
		block_height, block_width = self.block_height, self.block_width
		geometroies = []
		for x in range(self.height):
			for y in range(self.width):
				# 绘制地图
				if self.map[x, y] == 1: # 赛道
					floor = rendering.Block([x*block_height,y*block_width], \
										block_height, block_width)
					floor.set_line_color(235,235,235)
					floor.set_fill_color(248,248,255)
					geometroies.append(floor)
				elif self.map[x, y] == 0: # 场外
					wall = rendering.Block([x*block_height,y*block_width], \
										block_height, block_width)
					wall.set_line_color(64,64,64)
					wall.set_fill_color(13,13,13)
					geometroies.append(wall)
				if self.map[x, y] == 2: # 起点
					start = rendering.Block([x*block_height,y*block_width], \
										block_height, block_width)
					start.set_line_color(124,205,124)
					start.set_fill_color(124,205,124)
					geometroies.append(start)
				if self.map[x, y] == 3: # 终点
					end = rendering.Block([x*block_height,y*block_width], \
										block_height, block_width)
					end.set_line_color(238,59,59)
					end.set_fill_color(238,59,59)
					geometroies.append(end)
				if state_action_func:
					state_string = '%i#%i' % (x, y)
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
				if x == self.state[0] and y == self.state[1]: # 选手位置
					agent = rendering.Agent([(x+0.5)*block_height, (y+0.5)*block_width], \
										block_height/2, block_width/2)
					agent.set_fill_color(100,149,237, alpha=self.life)
					geometroies.append(agent)
				elif self.map[x, y] == 4: # 选手走过的位置
					end = rendering.Block([x*block_height,y*block_width], \
										block_height, block_width)
					end.set_line_color(100,149,237)
					end.set_fill_color(100,149,237)
					geometroies.append(end)
					
		self.viewer.render(geometroies, is_save, path)
		
	def step(self, action):
		done, reward = False, 0
		
		# 计算下一个状态
		self.velx += action[0]
		if self.velx > 5:
			self.velx = 5
		elif self.velx < -5:
			self.velx = -5
		self.vely += action[1]
		if self.vely > 5:
			self.vely = 5
		elif self.vely < -5:
			self.vely = -5
		next_state = [self.state[0]+self.velx, self.state[1]+self.vely]
		self.map[self.state[0], self.state[1]] = 4
			
		# 判断轨迹是否结束
		if self._is_out_off_court(next_state):
			done = True
			reward = -100
			self.state = [self.start_point[0], self.start_point[1]]
		elif self._is_end_point(next_state):
			done = True
			reward = 1000
			self.state = [self.start_point[0], self.start_point[1]]
		else:
			done = False
			reward = -1
			self.state = [next_state[0], next_state[1]]
			
		return self.state, reward, done
			
	def _is_end_point(self, state):
		return True if self.map[state[0], state[1]] == 3 else False
			
	def _is_out_off_court(self, state):
		isout = False
		if state[0] >= self.height or state[0] < 0:
			return True
		if state[1] >= self.width or state[1] < 0:
			return True
		if self.map[state[0], state[1]] == 0:
			isout = True
			
		return isout
		