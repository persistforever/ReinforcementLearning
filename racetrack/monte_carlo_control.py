# -*- coding: utf8 -*-
import time
import copy
import numpy
import random

from environment.racetrack import Environment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MonteCarlo:
    
    def __init__(self, seed=0, path=None):
        self.path = path
        self.env = Environment()
        self.gamma = 1.0
        self.epsilon = 0.1
        self.height, self.width = 30, 15
        self.states, self.states_dict = self._init_states()
        self.actions = self._init_actions()
        self.value_function = numpy.zeros((len(self.states), ), dtype='float')
        self.value_count = numpy.zeros((len(self.states), ), dtype='float')
        self.action_function = numpy.zeros((len(self.states), len(self.actions)), dtype='float')
        self.action_count = numpy.zeros((len(self.states), len(self.actions)), dtype='float')
        self.policy = self._init_policy(seed=seed)
        self.episodes = []
        self.wall_rewards = {}
    
    def _init_states(self):
        states, states_dict, n = [], {}, 0
        for i in range(self.height):
            for j in range(self.width):
                state = '%i#%i' % (i, j)
                states.append(state)
                states_dict[state] = n
                n += 1
        
        return states, states_dict
    
    def _init_actions(self):
        actions = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                actions.append([i, j]) 
    
        return actions
    
    def _init_policy(self, seed=0):
        # 使用随机策略初始化
        random.seed(seed)
        policy = numpy.zeros((len(self.states), len(self.actions)), dtype='float')
        for stateid in range(len(self.states)):
            for actionid in range(len(self.actions)):
                policy[stateid, actionid] = 1.0 / len(self.actions)
        
        return policy
        
    def monte_carlo_control_first_visit_exploring_start(self):
        n_iter = 0
        pic_dict = {}
        while True:
            # 初始化
            state, reward, done = self.env.reset()
            position_str = '%i#%i' % (state[0], state[1])
            stateid = self.states_dict[position_str]
            episode = []
            # 开始时探索
            self.env.render()
            actionid = random.choice(range(len(self.actions)))
            new_state, reward, done = self.env.step(action=self.actions[actionid])
            new_stateid = self.states_dict['%i#%i' % (new_state[0], new_state[1])]
            # print new_stateid, reward, done
            episode.append([stateid, actionid, reward])
            stateid = new_stateid
            # 根据当前策略生成轨迹
            while not done and len(episode) < 100:
                self.env.render()
                actionid = self._sample_episode(self.policy, stateid)
                new_state, reward, done = self.env.step(action=self.actions[actionid])
                new_stateid = self.states_dict['%i#%i' % (new_state[0], new_state[1])]
                # print new_stateid, reward, done
                episode.append([stateid, actionid, reward])
                stateid = new_stateid
            # 如果轨迹中走了回头路，就对其reward进行惩罚
            state_visited = {}
            for idx, [stateid, actionid, reward] in enumerate(episode):
                if stateid not in state_visited:
                    state_visited[stateid] = None
                else:
                    episode[idx][2] = -100
            # 对每一个冲出赛道的点进行重复惩罚
            if episode[-1][2] == -100:
                [stateid, actionid, reward] = episode[-1]
                if stateid not in self.wall_rewards:
                    self.wall_rewards[stateid] = 0
                self.wall_rewards[stateid] += reward
                episode[-1][2] = self.wall_rewards[stateid]
            print episode
            # 根据轨迹更新状态值函数
            state_visited = {}
            state_action_visited = {}
            for idx, [stateid, actionid, reward] in enumerate(episode):
                # print dealer_state, self.states[stateid], self.actions[actionid], reward
                if stateid not in state_visited:
                    state_visited[stateid] = None
                    sum_reward = sum([t[2] for t in episode[idx:]])
                    self.value_count[stateid] += 1.0
                    self.value_function[stateid] += (sum_reward - self.value_function[stateid]) \
                        / self.value_count[stateid]
                if '%i#%i' % (stateid, actionid) not in state_action_visited:
                    state_action_visited['%i#%i' % (stateid, actionid)] = None
                    sum_reward = sum([t[2] for t in episode[idx:]])
                    self.action_count[stateid, actionid] += 1.0
                    self.action_function[stateid, actionid] += (sum_reward - \
                        self.action_function[stateid, actionid]) \
                        / self.action_count[stateid, actionid]
            # 根据动作值函数更新策略
            state_visited = {}
            for _, [stateid, actionid, reward] in enumerate(episode):
                if stateid not in state_visited:
                    state_visited[stateid] = None
                    for actionid in range(len(self.actions)):
                        self.policy[stateid, actionid] = 0.0
                    actionid = max(enumerate(list(self.action_function[stateid,:])), \
                                   key=lambda x: x[1])[0]
                    self.policy[stateid, actionid] = 1.0
            # 如果本次走到终点，则保存图片
            if episode[-1][2] == 1000:
                # 获得保存图片的序号
                if position_str not in pic_dict:
                    pic_dict[position_str] = 0
                pic_dict[position_str] += 1
                path = self.path + '_' + position_str + '_' + str(pic_dict[position_str]) +'.jpg'
                self.env.render(is_save=True, path=path)
            n_iter += 1
        
    def monte_carlo_control_first_visit_on_policy(self):
        n_iter = 0
        pic_dict = {}
        while True:
            # 初始化
            state, reward, done = self.env.reset()
            position_str = '%i#%i' % (state[0], state[1])
            stateid = self.states_dict[position_str]
            episode = []
            # 根据当前策略生成轨迹
            while not done and len(episode) < 100:
                self.env.render()
                actionid = self._sample_episode(self.policy, stateid)
                new_state, reward, done = self.env.step(action=self.actions[actionid])
                new_stateid = self.states_dict['%i#%i' % (new_state[0], new_state[1])]
                # print new_stateid, reward, done
                episode.append([stateid, actionid, reward])
                stateid = new_stateid
            # 如果轨迹中走了回头路，就对其reward进行惩罚
            state_visited = {}
            for idx, [stateid, actionid, reward] in enumerate(episode):
                if stateid not in state_visited:
                    state_visited[stateid] = None
                else:
                    episode[idx][2] = -100
            # 对每一个冲出赛道的点进行重复惩罚
            if episode[-1][2] == -100:
                [stateid, actionid, reward] = episode[-1]
                if stateid not in self.wall_rewards:
                    self.wall_rewards[stateid] = 0
                self.wall_rewards[stateid] += reward
                episode[-1][2] = self.wall_rewards[stateid]
            print episode
            # 根据轨迹更新状态值函数
            state_visited = {}
            state_action_visited = {}
            for idx, [stateid, actionid, reward] in enumerate(episode):
                # print dealer_state, self.states[stateid], self.actions[actionid], reward
                if stateid not in state_visited:
                    state_visited[stateid] = None
                    sum_reward = sum([t[2] for t in episode[idx:]])
                    self.value_count[stateid] += 1.0
                    self.value_function[stateid] += (sum_reward - self.value_function[stateid]) \
                        / self.value_count[stateid]
                if '%i#%i' % (stateid, actionid) not in state_action_visited:
                    state_action_visited['%i#%i' % (stateid, actionid)] = None
                    sum_reward = sum([t[2] for t in episode[idx:]])
                    self.action_count[stateid, actionid] += 1.0
                    self.action_function[stateid, actionid] += (sum_reward - \
                        self.action_function[stateid, actionid]) \
                        / self.action_count[stateid, actionid]
            # 根据动作值函数更新策略
            state_visited = {}
            for _, [stateid, actionid, reward] in enumerate(episode):
                if stateid not in state_visited:
                    state_visited[stateid] = None
                    for actionid in range(len(self.actions)):
                        self.policy[stateid, actionid] = self.epsilon / len(self.actions)
                    actionid = max(enumerate(list(self.action_function[stateid,:])), \
                                   key=lambda x: x[1])[0]
                    self.policy[stateid, actionid] = 1.0 - self.epsilon + \
                        self.epsilon / len(self.actions)
            # 如果本次走到终点，则保存图片
            if episode[-1][2] == 1000:
                # 获得保存图片的序号
                if position_str not in pic_dict:
                    pic_dict[position_str] = 0
                pic_dict[position_str] += 1
                path = self.path + '_' + position_str + '.jpg'
                self.env.render(is_save=True, path=path)
            n_iter += 1
            
    def _sample_episode(self, policy, stateid):
        action_list = []
        for actionid in range(len(self.actions)):
            action_list.extend([actionid] * int(policy[stateid, actionid] * 100))
        actionid = random.choice(action_list)
        
        return actionid
    
    def _print_episode(self, episode):
        print 'episode start %s' % ('='*20)
        for stateid, actionid, reward in episode:
            print self.states[stateid], self.actions[actionid], reward
        print 'episode end %s' % ('='*20)
        
        
mc = MonteCarlo(path='experiments/on_policy/final')
# mc.monte_carlo_control_first_visit_exploring_start()
mc.monte_carlo_control_first_visit_on_policy()