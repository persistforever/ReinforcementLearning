# -*- coding: utf8 -*-
import time
import copy
import numpy
import random

from environment.blackjack import Environment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MonteCarlo:
    
    def __init__(self):
        self.env = Environment()
        self.gamma = 1.0
        self.states, self.states_dict = self._init_states()
        self.actions = ['hit', 'stick']
        self.value_function = numpy.zeros((len(self.states), ), dtype='float')
        self.value_count = numpy.zeros((len(self.states), ), dtype='float')
        self.action_function = numpy.zeros((len(self.states), len(self.actions)), dtype='float')
        self.action_count = numpy.zeros((len(self.states), len(self.actions)), dtype='float')
        self.policy = self._init_policy()
        self.episodes = []
    
    def _init_states(self):
        # states [0,1,2,3,...,20,21,bust] number of states is 22
        cards = ['A'] + [str(t) for t in range(2,11)]
        states, states_dict, n = [], {}, 0
        for card in cards:
            for j in range(0, 22+1):
                state = '%s#%i' % (card, j)
                states.append(state)
                states_dict[state] = n
                n += 1
        
        return states, states_dict
    
    def _init_policy(self):
        policy = numpy.zeros((len(self.states), len(self.actions)), dtype='float')
        for stateid in range(len(self.states)):
            player_state = int(self.states[stateid].split('#')[1])
            if player_state >= 20:
                policy[stateid, 0] = 0.0
                policy[stateid, 1] = 1.0
            else:
                policy[stateid, 0] = 1.0
                policy[stateid, 1] = 0.0
        
        return policy
        
    def policy_estimating_first_visit(self):
        n_iter = 0
        while True:
            dealer_showing, player_state, reward, done = self.env.reset()
            stateid = self.states_dict['%s#%i' % (dealer_showing, player_state)]
            episode = []
            if done:
                episode.append([stateid, 1, reward])
            # 根据当前策略生成轨迹
            while not done:
                actionid = self._sample_episode(self.policy, stateid)
                player_state, reward, done = self.env.step(action=self.actions[actionid])
                new_stateid = self.states_dict['%s#%i' % (dealer_showing, player_state)]
                # print new_stateid, reward, done
                episode.append([stateid, actionid, reward])
                stateid = new_stateid
            # self._print_episode(episode)
            # 根据轨迹更新状态值函数
            state_visited = {}
            for stateid, actionid, reward in episode:
                # print dealer_state, self.states[stateid], self.actions[actionid], reward
                if stateid not in state_visited:
                    state_visited[stateid] = None
                    sum_reward = episode[-1][2]
                    self.value_count[stateid] += 1.0
                    self.value_function[stateid] += (sum_reward - self.value_function[stateid]) \
                        / self.value_count[stateid]
            if n_iter % 100000 == 0:
                print list(self.value_function)
                self._plot_value_function(list(self.value_function), n_iter)
            n_iter += 1
        
    def monte_carlo_control_first_visit(self):
        n_iter = 0
        while True:
            if n_iter % 100000 == 0:
                print list(self.value_function)
                self._plot_value_function(list(self.value_function), n_iter)
                self._plot_policy(self.policy, n_iter)
            # 初始化
            dealer_showing, player_state, reward, done = self.env.reset()
            stateid = self.states_dict['%s#%i' % (dealer_showing, player_state)]
            episode = []
            if done:
                episode.append([stateid, 1, reward])
            # 根据当前策略生成轨迹
            while not done:
                actionid = self._sample_episode(self.policy, stateid)
                player_state, reward, done = self.env.step(action=self.actions[actionid])
                new_stateid = self.states_dict['%s#%i' % (dealer_showing, player_state)]
                # print new_stateid, reward, done
                episode.append([stateid, actionid, reward])
                stateid = new_stateid
            # self._print_episode(episode)
            # 根据轨迹更新状态值函数
            state_visited = {}
            state_action_visited = {}
            for stateid, actionid, reward in episode:
                # print dealer_state, self.states[stateid], self.actions[actionid], reward
                if stateid not in state_visited:
                    state_visited[stateid] = None
                    sum_reward = episode[-1][2]
                    self.value_count[stateid] += 1.0
                    self.value_function[stateid] += (sum_reward - self.value_function[stateid]) \
                        / self.value_count[stateid]
                if '%i#%i' % (stateid, actionid) not in state_action_visited:
                    state_action_visited['%i#%i' % (stateid, actionid)] = None
                    sum_reward = episode[-1][2]
                    self.action_count[stateid, actionid] += 1.0
                    self.action_function[stateid, actionid] += (sum_reward - \
                        self.action_function[stateid, actionid]) \
                        / self.action_count[stateid, actionid]
            # 根据动作值函数更新策略
            state_visited = {}
            for _, [stateid, actionid, reward] in enumerate(episode):
                if stateid not in state_visited:
                    state_visited[stateid] = None
                    # bust状态一定执行 stick动作
                    dealer_showing, player_state = self.states[stateid].split('#')
                    if int(player_state) >= 22:
                        self.policy[stateid, 0] = 0.0
                        self.policy[stateid, 1] = 1.0
                    else:
                        for actionid in range(len(self.actions)):
                            self.policy[stateid, actionid] = 0.0
                        # print self.states[stateid], list(self.action_function[stateid,:])
                        actionid = max(enumerate(list(self.action_function[stateid,:])), \
                                       key=lambda x: x[1])[0]
                        self.policy[stateid, actionid] = 1.0
            n_iter += 1
            
    def _sample_episode(self, policy, stateid):
        action_list = []
        for actionid in range(len(self.actions)):
            action_list.extend([actionid] * int(self.policy[stateid, actionid] * 100))
        actionid = random.choice(action_list)
        
        return actionid
    
    def _print_episode(self, episode):
        print 'episode start %s' % ('='*20)
        for stateid, actionid, reward in episode:
            print self.states[stateid], self.actions[actionid], reward
        print 'episode end %s' % ('='*20)
        
    def _plot_value_function(self, value_functions, n_iter):
        value_matrix = numpy.zeros((10, 10), dtype='float')
        for stateid in range(len(self.states)):
            dealer_showing, player_state = self.states[stateid].split('#')
            dealer_showing = 0 if dealer_showing == 'A' else int(dealer_showing)-1
            player_state = int(player_state)
            if player_state >= 12 and player_state < 22:
                value_matrix[player_state-12, dealer_showing] = value_functions[stateid]
        fig = plt.figure()
        ax = Axes3D(fig)
        Y, X = numpy.meshgrid(range(10), range(12,22))
        ax.plot_surface(Y, X, value_matrix, rstride=1, cstride=1, cmap='coolwarm')
        ax.set_title('value function in iteration %i' % n_iter)
        ax.set_xlabel('dealer showing')
        ax.set_ylabel('player sum')
        ax.set_zlabel('value function')
        plt.show()
        
    def _plot_policy(self, policy, n_iter):
        policy_matrix = numpy.zeros((10, 10), dtype='float')
        for stateid in range(len(self.states)):
            dealer_showing, player_state = self.states[stateid].split('#')
            dealer_showing = 0 if dealer_showing == 'A' else int(dealer_showing)-1
            player_state = int(player_state)
            if player_state >= 12 and player_state < 22:
                for actionid in range(len(self.actions)):
                    if policy[stateid, actionid] == 1.0:
                        policy_matrix[player_state-12, dealer_showing] = actionid
        fig = plt.figure()
        print policy_matrix
        plt.contourf(range(10), range(12,22), policy_matrix, 1, cmap='coolwarm', \
                     corner_mask=True)
        plt.title('policy in iteration %i' % n_iter)
        plt.xlabel('dealer showing')
        plt.ylabel('player sum')
        plt.show()
        # fig.savefig('experiments/policy%i' % n_iter)
        
        
mc = MonteCarlo()
# mc.policy_estimating_first_visit()
mc.monte_carlo_control_first_visit()