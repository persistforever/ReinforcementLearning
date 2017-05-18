# -*- coding: utf8 -*-
import time
import copy
import numpy

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
            if player_state >= 20 and player_state != 22:
                policy[stateid][0] = 0.0
                policy[stateid][1] = 1.0
            else:
                policy[stateid][0] = 1.0
                policy[stateid][1] = 0.0
            # print dealer_state, player_state, policy[stateid]
        
        return policy
        
    def policy_estimating(self):
        n_iter = 0
        value_functions = []
        while True:
            dealer_showing, player_state, reward, done = self.env.reset()
            stateid = self.states_dict['%s#%i' % (dealer_showing, player_state)]
            if done:
                episode = [[stateid, 1, reward]]
            else:
                episode = []
            # 根据当前策略生成轨迹
            while not done:
                actionid = self._sample_episode(self.policy, stateid)
                player_state, reward, done = self.env.step(action=self.actions[actionid])
                stateid = self.states_dict['%s#%i' % (dealer_showing, player_state)]
                # print new_stateid, reward, done
                episode.append([stateid, actionid, reward])
            # 根据轨迹更新状态值函数
            state_visited = {}
            for i, [stateid, actionid, reward] in enumerate(episode):
                # print dealer_state, self.states[stateid], self.actions[actionid], reward
                if stateid not in state_visited:
                    state_visited[stateid] = None
                    sum_reward = sum([episode[j][2] for j in range(i+1, len(episode))])
                    self.value_count[stateid] += 1.0
                    # print self.states[stateid], sum_reward
                    self.value_function[stateid] += (sum_reward - self.value_function[stateid]) \
                        / self.value_count[stateid]
            if n_iter % 100000 == 0:
                value_functions.append(list(self.value_function))
                print list(self.value_function)
                self._plot_value_function(list(self.value_function), n_iter)
            n_iter += 1
            
    def _sample_episode(self, policy, stateid):
        if stateid < 12:
            actionid = 0
        else:
            actionid = max(enumerate(policy[stateid]), key=lambda x: x[1])[0]
        
        return actionid
        
    def _plot_value_function(self, value_functions, n_iter):
        value_matrix = numpy.zeros((10, 10), dtype='float')
        for stateid in range(len(self.states)):
            dealer_showing, player_state = self.states[stateid].split('#')
            dealer_showing = 0 if dealer_showing == 'A' else int(dealer_showing)-1
            player_state = int(player_state)
            if player_state >= 12 and player_state < 22:
                value_matrix[dealer_showing, player_state-12] = value_functions[stateid]
        fig = plt.figure()
        ax = Axes3D(fig)
        X, Y = numpy.meshgrid(range(10), range(10))
        ax.plot_surface(Y, X, value_matrix, rstride=1, cstride=1, cmap='coolwarm')
        ax.set_title('value function in iteration %i' % n_iter)
        ax.set_xlabel('dealer showing')
        ax.set_ylabel('player sum')
        ax.set_zlabel('value function')
        plt.show()
        
        
mc = MonteCarlo()
mc.policy_estimating()