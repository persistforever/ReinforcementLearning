# -*- coding: utf-8 -*-
# Author: Ronniecao
import copy
import numpy
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MDP:
    
    def __init__(self):
        self.max_car = 20
        self.states, self.states_dict = self._init_states(max_car=self.max_car)
        self.actions = range(-5, 6, 1)
        self.gamma = 0.9
        self.const_rental = False
        self.const_return = True
    
    def _init_states(self, max_car):
        states, states_dict, n = [], {}, 0
        for i in range(max_car+1):
            for j in range(max_car+1):
                state = '%i#%i' % (i, j)
                states.append(state)
                if state not in states_dict:
                    states_dict[state] = n
                n += 1
        
        return states, states_dict
        
    def policy_iteration(self):
        # 初始化值函数和策略
        self.value_function = numpy.zeros((len(self.states),), dtype='float')
        self.policy = numpy.zeros((len(self.states), len(self.actions)), dtype='float')
        for stateid in range(len(self.states)):
            for actionid in range(len(self.actions)):
                if self.actions[actionid] == 0:
                    self.policy[stateid][actionid] = 1.0
        n_iter = 0
        while True:
            # 策略评估
            self._plot_policy(self.policy, n_iter)
            self._plot_value_function(self.value_function, n_iter)
            while True:
                last_value_function = copy.deepcopy(self.value_function)
                new_value_function = numpy.zeros((len(self.states),), dtype='float')
                for stateid in range(len(self.states)):
                    actionid = self._get_action_from_policy(self.states[stateid], \
                                                            self.policy[stateid])
                    transition, reward = self._get_transition_reward(self.states[stateid], \
                                                                     self.actions[actionid])
                    for new_stateid in range(len(self.states)):
                        p = transition[new_stateid]
                        r = reward[new_stateid]
                        new_value_function[stateid] += p * \
                            (r + self.gamma * self.value_function[new_stateid])
                self.value_function = new_value_function
                if self._can_exit(last_value_function, self.value_function):
                    break
            print 'iter%i: finish policy evaluation' % n_iter
            # 策略提升
            last_policy = copy.deepcopy(self.policy)
            for stateid in range(len(self.states)):
                valid_actions = self._get_valid_actions(self.states[stateid])
                max_value = 0.0
                for actionid in range(len(self.actions)):
                    if self.actions[actionid] in valid_actions:
                        action_value = 0.0
                        transition, reward = self._get_transition_reward(self.states[stateid], \
                                                                         self.actions[actionid])
                        for new_stateid in range(len(self.states)):
                            p = transition[new_stateid]
                            r = reward[new_stateid]
                            action_value += p * (r + self.gamma * \
                                                 self.value_function[new_stateid])
                        if action_value >= max_value:
                            max_action = actionid
                            max_value = action_value
                for actionid in range(len(self.actions)):
                    self.policy[stateid, actionid] = 0.0
                self.policy[stateid, max_action] = 1.0
            print 'iter%i: finish policy improvement' % n_iter
            n_iter += 1
            if self._can_exit_policy(last_policy, self.policy):
                break
        self._plot_policy(self.policy, n_iter)
        self._plot_value_function(self.value_function, n_iter)
                
    def _get_action_from_policy(self, state, policy):
        valid_actions = self._get_valid_actions(state)
        max_prob = 0.0
        for actionid in range(len(self.actions)):
            if self.actions[actionid] in valid_actions:
                if policy[actionid] >= max_prob:
                    max_prob = policy[actionid]
                    max_action = actionid
        
        return max_action
            
    def _get_valid_actions(self, state):
        state = [int(t) for t in state.split('#')]
        
        return range(-min(self.max_car-state[0], state[1], 5), \
                     min(self.max_car-state[1], state[0], 5)+1, 1)
    
    def _poisson(self, n, lam):
        self.poissonBackup = {}
        key = '%i#%i' % (n, lam)
        if key not in self.poissonBackup.keys():
            self.poissonBackup[key] = math.exp(-lam) * pow(lam, n) / math.factorial(n)
        return self.poissonBackup[key]
    
    def _get_transition_reward(self, state, action):
        state = [int(t) for t in state.split('#')]
        state[0] -= action
        state[1] += action
        transition = numpy.zeros((len(self.states), ), dtype='float')
        reward = numpy.zeros((len(self.states), ), dtype='float')
        if self.const_rental:
            rentala, rentalb = min(state[0], 3), min(state[1], 4)
            temp_state = [state[0] - rentala, state[1] - rentalb]
            new_state = [temp_state[0] + min(3, rentala), \
                         temp_state[1] + min(2, rentalb)]
            new_stateid = self.states_dict['%i#%i' % (new_state[0], new_state[1])]
            transition[new_stateid] = 1.0
            reward[new_stateid] = action * (-2) + (rentala + rentalb) * 10
        else:
            for rentala in range(0, state[0]+1):
                for rentalb in range(0, state[1]+1):
                    temp_state = [state[0] - rentala, state[1] - rentalb]
                    if self.const_return:
                        new_state = [temp_state[0] + min(3, rentala), \
                                     temp_state[1] + min(2, rentalb)]
                        new_stateid = self.states_dict['%i#%i' % (new_state[0], new_state[1])]
                        transition[new_stateid] = self._poisson(rentala, 3) * \
                            self._poisson(rentalb, 4)
                    else:
                        for returna in range(0, self.max_car-temp_state[0]+1):
                            for returnb in range(0, self.max_car-temp_state[1]+1):
                                new_state = [temp_state[0] + returna, \
                                             temp_state[1] + returnb]
                                new_stateid = self.states_dict['%i#%i' % (new_state[0], \
                                                                          new_state[1])]
                                transition[new_stateid] = self._poisson(rentala, 3) * \
                                    self._poisson(rentalb, 4) * self._poisson(returna, 3) * \
                                    self._poisson(returnb, 2)
                    reward[new_stateid] = (action * (-2) + (rentala + rentalb) * 10)
                    
        return transition, reward
        
    def _plot_value_function(self, value_function, n_iter):
        value_matrix = numpy.zeros((self.max_car+1, self.max_car+1), dtype='float')
        for stateid in range(len(self.states)):
            state = [int(t) for t in self.states[stateid].split('#')]
            value_matrix[state[0], state[1]] = value_function[stateid]
        fig = plt.figure()
        ax = Axes3D(fig)
        X, Y = numpy.meshgrid(range(self.max_car+1), range(self.max_car+1))
        ax.plot_surface(Y, X, value_matrix, rstride=1, cstride=1, cmap='coolwarm')
        ax.set_title('value function in iteration %i' % n_iter)
        ax.set_xlabel('#cars at A')
        ax.set_ylabel('#cars at B')
        ax.set_zlabel('value function')
        # plt.show()
        fig.savefig('experiments/value%i' % n_iter)
        
    def _plot_policy(self, policy, n_iter):
        policy_matrix = numpy.zeros((self.max_car+1, self.max_car+1), dtype='float')
        for stateid in range(len(self.states)):
            state = [int(t) for t in self.states[stateid].split('#')]
            for actionid in range(len(self.actions)):
                if policy[stateid, actionid] == 1.0:
                    policy_matrix[state[0], state[1]] = self.actions[actionid]
        fig = plt.figure()
        plt.contourf(range(self.max_car+1), range(self.max_car+1), policy_matrix, 10, \
                     cmap='coolwarm')
        plt.title('policy in iteration %i' % n_iter)
        plt.xlabel('#cars at A')
        plt.ylabel('#cars at B')
        # plt.show()
        fig.savefig('experiments/policy%i' % n_iter)
            
    def _can_exit(self, last_value_function, value_function):
        can_exit = True
        for last, value in zip(last_value_function, value_function):
            if abs(last - value) > 1e0:
                can_exit = False
                break
        return can_exit
            
    def _can_exit_policy(self, last_policy, policy):
        can_exit = True
        for stateid in range(len(self.states)):
            for actionid in range(len(self.actions)):
                if last_policy[stateid, actionid] != policy[stateid, actionid]:
                    can_exit = False
                    break
            if not can_exit:
                break
        return can_exit
        
        
mdp = MDP()
mdp.policy_iteration()