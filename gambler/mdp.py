# -*- encoding = utf8 -*-
import time
import copy

from environment.gambler import Environment
import matplotlib.pyplot as plt


class MDP:
    
    def __init__(self, p=0.4):
        self.env = Environment(p=p)
        self.p = p
        self.gamma = 1.0
        self.win_state = 100
        self.states = range(1, self.win_state)
        self.value_function = [0] * len(self.states)
            
    def _get_valid_actions(self, state):
        return range(1, min(state, self.win_state-state)+1)
        
    def value_iteration(self):
        n_iter = 0
        while True:
            n_iter += 1
            last_value_function = copy.deepcopy(self.value_function)
            print n_iter, list(enumerate(last_value_function))
            new_value_function = [0] * len(self.states)
            for state in self.states:
                max_value = 0
                for action in self._get_valid_actions(state):
                    if state-action <= 0:
                        lose_state = 0
                        lose_reward = 0
                    else:
                        lose_state = self.value_function[state-action-1]
                        lose_reward = 0
                    if state+action >= self.win_state:
                        win_state = 0
                        win_reward = 1
                    else:
                        win_state = self.value_function[state+action-1]
                        win_reward = 0
                    reward = (1-self.p) * (lose_reward + self.gamma * lose_state) + \
                        self.p * (win_reward + self.gamma * win_state)
                    if reward > max_value:
                        max_value = reward
                        max_action = action
                new_value_function[state-1] = max_value
            self.value_function = new_value_function
            # judge exit
            if self._can_exit(last_value_function, self.value_function):
                break
        # get policy
        self.policy = []
        for state in self.states:
            max_value = 0
            for action in self._get_valid_actions(state):
                if state-action <= 0:
                    lose_state = 0
                    lose_reward = 0
                else:
                    lose_state = self.value_function[state-action-1]
                    lose_reward = 0
                if state+action >= self.win_state:
                    win_state = 0
                    win_reward = 1
                else:
                    win_state = self.value_function[state+action-1]
                    win_reward = 0
                reward = (1-self.p) * (lose_reward + self.gamma * lose_state) + \
                    self.p * (win_reward + self.gamma * win_state)
                if reward > max_value:
                    max_value = reward
                    max_action = action
            self.policy.append(max_action)
        print self.policy
        plt.figure()
        plt.plot(self.policy, '.--')
        plt.show()
    
    def _can_exit(self, last_value_function, value_function):
        can_exit = True
        for last, value in zip(last_value_function, value_function):
            if abs(last - value) > 1e-8:
                can_exit = False
                break
        return can_exit
        
        
mdp = MDP(p=0.4)
mdp.value_iteration()