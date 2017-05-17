# -*- coding: utf8 -*-
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
        self.policy = []
        for _ in range(len(self.states)):
            self.policy.append([1])
            
    def _get_valid_actions(self, state):
        return range(1, min(state, self.win_state-state)+1)
        
    def value_iteration(self):
        n_iter = 0
        value_functions = []
        start_time = time.time()
        while True:
            n_iter += 1
            last_value_function = copy.deepcopy(self.value_function)
            value_functions.append(last_value_function)
            # print n_iter, list(enumerate(last_value_function))
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
            action_function = []
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
                action_function.append([action, reward])
                max_action_value = max(action_function, key=lambda x: x[1])[1]
                max_actions = [action for action, reward in action_function \
                               if reward == max_action_value]
            self.policy.append(max_actions)
        end_time = time.time()
        print 'time consuming %.4f, #iteration %i' % (end_time-start_time, n_iter)
        # print self.policy
        self._plot_value_function(value_functions, show_index=[1,2,4,20])
        self._plot_final_policy(self.policy)
        
    def policy_iteration(self):
        n_iter = 0
        value_functions = []
        start_time = time.time()
        while True:
            # 策略评估
            while True:
                n_iter += 1
                last_value_function = copy.deepcopy(self.value_function)
                value_functions.append(last_value_function)
                # print n_iter, list(enumerate(last_value_function))
                new_value_function = [0] * len(self.states)
                for state in self.states:
                    action = self.policy[state-1][0]
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
                    value = (1-self.p) * (lose_reward + self.gamma * lose_state) + \
                        self.p * (win_reward + self.gamma * win_state)
                    new_value_function[state-1] += value
                self.value_function = new_value_function
                # judge exit
                if self._can_exit(last_value_function, self.value_function):
                    break
            # 策略提升
            last_policy = copy.deepcopy(self.policy)
            new_policy = []
            for state in self.states:
                action_function = []
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
                    action_function.append([action, reward])
                    max_action_value = max(action_function, key=lambda x: x[1])[1]
                    max_actions = [action for action, reward in action_function \
                                   if reward == max_action_value]
                new_policy.append(max_actions)
            self.policy = new_policy
            # print self.policy
            if self._can_exit_policy(last_policy, self.policy):
                break
        end_time = time.time()
        print 'time consuming %.4f, #iteration %i' % (end_time-start_time, n_iter)
        self._plot_value_function(value_functions, show_index=[1,200,470,500])
        self._plot_final_policy(self.policy)
        
    def _plot_value_function(self, value_functions, show_index=[1,2,4,20]):
        colors = ['#FF4040', '#6495ED', '#66CDAA', '#8470FF']
        fig = plt.figure()
        axs = []
        for index, color in zip(show_index, colors):
            axs.append(plt.plot(self.states, value_functions[index], '-', c=color))
        plt.title('value function in different iterations')
        plt.xlabel('states')
        plt.ylabel('value function')
        plt.legend([t[0] for t in axs], ['iter '+str(t) for t in show_index])
        plt.show()
        
    def _plot_final_policy(self, policy):
        state_actions = []
        for state in range(len(policy)):
            # for action in min(policy[state]):
            state_actions.append([state+1, min(policy[state])])
        # print state_actions
        fig = plt.figure()
        plt.bar([t[0] for t in state_actions], [t[1] for t in state_actions], width=0.5, \
                color='#6495ED')
        plt.title('final target policy')
        plt.xlabel('states')
        plt.ylabel('action')
        plt.show()
            
    def _can_exit(self, last_value_function, value_function):
        can_exit = True
        for last, value in zip(last_value_function, value_function):
            if abs(last - value) > 1e-8:
                can_exit = False
                break
        return can_exit
            
    def _can_exit_policy(self, last_policy, policy):
        can_exit = True
        for stateid in range(len(self.states)):
            if last_policy[stateid] != policy[stateid]:
                can_exit = False
                break
        return can_exit
        
        
mdp = MDP(p=0.4)
mdp.policy_iteration()