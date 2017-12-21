# -*- encoding = utf8 -*-
import random
import sys
import os
import copy
import time
import numpy
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
import tensorflow as tf
from layer.conv_layer import ConvLayer
from layer.pool_layer import PoolLayer
from layer.dense_layer import DenseLayer
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


class Network:
    def __init__(self, batch_size, state_size, n_action, gamma, name):
        self.batch_size = batch_size
        self.state_size = state_size
        self.n_action = n_action
        self.gamma = gamma
        self.layers = []

        # 网络结构
        print('\n%-10s\t%-20s\t%-20s\t%s' % ('Name', 'Filter', 'Input', 'Output'))

        self.dense_layer1 = DenseLayer(
            input_shape=(None, self.state_size),
            hidden_dim=64, activation='tanh', dropout=False,
            keep_prob=None, batch_normal=False, weight_decay=None, name='%s_dense1' % (name))
        self.layers.append(self.dense_layer1)
        
        self.dense_layer2 = DenseLayer(
            input_shape=(None, 64),
            hidden_dim=128, activation='tanh', dropout=False,
            keep_prob=None, batch_normal=False, weight_decay=None, name='%s_dense2' % (name))
        self.layers.append(self.dense_layer2)
        
        self.dense_layer3 = DenseLayer(
            input_shape=(None, 128),
            hidden_dim=self.n_action, activation='none', dropout=False,
            keep_prob=None, batch_normal=False, weight_decay=None, name='%s_dense3' % (name))
        self.layers.append(self.dense_layer3)
        
        print('')
        sys.stdout.flush()
    
    def get_inference(self, states, batch_size=1):
        # 数据流
        hidden_dense1 = self.dense_layer1.get_output(input=states)
        hidden_dense2 = self.dense_layer2.get_output(input=hidden_dense1)
        hidden_dense3 = self.dense_layer3.get_output(input=hidden_dense2)
        
        return hidden_dense3

    def cal_labels(self, next_states, rewards, is_terminals):
        next_action_score = self.get_inference(next_states, batch_size=self.batch_size)
        max_action_score = tf.reduce_max(next_action_score, axis=1, keep_dims=True)
        labels = tf.stop_gradient(rewards + self.gamma * max_action_score * is_terminals)
        
        return labels

    def get_loss(self, states, actions, labels):
        action_score = self.get_inference(states, batch_size=self.batch_size)
        preds = tf.reduce_sum(action_score * actions, axis=1, keep_dims=True)
        loss = tf.nn.l2_loss(labels - preds)
        tf.add_to_collection('losses', loss / self.batch_size)
        avg_loss = tf.add_n(tf.get_collection('losses'))

        return avg_loss


class QLearning:
    
    def __init__(self, index=0, seed=0):
        self.env = gym.make('CartPole-v0')
        self.index = index
        # init variable
        self.actions = ['left', 'right']
        self.trajectory_list = []
        self.memory = []
        self.memory_size = 32
        # init params
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_bound = 0.01
        self.epsilon_decrease = 0.99
        # init model params
        self.state_size = 4
        self.hidden_size = 20
        self.learning_rate = 0.01
        self.init_q_network()

    def train(self, iteration=1000):
        start_time = time.time()
        n_iter = 0
        while True:
            n_iter += 1
            # initial state
            state = self.env.reset()
            # initial gone
            done, trajectory = False, []
            whole_reward = 0
            while not done:
                # render and observe
                if random.random() < self.epsilon:
                    action = 0 if random.random() < 0.5 else 1
                else:
                    action_score = self.sess.run(self.action_score, feed_dict={
                        self.states: numpy.reshape(state, [1, self.state_size])})
                    action = numpy.argmax(action_score[0])

                # get information from evironment
                next_state, reward, done, _ = self.env.step(action=action)
                reward = reward if not done else -10
                whole_reward += reward

                # store memory
                self.memory.append({
                    'state': numpy.reshape(state, [1, self.state_size]), 
                    'action': action, 
                    'reward': reward,
                    'next_state': numpy.reshape(next_state, [1, self.state_size]), 
                    'is_end': done})
                state = copy.deepcopy(next_state)
		
                # memory replay
                self._memory_replay(size=self.memory_size)

            print('@iter: %i, score: %i, epsilon: %.2f' % (n_iter, \
                whole_reward, self.epsilon))
            self.trajectory_list.append(trajectory)

    def init_q_network(self):
    	# 创建placeholder
        self.states = tf.placeholder(
            dtype=tf.float32, shape=[
                None, 4],
            name='states')
        self.next_states = tf.placeholder(
            dtype=tf.float32, shape=[
                None, 4],
            name='next_states')
        self.acs = tf.placeholder(
            dtype=tf.float32, shape=[
                None,2],
            name='actions')
        self.rewards = tf.placeholder(
            dtype=tf.float32, shape=[
                None,1],
            name='rewards')
        self.labels = tf.placeholder(
        	dtype=tf.float32, shape=[
        		None,1],
        	name='labels')
        self.is_ends = tf.placeholder(
            dtype=tf.float32, shape=[
                None,1],
            name='is_ends')

        # 构建会话和Network对象
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.q_network = Network(
            batch_size=32, state_size=4,
            n_action=2, gamma=self.gamma, name='q_network')

        # 构建优化器
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
        self.temp_labels = self.q_network.cal_labels(self.next_states, self.rewards, self.is_ends)
        self.avg_loss = self.q_network.get_loss(self.states, self.acs, self.temp_labels)
        self.optimizer_handle = self.optimizer.minimize(self.avg_loss)
        # 构建预测器
        self.action_score = self.q_network.get_inference(self.states, batch_size=1)
        
        # 模型初始化
        self.sess.run(tf.global_variables_initializer())
        
    def _memory_replay(self, size=32):
        batch_size = min(size, len(self.memory))
        states = numpy.zeros((batch_size, self.state_size))
        next_states = numpy.zeros((batch_size, self.state_size))
        actions = numpy.zeros((batch_size, len(self.actions)))
        rewards = numpy.zeros((batch_size, 1)) 
        is_ends = numpy.zeros((batch_size, 1))
        for i in range(batch_size):
            index = random.randint(0, len(self.memory)-1)
            state = self.memory[index]['state']
            action = self.memory[index]['action']
            reward = self.memory[index]['reward']
            next_state = self.memory[index]['next_state']
            is_end = self.memory[index]['is_end']

            states[i] = self.memory[index]['state']
            next_states[i] = self.memory[index]['next_state']
            actions[i,:] = [1, 0] if self.memory[index]['action'] == 0 else [0, 1]
            is_ends[i,:] = [0.0] if self.memory[index]['is_end'] else [1.0]
            rewards[i,:] = [self.memory[index]['reward']]
        
        [_, avg_loss] = self.sess.run(
            fetches=[self.optimizer_handle, self.avg_loss],
            feed_dict={self.states: states, self.acs:actions, self.next_states: next_states,
                self.rewards: rewards, self.is_ends: is_ends})
        
        if self.epsilon > self.epsilon_bound:
            self.epsilon *= self.epsilon_decrease
	

ql = QLearning(index=0, seed=0)
ql.train()
