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

    def get_loss(self, states, labels):
        preds = self.get_inference(states, batch_size=self.batch_size)
        loss = tf.nn.l2_loss(labels - preds)
        tf.add_to_collection('losses', loss / self.batch_size)
        avg_loss = tf.add_n(tf.get_collection('losses'))

        return avg_loss

    def get_loss1(self, states, actions, labels):
        action_score = self.get_inference(states, batch_size=self.batch_size)
        actions = tf.cast(actions, dtype=tf.float32)
        preds = tf.reduce_sum(action_score * tf.stop_gradient(actions), axis=1, keep_dims=True)
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
        self.model = self._build_model()
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
                sample = [list(state)]
                actionid = self._sample_action(state)
                sample.append(actionid)
                # get information from evironment
                new_state, reward, done, _ = self.env.step(action=actionid)
                reward = reward if not done else -10
                whole_reward += reward
                sample.append(reward)
                # store memory
                self.memory.append((numpy.reshape(state, [1, self.state_size]), 
                    actionid, reward, numpy.reshape(new_state, [1, self.state_size]), done))
                # update state
                state = copy.deepcopy(new_state)		
		
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
        self.acs = tf.placeholder(
            dtype=tf.float32, shape=[
                None,2],
            name='actions')
        self.labels = tf.placeholder(
        	dtype=tf.float32, shape=[
        		None,1],
        	name='labels')

        # 构建会话和Network对象
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.q_network = Network(
            batch_size=32, state_size=4,
            n_action=2, gamma=self.gamma, name='q_network')

        # 构建优化器
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
        self.avg_loss = self.q_network.get_loss1(self.states, self.acs, self.labels)
        self.optimizer_handle = self.optimizer.minimize(self.avg_loss)
        # 构建预测器
        self.action_score = self.q_network.get_inference(self.states, batch_size=1)
        
        # 模型初始化
        self.sess.run(tf.global_variables_initializer())
        
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.hidden_size, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(self.hidden_size, activation='tanh', kernel_initializer='uniform'))
        model.add(Dense(len(self.actions), activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))
        return model
        
    def _memory_replay(self, size=32):
        batch_size = min(size, len(self.memory))
        batch_data = random.sample(self.memory, batch_size)
        X = numpy.zeros((batch_size, self.state_size))
        Y = numpy.zeros((batch_size, len(self.actions)))
        actions = numpy.zeros((batch_size, len(self.actions)))
        labels = numpy.zeros((batch_size, 1))
        for i in range(batch_size):
            state, action, reward, next_state, done = batch_data[i]
            action_score = self.sess.run(self.action_score, feed_dict={
                self.states: state})
            target = action_score[0]
            if action == 0:
                actions[i,:] = [1, 0]
            else:
                actions[i,:] = [0, 1]
            if done:
                target[action] = reward
                labels[i,0] = reward
            else:
                action_score = self.sess.run(self.action_score, feed_dict={
                    self.states: next_state})
                # target[action] = reward + self.gamma * numpy.amax(self.model.predict(next_state)[0])
                target[action] = reward + self.gamma * numpy.amax(action_score[0])
                labels[i,0] = reward + self.gamma * numpy.amax(action_score[0])
            X[i], Y[i] = state, target
        # print(actions, labels)
        
        [_, avg_loss] = self.sess.run(
            fetches=[self.optimizer_handle, self.avg_loss],
            feed_dict={self.states: X, self.acs:actions, self.labels: labels})
        # print(avg_loss)
        
        # self.model.fit(X, Y, batch_size=batch_size, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_bound:
            self.epsilon *= self.epsilon_decrease
    
    def _sample_action(self, state):
        state = numpy.reshape(state, [1, self.state_size])
        if random.random() < self.epsilon:
            action = random.choice(range(len(self.actions)))
        else:
            action_score = self.sess.run(self.action_score, feed_dict={
                self.states: state})
            # q_value = self.model.predict(state)[0,:]
            action = numpy.argmax(action_score[0])
        return action
	

ql = QLearning(index=0, seed=0)
ql.train()
