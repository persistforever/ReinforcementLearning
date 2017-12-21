# -*- coding: utf-8 -*-
# author: ronniecao
# time: 2017/12/17
# description: DQN of flappy bird
import sys
import os
import random
import time
import numpy
import copy
import cv2
import argparse
import tensorflow as tf
import gym
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
    def __init__(self, is_show=False):
        self.env = gym.make('CartPole-v0')
        self.left_prob = 0.5
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_iters = 1000
        self.epsilon_reduce = 1.0 * (self.epsilon - self.epsilon_min) / self.epsilon_iters
        self.replay_memory = []
        self.replay_memory_maxsize = 100
        self.batch_size = 32
        self.state_size = 4
        self.n_action = 2
        self.gamma = 0.9
        self.n_before = 500
        self.n_update_target = 1000

    def init_replay_memory(self):
        n_frame = 0
        while n_frame <= self.n_before:
            state = self.env.reset()
            is_end = False
            n_frame += 1
            while not is_end:
                # self.env.render()
                rnd = random.random()
                action = 0 if rnd < self.left_prob else 1
                next_state, reward, is_end, _ = self.env.step(action)
                reward = -10 if is_end else 1
                self.replay_memory.append({
                    'state': copy.deepcopy(state), 'action': action, 'reward': reward, 
                    'is_end': is_end, 'next_state': copy.deepcopy(next_state)})
                n_frame += 1
            print(n_frame)

    def init_q_network(self):
        # 创建placeholder
        self.states = tf.placeholder(
            dtype=tf.float32, shape=[
                None, self.state_size],
            name='states')
        self.next_states = tf.placeholder(
            dtype=tf.float32, shape=[
                None, self.state_size],
            name='next_states')
        self.actions = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.n_action],
            name='actions')
        self.rewards = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, 1],
            name='rewards')
        self.is_terminals = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, 1],
            name='is_terminals')
        self.labels = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, 1],
            name='labels')
        self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step')
        
        # 构建会话和Network对象
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.q_network = Network(
            batch_size=self.batch_size, state_size=self.state_size,
            n_action=self.n_action, gamma=self.gamma, name='q_network')
        self.target_network = Network(
            batch_size=self.batch_size, state_size=self.state_size,
            n_action=self.n_action, gamma=self.gamma, name='target_network')
        
        # 构建优化器
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
        self.temp_labels = self.q_network.cal_labels(self.next_states, self.rewards, self.is_terminals)
        self.avg_loss = self.q_network.get_loss(self.states, self.actions, self.temp_labels)
        self.optimizer_handle = self.optimizer.minimize(self.avg_loss, global_step=self.global_step)
        # 构建预测器
        self.action_score = self.q_network.get_inference(self.states, batch_size=1)
        
        # 模型保存器
        self.saver = tf.train.Saver(
            var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2, max_to_keep=100)
        # 模型初始化
        self.sess.run(tf.global_variables_initializer())

    def train(self, n_episodes, backup_dir):        
        self.init_replay_memory()
        self.init_q_network()

        print('\nstart training ...\n')
        n_frame = 0
        for n_episode in range(n_episodes):
            # 用q_network更新target_network的参数
            if n_frame % self.n_update_target == 0:
                self._update_target(self.q_network, self.target_network)
            
            # 初始化trajectory
            state = self.env.reset()
            # self.env.render()
            total_reward = 0.0
            is_end = False
            n_step = 0
            n_frame += 1
            
            while not is_end:
                # 采样action
                if random.random() < self.epsilon:
                    action = 0 if random.random() < self.left_prob else 1
                else:
                    state_np = numpy.array([state], dtype='float32')
                    action_score = self.sess.run(
                        fetches=[self.action_score], 
                        feed_dict={self.states: state_np})
                    # print(action_score[0], numpy.argmax(action_score[0]))
                    action = 0 if numpy.argmax(action_score[0]) == 0 else 1
                
                # 更新env
                next_state, reward, is_end, _ = self.env.step(action)
                reward = -10 if is_end else 1
                # self.env.render()
                self.epsilon = max(self.epsilon - self.epsilon_reduce, self.epsilon_min)
                total_reward += reward
                n_step += 1
                n_frame += 1
                self.replay_memory.append({
                    'state': copy.deepcopy(state), 'action': action, 'reward': reward,
                    'is_end': is_end, 'next_state': copy.deepcopy(next_state)})
                if len(self.replay_memory) > self.replay_memory_maxsize:
                    del self.replay_memory[0]

                state = copy.deepcopy(next_state)
                
                # 随机从replay_memory中取出1个batch
                batch_states = numpy.zeros((self.batch_size, self.state_size))
                batch_next_states = numpy.zeros((self.batch_size, self.state_size))
                batch_actions = numpy.zeros((self.batch_size, 2))
                rewards = numpy.zeros((self.batch_size, 1)) 
                batch_rewards = numpy.zeros((self.batch_size, 1))
                batch_is_terminals = numpy.zeros((self.batch_size, 1))
                for j in range(self.batch_size):
                    index = random.randint(0, len(self.replay_memory)-1)
                    item = self.replay_memory[index]
                    batch_states[j,:] = item['state']
                    batch_next_states[j,:] = item['next_state']
                    batch_actions[j,:] = [1.0, 0.0] if item['action'] == 0 else [0.0, 1.0]
                    batch_rewards[j,:] = item['reward']
                    batch_is_terminals[j,:] = [0.0] if item['is_end'] else [1.0]
                [_, avg_loss] = self.sess.run(
                    fetches=[self.optimizer_handle, self.avg_loss],
                    feed_dict={
                        self.states: batch_states, self.next_states: batch_next_states,
                        self.actions: batch_actions, self.rewards: batch_rewards, 
                        self.is_terminals: batch_is_terminals})
            
            print('[%d] avg_loss: %.6f, total_reward: %.1f, n_frame: %d' % (
                n_episode, avg_loss, total_reward, n_frame))
            
            # trajectory结束后保存模型
            if (n_episode <= 1000 and n_episode % 100 == 0) or \
                (1000 < n_episode <= 10000 and n_episode % 500 == 0) or \
                (10000 < n_episode <= 50000 and n_episode % 2000 == 0):
                
                model_path = os.path.join(backup_dir, 'model_%d.ckpt' % (n_episode))
                self.saver.save(self.sess, model_path)

    def test(self, model_path):
        self.init_q_network()
        self.saver.restore(self.sess, model_path)

        init_image = self.env.reset()
        image_queue = [init_image, init_image, init_image, init_image]
        is_end = False
        while not is_end:
            state = self._extract_feature(image_queue)
            state_np = numpy.array([state], dtype='float32')
            max_action, action_prob = self.sess.run(
                fetches=[self.max_action, self.action_prob], 
                feed_dict={self.images: state_np})
            print(max_action, action_prob)
            action = 'flap' if max_action[0] == 0 else 'noflap'
            next_image, reward, is_end = self.env.render(action)
            image_queue.pop(0)
            image_queue.append(next_image)

    def _extract_feature(self, images):
        features = []
        for image in images:
            new_image = cv2.resize(image, (self.image_x_size, self.image_y_size))
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
            new_image = numpy.array(new_image/255.0, dtype='float32')
            new_image = numpy.reshape(new_image, (self.image_y_size, self.image_x_size, 1))
            features.append(new_image)
        feature = numpy.concatenate(features, axis=2)
        
        return feature

    def _update_target(self, q_network, target_network):
        for i in range(len(q_network.layers)):
            for j in range(len(q_network.layers[i].variables)):
                target_network.layers[i].variables[j].assign(
                    q_network.layers[i].variables[j])
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parsing')
    parser.add_argument('-method')
    parser.add_argument('-gpus')
    arg = parser.parse_args()
    method = arg.method
    
    if method == 'train':
        gpus = arg.gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        main_dir = 'D://Github/ReinforcementLearning'
        qlearning = QLearning(is_show=False)
        qlearning.train(n_episodes=50000, 
            backup_dir=os.path.join(main_dir, 'backup', 'cartpole'))
    elif method == 'test':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        main_dir = 'D://Github/ReinforcementLearning'
        qlearning = QLearning(is_show=True)
        qlearning.test(
            model_path=os.path.join(main_dir, 'backup', 'cartpole', 'model_150.ckpt'))
