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
from environment.flappy import Environment
from layer.conv_layer import ConvLayer
from layer.pool_layer import PoolLayer
from layer.dense_layer import DenseLayer
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


class Network:
    def __init__(self, batch_size, n_history, image_y_size, image_x_size, n_action, gamma, name):
        self.batch_size = batch_size
        self.n_history = n_history
        self.image_y_size = image_y_size
        self.image_x_size = image_x_size
        self.n_action = n_action
        self.gamma = gamma
        self.layers = []

        # 网络结构
        print('\n%-10s\t%-20s\t%-20s\t%s' % ('Name', 'Filter', 'Input', 'Output')) 
        self.conv_layer1 = ConvLayer(
            input_shape=(None, self.image_y_size, self.image_x_size, self.n_history), 
            n_size=8, n_filter=32, stride=4, activation='relu', 
            batch_normal=False, weight_decay=None, name='%s_conv1' % (name))
        self.layers.append(self.conv_layer1)
        
        self.conv_layer2 = ConvLayer(
            input_shape=(None, int(self.image_y_size/4), int(self.image_x_size/4), 32), 
            n_size=4, n_filter=64, stride=2, activation='relu', 
            batch_normal=False, weight_decay=None, name='%s_conv2' % (name))
        self.layers.append(self.conv_layer2)
        
        self.conv_layer3 = ConvLayer(
            input_shape=(None, int(self.image_y_size/8), int(self.image_x_size/8), 64), 
            n_size=3, n_filter=64, stride=1, activation='relu', 
            batch_normal=False, weight_decay=None, name='%s_conv3' % (name))
        self.layers.append(self.conv_layer3)

        self.dense_layer1 = DenseLayer(
            input_shape=(None, int(self.image_y_size/8) * int(self.image_x_size/8) * 64),
            hidden_dim=512, activation='relu', dropout=False,
            keep_prob=None, batch_normal=False, weight_decay=None, name='%s_dense1' % (name))
        self.layers.append(self.dense_layer1)
        
        self.dense_layer2 = DenseLayer(
            input_shape=(None, 512),
            hidden_dim=2, activation='none', dropout=False,
            keep_prob=None, batch_normal=False, weight_decay=None, name='%s_dense2' % (name))
        self.layers.append(self.dense_layer2)
        
        print('')
        sys.stdout.flush()
    
    def get_inference(self, images, batch_size=1):
        # 数据流
        hidden_conv1 = self.conv_layer1.get_output(input=images)
        hidden_conv2 = self.conv_layer2.get_output(input=hidden_conv1)
        hidden_conv3 = self.conv_layer3.get_output(input=hidden_conv2)
        hidden_conv3 = tf.reshape(hidden_conv3, shape=(
            batch_size, int(self.image_y_size/8) * int(self.image_x_size/8) * 64))
        hidden_dense1 = self.dense_layer1.get_output(input=hidden_conv3)
        hidden_dense2 = self.dense_layer2.get_output(input=hidden_dense1)
        
        return hidden_dense2

    def cal_labels(self,next_images, rewards, is_terminals):
        next_action_score = self.get_inference(next_images, batch_size=self.batch_size)
        max_action_score = tf.reduce_max(next_action_score, axis=1, keep_dims=True)
        labels = tf.stop_gradient(rewards + self.gamma * max_action_score * is_terminals)
        
        return labels

    def get_loss(self, images, actions, labels):
        action_score = self.get_inference(images, batch_size=self.batch_size)
        preds = tf.reduce_sum(action_score * actions, axis=1, keep_dims=True)
        loss = tf.nn.l2_loss(labels - preds)
        tf.add_to_collection('losses', loss / self.batch_size)
        avg_loss = tf.add_n(tf.get_collection('losses'))

        return avg_loss


class QLearning:
    def __init__(self, is_show=False):
        self.env = Environment(is_show=is_show)
        self.flap_prob = 0.1
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_iters = 600000
        self.epsilon_reduce = 1.0 * (self.epsilon - self.epsilon_min) / self.epsilon_iters
        self.image_queue_maxsize = 5
        self.replay_memory = []
        self.replay_memory_maxsize = 20000
        self.batch_size = 32
        self.n_history = self.image_queue_maxsize
        self.image_y_size = 80
        self.image_x_size = 80
        self.action_options = ['flap', 'noflap']
        self.n_action = 2
        self.gamma = 0.95
        self.n_before = 3000
        self.n_update_target = 1000

    def init_replay_memory(self):
        n_frame = 0
        while n_frame <= self.n_before:
            init_image = self.env.reset()
            is_end = False
            image_queue = []
            for j in range(self.image_queue_maxsize):
                image_queue.append(copy.deepcopy(init_image)) 
            n_frame += 1
            while not is_end:
                rnd = random.random()
                action = 0 if rnd < self.flap_prob else 1
                next_image, reward, is_end = self.env.render(self.action_options[action])
                state = self._extract_feature(image_queue)
                del image_queue[0]
                image_queue.append(copy.deepcopy(next_image))
                next_state = self._extract_feature(image_queue)
                self.replay_memory.append({
                    'state': state, 'action': action, 'reward': reward, 
                    'is_end': is_end, 'next_state': next_state})
                n_frame += 1

    def init_q_network(self):
        # 创建placeholder
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[
                None, self.image_y_size, self.image_x_size, self.n_history],
            name='images')
        self.next_images = tf.placeholder(
            dtype=tf.float32, shape=[
                None, self.image_y_size, self.image_x_size, self.n_history],
            name='next_images')
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
        self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step')
        
        # 构建会话和Network对象
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.q_network = Network(
            batch_size=self.batch_size, n_history=self.image_queue_maxsize, 
            image_y_size=self.image_y_size, image_x_size=self.image_x_size,
            n_action=self.n_action, gamma=self.gamma, name='q_network')
        self.target_network = Network(
            batch_size=self.batch_size, n_history=self.image_queue_maxsize, 
            image_y_size=self.image_y_size, image_x_size=self.image_x_size,
            n_action=self.n_action, gamma=self.gamma, name='target_network')
        
        # 构建优化器
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-6, decay=0.9, momentum=0.95)
        self.temp_labels = self.target_network.cal_labels(self.next_images, self.rewards, self.is_terminals)
        self.avg_loss = self.q_network.get_loss(self.images, self.actions, self.temp_labels)
        self.optimizer_handle = self.optimizer.minimize(self.avg_loss, global_step=self.global_step)
        # 构建预测器
        self.action_score = self.q_network.get_inference(self.images, batch_size=1)
        
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
        max_total_reward = 0
        for n_episode in range(n_episodes):
            # 用q_network更新target_network的参数
            if n_frame % self.n_update_target == 0:
                self._update_target(self.q_network, self.target_network)
            
            # 初始化trajectory
            init_image = self.env.reset()
            image_queue = []
            for i in range(self.image_queue_maxsize):
                image_queue.append(copy.deepcopy(init_image)) 
            total_reward = 0.0
            is_end = False
            n_step = 0
            n_frame += 1
            
            while not is_end:
                state = self._extract_feature(image_queue)
                # 采样action
                if random.random() < self.epsilon:
                    action = 0 if random.random() < self.flap_prob else 1
                else:
                    state_np = numpy.array([state], dtype='float32')
                    action_score = self.sess.run(
                        fetches=[self.action_score], 
                        feed_dict={self.images: state_np})
                    action = 0 if numpy.argmax(action_score[0]) == 0 else 1
                
                # 更新env
                next_image, reward, is_end = self.env.render(self.action_options[action])
                self.epsilon = max(self.epsilon - self.epsilon_reduce, self.epsilon_min)
                total_reward += reward
                n_step += 1
                n_frame += 1
                del image_queue[0]
                image_queue.append(copy.deepcopy(next_image))
                next_state = self._extract_feature(image_queue)
                self.replay_memory.append({
                    'state': state, 'action': action, 'reward': reward,
                    'is_end': is_end, 'next_state': next_state})
                if len(self.replay_memory) > self.replay_memory_maxsize:
                    del self.replay_memory[0]
                
                # 随机从replay_memory中取出1个batch
                batch_images = numpy.zeros((
                    self.batch_size, self.image_y_size, self.image_x_size, 
                    self.n_history), dtype='float32')
                batch_next_images = numpy.zeros((
                    self.batch_size, self.image_y_size, self.image_x_size, 
                    self.n_history), dtype='float32')
                batch_actions = numpy.zeros((
                    self.batch_size, self.n_action), dtype='float32')
                batch_rewards = numpy.zeros((
                    self.batch_size, 1), dtype='float32')
                batch_is_terminals = numpy.zeros((
                    self.batch_size, 1), dtype='float32')
                for j in range(self.batch_size):
                    index = random.randint(0, len(self.replay_memory)-1)
                    item = self.replay_memory[index]
                    batch_images[j,:,:,:] = item['state']
                    batch_next_images[j,:,:,:] = item['next_state']
                    batch_actions[j,:] = [1.0, 0.0] if item['action'] == 0 else [0.0, 1.0]
                    batch_rewards[j,:] = [item['reward']]
                    batch_is_terminals[j,:] = [0.0] if item['is_end'] else [1.0]
                [_, avg_loss] = self.sess.run(
                    fetches=[self.optimizer_handle, self.avg_loss],
                    feed_dict={
                        self.images: batch_images, self.next_images: batch_next_images,
                        self.actions: batch_actions, self.rewards: batch_rewards, 
                        self.is_terminals: batch_is_terminals})
            
            print('[%d] avg_loss: %.6f, total_reward: %.1f, n_score: %d' % (
                n_episode, avg_loss, total_reward, self.env.n_score))
            
            # trajectory结束后保存模型
            total_rewards = self.valid()
            if total_rewards >= max_total_reward:
                model_path = os.path.join(backup_dir, 'model_best.ckpt')
                self.saver.save(self.sess, model_path)
                max_total_reward = total_rewards

    def valid(self):
        total_rewards = 0
        n_iters = 50
        for i in range(n_iters):
            init_image = self.env.reset()
            image_queue = []
            for i in range(self.image_queue_maxsize):
                image_queue.append(copy.deepcopy(init_image))
            total_reward = 0.0
            is_end = False
            while not is_end:
                state = self._extract_feature(image_queue)
                state_np = numpy.array([state], dtype='float32')
                action_score = self.sess.run(
                    fetches=[self.action_score],
                    feed_dict={self.images: state_np})
                action = 0 if numpy.argmax(action_score[0]) == 0 else 1
                next_image, reward, is_end = self.env.render(self.action_options[action])
                total_reward += reward
                del image_queue[0]
                image_queue.append(copy.deepcopy(next_image))
            total_rewards += total_reward

        return 1.0 * total_rewards / n_iters

    def test(self, model_path):
        self.init_q_network()
        self.saver.restore(self.sess, model_path)

        for i in range(100):
            init_image = self.env.reset()
            image_queue = []
            for i in range(self.image_queue_maxsize):
                image_queue.append(copy.deepcopy(init_image))
            total_reward = 0.0
            is_end = False
            while not is_end:
                state = self._extract_feature(image_queue)
                state_np = numpy.array([state], dtype='float32')
                action_score = self.sess.run(
                    fetches=[self.action_score], 
                    feed_dict={self.images: state_np})
                action = 0 if numpy.argmax(action_score[0]) == 0 else 1
                next_image, reward, is_end = self.env.render(self.action_options[action])
                total_reward += reward
                del image_queue[0]
                image_queue.append(copy.deepcopy(next_image))
            print('total reward: %d' % (total_reward))

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
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        main_dir = '/home/caory/github/ReinforcementLearning'
        qlearning = QLearning(is_show=False)
        qlearning.train(n_episodes=50000, 
            backup_dir=os.path.join(main_dir, 'backup', 'flappy-v2'))
    elif method == 'test':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        main_dir = 'D://Github/ReinforcementLearning'
        qlearning = QLearning(is_show=True)
        qlearning.test(
            model_path=os.path.join(main_dir, 'backup', 'flappy', 'model_6000.ckpt'))