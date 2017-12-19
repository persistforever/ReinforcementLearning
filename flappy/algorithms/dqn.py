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
    def __init__(self, batch_size, n_history, image_y_size, image_x_size, n_action, gamma):
        self.batch_size = batch_size
        self.n_history = n_history
        self.image_y_size = image_y_size
        self.image_x_size = image_x_size
        self.n_action = n_action
        self.gamma = gamma

        # 网络结构
        print('\n%-10s\t%-20s\t%-20s\t%s' % ('Name', 'Filter', 'Input', 'Output')) 
        self.conv_layer1 = ConvLayer(
            input_shape=(None, self.image_y_size, self.image_x_size, self.n_history), 
            n_size=8, n_filter=32, stride=4, activation='relu', 
            batch_normal=False, weight_decay=None, name='conv1')
        
        self.conv_layer2 = ConvLayer(
            input_shape=(None, int(self.image_y_size/4), int(self.image_x_size/4), 32), 
            n_size=4, n_filter=64, stride=2, activation='relu', 
            batch_normal=False, weight_decay=None, name='conv2')
        
        self.conv_layer3 = ConvLayer(
            input_shape=(None, int(self.image_y_size/8), int(self.image_x_size/8), 64), 
            n_size=4, n_filter=64, stride=1, activation='relu', 
            batch_normal=False, weight_decay=None, name='conv3')

        self.dense_layer1 = DenseLayer(
            input_shape=(None, int(self.image_y_size/8) * int(self.image_x_size/8) * 64),
            hidden_dim=512, activation='relu', dropout=False,
            keep_prob=None, batch_normal=False, weight_decay=None, name='dense1')
        
        self.dense_layer2 = DenseLayer(
            input_shape=(None, 512),
            hidden_dim=2, activation='none', dropout=False,
            keep_prob=None, batch_normal=False, weight_decay=None, name='dense2')
        
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

    def get_loss(self, images, actions, rewards, next_images, is_terminals):
        next_action_score = self.get_inference(next_images, batch_size=self.batch_size)
        max_action_score = tf.reduce_max(next_action_score, axis=1, keep_dims=True)
        labels = tf.stop_gradient(rewards + self.gamma * max_action_score * is_terminals)
        preds = self.get_inference(images, batch_size=self.batch_size)
        actions = tf.cast(actions, dtype=tf.float32)
        preds = preds * actions
        loss = tf.nn.l2_loss(labels - preds)
        tf.add_to_collection('losses', loss / self.batch_size)
        avg_loss = tf.add_n(tf.get_collection('losses'))

        return avg_loss

    def get_max_action(self, image):
        action_score = self.get_inference(image, batch_size=1)
        max_action = tf.argmax(action_score, axis=1)

        return max_action, action_score


class QLearning:
    def __init__(self, is_show=False):
        self.env = Environment(is_show=is_show)
        self.init_image = self.env.reset()
        self.flap_prob = 0.1
        self.epsilon = 1.0
        self.image_queue_maxsize = 4
        self.replay_memory = []
        self.replay_memory_maxsize = 5000
        self.batch_size = 32
        self.n_history = self.image_queue_maxsize
        self.image_y_size = 80
        self.image_x_size = 80 
        self.n_action = 2
        self.gamma = 0.95

    def init_replay_memory(self):
        init_image = self.init_image
        image_queue = [init_image, init_image, init_image, init_image]
        is_end = False
        while not is_end:
            rnd = random.random()
            action = 'flap' if rnd < self.flap_prob else 'noflap'
            next_image, reward, is_end = self.env.render(action)
            state = self.extract_feature(copy.deepcopy(image_queue))
            image_queue.pop(0)
            image_queue.append(next_image)
            next_state = self.extract_feature(copy.deepcopy(image_queue))
            self.replay_memory.append({
                'state': state, 'action': action, 'reward': reward, 
                'is_end': is_end, 'next_state': next_state})

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
            dtype=tf.int32, shape=[
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
            batch_size=32, n_history=self.image_queue_maxsize, 
            image_y_size=self.image_y_size, image_x_size=self.image_x_size,
            n_action=self.n_action, gamma=self.gamma)
        
        # 构建优化器
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9)
        self.avg_loss = self.q_network.get_loss(
            self.images, self.actions, self.rewards, self.next_images, self.is_terminals)
        self.optimizer_handle = self.optimizer.minimize(self.avg_loss, global_step=self.global_step)
        # 构建预测器
        self.max_action, self.action_prob = self.q_network.get_max_action(self.images)
        
        # 模型保存器
        self.saver = tf.train.Saver(
            var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2, max_to_keep=50)
        # 模型初始化
        self.sess.run(tf.global_variables_initializer())

    def extract_feature(self, images):
        features = []
        for image in images:
            new_image = cv2.resize(image, (self.image_y_size, self.image_x_size))
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
            _, new_image = cv2.threshold(new_image, 1, 255, cv2.THRESH_BINARY)
            new_image = numpy.array(new_image / 255.0, dtype='float32')
            new_image = numpy.reshape(new_image, (self.image_y_size, self.image_x_size, 1))
            features.append(new_image)
        feature = numpy.concatenate(features, axis=2)
        
        return feature

    def train(self, n_episodes, backup_dir):        
        self.init_replay_memory()
        self.init_q_network()
        for item in self.replay_memory:
            print(len(item['state']), item['reward'], item['is_end'])

        for n_episode in range(n_episodes):
            # 初始化trajectory
            init_image = self.env.reset()
            image_queue = [init_image, init_image, init_image, init_image]
            is_end = False
            n_step = 0
            while not is_end:
                state = self.extract_feature(copy.deepcopy(image_queue))
                if random.random() < self.epsilon:
                    action = 'flap' if random.random() < self.flap_prob else 'noflap'
                else:
                    state_np = numpy.array([state], dtype='float32')
                    max_action = self.sess.run(
                        fetches=[self.max_action], 
                        feed_dict={self.images: state_np})
                    action = 'flap' if max_action[0] == 0 else 'noflap'
                self.epsilon = max(self.epsilon - 0.001, 0.1)
                print('action', action)
                next_image, reward, is_end = self.env.render(action)
                image_queue.pop(0)
                image_queue.append(next_image)
                next_state = self.extract_feature(copy.deepcopy(image_queue))
                self.replay_memory.append({
                    'state': state, 'action': action, 'reward': reward, 
                    'is_end': is_end, 'next_state': next_state})
                if len(self.replay_memory) >= self.replay_memory_maxsize:
                    self.replay_memory.pop(0)
                
                # 随机从replay_memory中取出1个batch
                batch_images, batch_next_images, batch_actions, batch_rewards, batch_is_terminals = \
                    [], [], [], [], []
                for j in range(self.batch_size):
                    index = random.randint(0, len(self.replay_memory)-1)
                    item = self.replay_memory[index]
                    batch_images.append(item['state'])
                    batch_next_images.append(item['next_state'])
                    batch_actions.append([1, 0] if item['action'] == 'flap' else [0, 1])
                    batch_rewards.append([item['reward']])
                    batch_is_terminals.append([0.0 if item['is_end'] else 1.0])
                batch_images = numpy.array(batch_images, dtype='float32')
                batch_next_images = numpy.array(batch_next_images, dtype='float32')
                batch_actions = numpy.array(batch_actions, dtype='int32')
                batch_rewards = numpy.array(batch_rewards, dtype='float32')
                batch_is_terminals = numpy.array(batch_is_terminals, dtype='float32')
                [_, avg_loss] = self.sess.run(
                    fetches=[self.optimizer_handle, self.avg_loss],
                    feed_dict={
                        self.images: batch_images, self.next_images: batch_next_images,
                        self.actions: batch_actions, self.rewards: batch_rewards, 
                        self.is_terminals: batch_is_terminals})
                print('[%d-%d] avg_loss: %.6f' % (n_episode, n_step, avg_loss))
                n_step += 1
            
            # trajectory结束后保存模型
            if (n_episode <= 100 and n_episode % 10 == 0) or \
                (100 < n_episode <= 1000 and n_episode % 50 == 0):
                
                model_path = os.path.join(backup_dir, 'model_%d.ckpt' % (n_episode))
                self.saver.save(self.sess, model_path)

    def test(self, model_path):
        self.init_q_network()
        self.saver.restore(self.sess, model_path)

        init_image = self.env.reset()
        image_queue = [init_image, init_image, init_image, init_image]
        is_end = False
        while not is_end:
            state = self.extract_feature(copy.deepcopy(image_queue))
            state_np = numpy.array([state], dtype='float32')
            max_action, action_prob = self.sess.run(
                fetches=[self.max_action, self.action_prob], 
                feed_dict={self.images: state_np})
            print(max_action, action_prob)
            action = 'flap' if max_action[0] == 0 else 'noflap'
            next_image, reward, is_end = self.env.render(action)
            image_queue.pop(0)
            image_queue.append(next_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parsing')
    parser.add_argument('-method')
    parser.add_argument('-gpus')
    arg = parser.parse_args()
    method = arg.method
    gpus = arg.gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    
    if method == 'train':
        main_dir = '/home/caory/github/ReinforcementLearning'
        qlearning = QLearning(is_show=False)
        qlearning.train(n_episodes=1000, 
            backup_dir=os.path.join(main_dir, 'backup', 'flappy'))
    elif method == 'test':
        main_dir = 'D://Github/ReinforcementLearning'
        qlearning = QLearning(is_show=True)
        qlearning.test(
            model_path=os.path.join(main_dir, 'backup', 'flappy', 'model_10000.ckpt'))
