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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Network:
    def __init__(self, batch_size, n_history, image_y_size, image_x_size, n_channel, n_action, gamma):
        self.batch_size = batch_size
        self.n_history = n_history
        self.image_y_size = image_y_size
        self.image_x_size = image_x_size
        self.cell_y_size = int(image_y_size / 32)
        self.cell_x_size = int(image_x_size / 32)
        self.final_dim = 128
        self.n_channel = n_channel
        self.n_action = n_action
        self.gamma = gamma

        # 网络结构
        print('\n%-10s\t%-20s\t%-20s\t%s' % ('Name', 'Filter', 'Input', 'Output')) 
        self.pool_layer1 = PoolLayer(
            input_shape=(None, self.image_y_size, self.image_x_size, self.n_channel),
            n_size=4, stride=4, mode='max', resp_normal=False, name='pool1')
        self.conv_layer1 = ConvLayer(
            input_shape=(None, int(self.image_y_size/4), int(self.image_x_size/4), self.n_channel), 
            n_size=4, n_filter=32, stride=1, activation='relu', 
            batch_normal=True, weight_decay=5e-4, name='conv1')
        
        self.pool_layer2 = PoolLayer(
            input_shape=(None, int(self.image_y_size/4), int(self.image_x_size/4), 32),
            n_size=4, stride=4, mode='max', resp_normal=False, name='pool2')
        self.conv_layer2 = ConvLayer(
            input_shape=(None, int(self.image_y_size/16), int(self.image_x_size/16), 32), 
            n_size=4, n_filter=64, stride=1, activation='relu', 
            batch_normal=True, weight_decay=5e-4, name='conv2')
        
        self.pool_layer3 = PoolLayer(
            input_shape=(None, int(self.image_y_size/16), int(self.image_x_size/16), 64),
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool3')
        self.conv_layer3 = ConvLayer(
            input_shape=(None, int(self.image_y_size/32), int(self.image_x_size/32), 64), 
            n_size=4, n_filter=self.final_dim, stride=1, activation='relu', 
            batch_normal=True, weight_decay=5e-4, name='conv3')

        self.dense_layer = DenseLayer(
            input_shape=(None, 
                self.n_history * self.cell_y_size * self.cell_x_size * self.final_dim),
            hidden_dim=2, activation='none', dropout=False,
            keep_prob=None, batch_normal=False, weight_decay=None, name='dense')
        
        print('')
        sys.stdout.flush()
    
    def inference_body(self, index, n_history, batch_size, images, outputs):
        # 数据流
        hidden_pool1 = self.pool_layer1.get_output(input=images[:,index,:,:,:])
        hidden_conv1 = self.conv_layer1.get_output(input=hidden_pool1)
        hidden_pool2 = self.pool_layer2.get_output(input=hidden_conv1)
        hidden_conv2 = self.conv_layer2.get_output(input=hidden_pool2)
        hidden_pool3 = self.pool_layer3.get_output(input=hidden_conv2)
        hidden_conv3 = self.conv_layer3.get_output(input=hidden_pool3)

        hidden_conv3 = tf.reshape(hidden_conv3, shape=(
            batch_size, 1, self.cell_y_size, self.cell_x_size, self.final_dim))
        output = tf.pad(hidden_conv3, paddings=[
            [0,0], [index,n_history-index-1], [0,0], [0,0], [0,0]], mode='CONSTANT')
        outputs += output
        index += 1

        return index, n_history, batch_size, images, outputs

    def get_inference(self, images, batch_size=1):
        outputs = tf.zeros(shape=(batch_size, self.n_history,
            self.cell_y_size, self.cell_x_size, self.final_dim), dtype=tf.float32)
        results = tf.while_loop(
            cond=lambda i,n,b,im,out: i<n, body=self.inference_body,
            loop_vars=[tf.constant(0), self.n_history, batch_size, images, outputs],
            parallel_iterations=self.batch_size, 
            shape_invariants=[
                tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(()),
                tf.TensorShape((None, self.n_history,
                    self.image_y_size, self.image_x_size, self.n_channel)),
                tf.TensorShape((None, self.n_history,
                    self.cell_y_size, self.cell_x_size, self.final_dim))])
        hidden_output = results[4]
        hidden_output = tf.reshape(hidden_output, (
            batch_size, self.n_history * self.cell_y_size * self.cell_x_size * self.final_dim))
        hidden_dense = self.dense_layer.get_output(input=hidden_output)

        return hidden_dense

    def get_loss(self, images, actions, rewards, next_images, is_terminals):
        next_action_prob = self.get_inference(next_images, batch_size=self.batch_size)
        max_action_prob = tf.reduce_max(next_action_prob, axis=1, keep_dims=True)
        labels = tf.stop_gradient(rewards + self.gamma * max_action_prob * is_terminals)
        preds = self.get_inference(images, batch_size=self.batch_size)
        actions = tf.cast(actions, dtype=tf.float32)
        preds = preds * actions
        loss = tf.nn.l2_loss(labels - preds)
        tf.add_to_collection('losses', loss / self.batch_size)
        avg_loss = tf.add_n(tf.get_collection('losses'))

        return avg_loss

    def get_max_action(self, image):
        action_prob = self.get_inference(image, batch_size=1)
        max_action = tf.argmax(action_prob, axis=1)

        return max_action, action_prob


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
        self.image_y_size = 512
        self.image_x_size = 288 
        self.n_channel = 3
        self.n_action = 2

    def init_replay_memory(self):
        init_image = self.init_image
        image_queue = [init_image, init_image, init_image, init_image]
        is_end = False
        while not is_end:
            rnd = random.random()
            action = 'flap' if rnd < self.flap_prob else 'noflap'
            next_image, reward, is_end = self.env.render(action)
            state = copy.deepcopy(image_queue)
            image_queue.pop(0)
            image_queue.append(next_image)
            next_state = copy.deepcopy(image_queue)
            self.replay_memory.append({
                'state': state, 'action': action, 'reward': reward, 
                'is_end': is_end, 'next_state': next_state})

    def init_q_network(self):
        # 创建placeholder
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[
                None, self.n_history, self.image_y_size, self.image_x_size, self.n_channel],
            name='images')
        self.next_images = tf.placeholder(
            dtype=tf.float32, shape=[
                None, self.n_history, self.image_y_size, self.image_x_size, self.n_channel],
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
            batch_size=32, n_history=self.image_queue_maxsize, image_y_size=512, image_x_size=288,
            n_channel=3, n_action=2, gamma=0.9)
        
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
                state = copy.deepcopy(image_queue)
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
                next_state = copy.deepcopy(image_queue)
                self.replay_memory.append({
                    'state': state, 'action': action, 'reward': reward, 
                    'is_end': is_end, 'next_state': next_state})
                
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
            if (n_episode <= 1000 and n_episode % 200 == 0) or (1000 < n_episode <= 10000 and n_episode % 2000 == 0) \
                or (n_episode > 10000 and n_episode % 20000 == 0):
                model_path = os.path.join(backup_dir, 'model_%d.ckpt' % (n_episode))
                self.saver.save(self.sess, model_path)

    def test(self, model_path):
        self.init_q_network()
        self.saver.restore(self.sess, model_path)

        init_image = self.env.reset()
        image_queue = [init_image, init_image, init_image, init_image]
        is_end = False
        while not is_end:
            state = copy.deepcopy(image_queue)
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
    arg = parser.parse_args()
    method = arg.method
    if method == 'train':
        main_dir = '/home/caory/github/ReinforcementLearning'
        qlearning = QLearning(is_show=False)
        qlearning.train(n_episodes=10000, 
            backup_dir=os.path.join(main_dir, 'backup', 'flappy'))
    elif method == 'test':
        main_dir = 'D://Github/ReinforcementLearning'
        qlearning = QLearning(is_show=True)
        qlearning.test(
            model_path=os.path.join(main_dir, 'backup', 'flappy', 'model_10000.ckpt'))
