# -*- coding: utf8 -*-
# author: ronniecao
# time: 2021/03/22
# description: model managering of space_invaders
import os
import re
import copy
import time
import random
import json
import collections
import multiprocessing as mp
import numpy
import tensorflow as tf
import gym
import cv2
import space_invaders.utils as utils


class Model:
    """
    模型类：控制模型训练、验证、预测和应用
    """
    def __init__(self, option, logs_dir, processor, network):

        # 读取配置
        self.option = option
        self.logs_dir = logs_dir
        self.processor = processor
        self.network = network

        # 设置参数
        self.image_y_size = self.option['option']['image_y_size']
        self.image_x_size = self.option['option']['image_x_size']
        self.n_action = self.option['option']['n_action']
        self.learning_rate = self.option['option']['learning_rate']
        self.update_function = self.option['option']['update_function']
        self.batch_size = self.option['option']['batch_size']
        self.n_gpus = self.option['option']['n_gpus']

    def _set_place_holders(self):
        """
        定义输入网络的变量place_holder
        """
        # 读取data_size
        place_holders = {}
        dtype_dict = {'int32': tf.int32, 'float32': tf.float32}
        for name in self.option['option']['data_size']:
            size = self.option['option']['data_size'][name]['size']
            dtype = self.option['option']['data_size'][name]['dtype']
            place_holders[name] = tf.placeholder(
                dtype=dtype_dict[dtype], name=name, shape=[None] + size)

        return place_holders

    def sess_init(self):
        """
        初始化训练过程使用的session
        """
        with self.network.graph.as_default():
            # 先计算loss
            self.global_step = tf.train.get_or_create_global_step()
            self.place_holders = self._set_place_holders()
            with tf.name_scope('cal_loss_and_eval'):
                self.avg_loss = \
                    self.network.get_regression_loss(self.place_holders)
                if self.update_function == 'momentum':
                    self.optimizer = tf.train.MomentumOptimizer(
                        learning_rate=self.learning_rate, momentum=0.9)
                elif self.update_function == 'adam':
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                elif self.update_function == 'adadelta':
                    self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
                update_ops = tf.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.optimizer_handle = self.optimizer.minimize(self.avg_loss,
                        global_step=self.global_step)

            # 预测
            self.q_values = self.network.get_inference(self.place_holders)

            # 赋值
            online_params = tf.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='online')
            target_params = tf.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='target')
            copy_ops = [target_var.assign(online_var) \
                for target_var, online_var in zip(target_params, online_params)]
            self.copy_online_to_target = tf.group(*copy_ops)

            # 构建update session
            gpu_options = tf.GPUOptions(allow_growth=True,
                visible_device_list=self.option['option']['train_gpus'])
            config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.copy_online_to_target)

            print('finishing initialization')

    def predict_action_values_by_model(self, example):
        """
        获取一些sample，进行模型预测
        """
        # 获取batch_phs
        batch_phs = {}
        for name in self.option['option']['data_size']:
            size = self.option['option']['data_size'][name]['size']
            dtype = self.option['option']['data_size'][name]['dtype']
            batch_phs[name] = {
                'size': [1] + size, 'dtype': dtype,
                'value': numpy.zeros([1] + size, dtype=dtype)}
        sample_ph = self.processor.get_sample_ph_from_example(example)
        for name in self.option['option']['data_size']:
            batch_phs[name]['value'][0] = sample_ph[name]['value']

        # 网络计算
        feed_dict = {}
        for name in self.option['option']['data_size']:
            feed_dict[self.place_holders[name]] = batch_phs[name]['value']
        [q_values] = self.sess.run(fetches=[self.q_values], feed_dict=feed_dict)

        return q_values[0]

    def train(self):
        """
        训练模型
        """
        # 初始化环境
        self.env = gym.make('SpaceInvaders-v0')
        self.sess_init()
        self.memory_buffer = []
        self.saver = None
        process_images = 0
        n_frame = 0
        n_epoch = 1

        # 打印内存情况
        print('currect memory: %dMB' % (
            int(float(utils.get_mem() / 1024.0 / 1024.0))))
        print('**** Start Training ****')

        start_time = time.time()
        while True:
            # 如果大于max frame则停止
            if n_frame >= self.option['option']['max_frame']:
                break

            # 重置env
            obs = self.env.reset()
            obs = self.processor.convert_image(obs)

            # 初始化trajectory中的变量
            n_step = 1
            rewards = []
            recent_obs_list = []
            for _ in range(self.option['option']['n_recent_frame']-1):
                blank = numpy.zeros((self.image_y_size, self.image_x_size, 1), dtype='float32')
                recent_obs_list.append(blank)
            recent_obs_list.append(obs)

            # 生成search tree
            is_end = False
            while not is_end:
                # 更新epsilon
                epsilon = self.option['option']['epsilon_start'] + \
                    1.0 * n_frame / self.option['option']['epsilon_decay'] * \
                        (self.option['option']['epsilon_end'] - \
                            self.option['option']['epsilon_start'])
                if epsilon < self.option['option']['epsilon_decay']:
                    epsilon = self.option['option']['epsilon_decay']

                # 计算actions
                example = {'state': recent_obs_list,
                    'action': None, 'reward': None, 'is_end': None}
                action_values = self.predict_action_values_by_model(example)

                # 以某种策略进行采样
                rnd = random.random()
                if rnd <= epsilon:
                    action = random.choice(list(range(self.n_action)))
                else:
                    # 获取prob最大的child
                    action = numpy.argmax(action_values)

                # 将action传递给env
                next_obs, reward, is_end, info = self.env.step(action)
                next_obs = self.processor.convert_image(next_obs)

                output_dir = os.path.join(self.logs_dir, 'train_images')
                image_path = os.path.join(output_dir, '%d_%d_%d.jpg' % (n_epoch, n_frame, reward))
                cv2.imwrite(image_path, next_obs)

                rewards.append(reward)
                recent_obs_list.append(next_obs)
                n_step += 1
                n_frame += 1

                # 存入example
                example['state'] = copy.deepcopy(recent_obs_list)
                example['action'] = action
                example['is_end'] = 1.0 if is_end else 0.0
                example['reward'] = -100 if is_end else reward
                del recent_obs_list[0]

                # 将sample存入memory_buffer
                self.memory_buffer.append(example)
                if len(self.memory_buffer) > self.option['option']['memory_buffer_size']:
                    del self.memory_buffer[0]

                # 训练
                if len(self.memory_buffer) >= self.batch_size:
                    examples = random.sample(self.memory_buffer, self.batch_size)

                    # 获取batch_phs
                    batch_phs = {}
                    for name in self.option['option']['data_size']:
                        size = self.option['option']['data_size'][name]['size']
                        dtype = self.option['option']['data_size'][name]['dtype']
                        batch_phs[name] = {
                            'size': [self.batch_size] + size, 'dtype': dtype,
                            'value': numpy.zeros([self.batch_size] + size, dtype=dtype)}
                    for j, example in enumerate(examples):
                        sample_ph = self.processor.get_sample_ph_from_example(example)
                        for name in self.option['option']['data_size']:
                            batch_phs[name]['value'][j] = sample_ph[name]['value']

                    # 生成feed_dict
                    feed_dict = {}
                    for name in self.option['option']['data_size']:
                        feed_dict[self.place_holders[name]] = batch_phs[name]['value']

                    # 获取网络输出
                    st = time.time()
                    [_, avg_loss] = self.sess.run(
                        fetches=[self.optimizer_handle, self.avg_loss],
                        feed_dict=feed_dict)
                    et = time.time()
                    model_time = et - st
                    process_images += self.batch_size
                    spend = (et - start_time) / 3600.0

                    print((time.ctime()))
                    print(('[epoch=%d] [frame=%d] [step=%d] '
                        'model time: %.4fs, spend: %.4fh, '
                        'image nums: %d, image speed: %.2f, '
                        'memory_buffer: %d' % (
                        n_epoch, n_frame, n_step, model_time, spend,
                        process_images, process_images / (spend * 3600.0),
                        len(self.memory_buffer))))

                    # 每1轮训练观测一次train_loss
                    print(('[epoch=%d] [frame=%d] [step=%d] '
                        'train loss: %.6f, epsilon: %.8f' % (
                        n_epoch, n_frame, n_step, avg_loss, epsilon)))
                    print()

                    # 保存模型
                    if n_frame % self.option['option']['save_frame_freq'] == 0:
                        with self.network.graph.as_default():
                            if self.saver is None:
                                self.saver = tf.train.Saver(
                                    var_list=tf.global_variables(),
                                    write_version=tf.train.SaverDef.V2, max_to_keep=100)
                            if not os.path.exists(os.path.join(self.logs_dir,
                                self.option['option']['model_name'])):
                                os.mkdir(os.path.join(self.logs_dir,
                                    self.option['option']['model_name']))
                            if not os.path.exists(
                                os.path.join(self.logs_dir,
                                    self.option['option']['model_name'], str(n_frame))):
                                os.mkdir(os.path.join(self.logs_dir,
                                    self.option['option']['model_name'], str(n_frame)))
                            model_path = os.path.join(self.logs_dir,
                                self.option['option']['model_name'],
                                str(n_frame), 'tensorflow.ckpt')
                            self.saver.save(self.sess, model_path)

            # 统计整个trajectory的reward
            sum_rewards = sum([reward for reward in rewards])
            print((time.ctime()))
            print('[epoch=%d] sum rewards: %d' % (n_epoch, sum_rewards))
            print()
            n_epoch += 1

            break

            # 更新Q function
            self.sess.run(self.copy_online_to_target)

        self.sess.close()