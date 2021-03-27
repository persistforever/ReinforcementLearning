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
from tqdm import tqdm
import logging
import numpy
import tensorflow as tf
import gym
import cv2
import space_invaders.utils as utils
from space_invaders.atari import AtariPlayer
from space_invaders.atari_wrapper import FireResetEnv, FrameStack, LimitLength, MapState


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
                self.avg_loss, self.mean_q_value = \
                    self.network.get_regression_loss(self.place_holders)
                if self.update_function == 'momentum':
                    self.optimizer = tf.train.MomentumOptimizer(
                        learning_rate=self.learning_rate, momentum=0.9)
                elif self.update_function == 'adam':
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                elif self.update_function == 'adadelta':
                    self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
                elif self.update_function == 'rmsprop':
                    self.optimizer = tf.train.RMSPropOptimizer(
                        self.learning_rate, decay=0.95, momentum=0.95, epsilon=1e-2)
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

    def resize_keepdims(self, im, size):
        # Opencv's resize remove the extra dimension for grayscale images. We add it back.
        ret = cv2.resize(im, size)
        if im.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, numpy.newaxis]
        return ret

    def train(self):
        """
        训练模型
        """
        # 初始化环境
        self.env = AtariPlayer(self.option['option']['env_path'], max_num_frames=60000)
        self.env = FireResetEnv(self.env)
        self.env = MapState(self.env,
            lambda im: self.resize_keepdims(im, (self.image_y_size, self.image_x_size)))
        self.sess_init()
        self.saver = None
        process_images = 0

        # 打印内存情况
        print('currect memory: %dMB' % (
            int(float(utils.get_mem() / 1024.0 / 1024.0))))
        print('**** Start Training ****')
        logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filename=self.option['option']['log_path'],
            filemode='w')

        # 定义logger handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)

        # 先随机填充memory_buffer
        start_time = time.time()
        obs = self.env.reset()
        episode = []
        for i in tqdm(range(self.option['option']['init_memory_size'])):
            # 存入memory
            action = random.choice(list(range(self.n_action)))
            next_obs, reward, is_end, info = self.env.step(action)
            if is_end:
                reward = -1
            sample = {'index': i, 'obs': obs, 'action': action,
                'reward': reward, 'is_end': is_end}
            episode.append(sample)
            # 本帧结束
            obs = next_obs
            # 如果结束，重启环境
            if is_end:
                obs = self.env.reset()
                if 'ale.lives' not in info or info['ale.lives'] == 0:
                    for sample in episode:
                        self.processor.put_to_memory(sample)
                    episode = []
        logging.info('finish filling init memory')

        # 重置env
        obs = self.env.reset()
        recent_obs_list = []
        for _ in range(self.option['option']['n_history']-1):
            blank = numpy.zeros((self.image_y_size, self.image_x_size), dtype='uint8')
            recent_obs_list.append(blank)
        recent_obs_list.append(obs)
        episode = []
        rewards = []

        max_epoch = self.option['option']['max_frame'] // \
            self.option['option']['log_frame_freq']
        for n_epoch in range(1, max_epoch+1):
            # 初始化一个epoch内的变量
            rewards_list = []
            pred_values = []
            losses = []

            # 开始epoch内循环
            for n_step in tqdm(range(self.option['option']['log_frame_freq'])):
                n_frame = (n_epoch - 1) * self.option['option']['log_frame_freq'] + n_step

                # 更新epsilon
                epsilon = self.option['option']['epsilon_start'] + \
                    1.0 * n_frame / self.option['option']['epsilon_decay'] * \
                        (self.option['option']['epsilon_end'] - \
                            self.option['option']['epsilon_start'])
                if epsilon < self.option['option']['epsilon_end']:
                    epsilon = self.option['option']['epsilon_end']

                # 以某种策略进行采样
                rnd = random.random()
                if rnd <= epsilon:
                    action = random.choice(list(range(self.n_action)))
                else:
                    # 获取prob最大的child
                    example = {'state': recent_obs_list, 'next_state': None,
                        'action': None, 'reward': None, 'is_end': None}
                    action_values = self.predict_action_values_by_model(example)
                    action = numpy.argmax(action_values)

                # 将action传递给env
                next_obs, reward, is_end, info = self.env.step(action)
                rewards.append(reward)
                if is_end:
                    reward = -1
                # 获取一个sample并存入memory
                sample = {'index': n_frame, 'obs': obs, 'action': action,
                    'reward': reward, 'is_end': is_end}
                episode.append(sample)
                # 更新recent_obs_list
                obs = next_obs
                recent_obs_list.append(obs)
                del recent_obs_list[0]

                # 如果episode结束，则初始化一些变量
                if is_end:
                    obs = self.env.reset()
                    if 'ale.lives' not in info or info['ale.lives'] == 0:
                        recent_obs_list = []
                        for _ in range(self.option['option']['n_history']-1):
                            blank = numpy.zeros(
                                (self.image_y_size, self.image_x_size), dtype='uint8')
                            recent_obs_list.append(blank)
                        recent_obs_list.append(obs)
                        for sample in episode:
                            self.processor.put_to_memory(sample)
                        episode = []
                        rewards_list.append(rewards)
                        rewards = []

                # 训练
                examples = []
                for _ in range(self.batch_size):
                    examples.append(self.processor.get_from_memory())

                # 打印
                if False:
                    for j, example in enumerate(examples):
                        index = example['index']
                        action = example['action']
                        reward = example['reward']
                        is_end = example['is_end']
                        state = example['state']
                        next_state = example['next_state']
                        for k in range(self.option['option']['n_history']):
                            pic = state[k]
                            output_path = os.path.join(self.logs_dir, 'train_images',
                                '%d_state_%d_%d_%d_%d.jpg' % (
                                    j, k, action, int(reward), int(is_end)))
                            cv2.imwrite(output_path, pic)
                        for k in range(self.option['option']['n_history']):
                            pic = next_state[k]
                            output_path = os.path.join(self.logs_dir, 'train_images',
                                '%d_nextstate_%d_%d_%d_%d.jpg' % (
                                    j, k, action, int(reward), int(is_end)))
                            cv2.imwrite(output_path, pic)
                        for k in range(-self.option['option']['n_history']+1, 2):
                            pic = self.processor.memory_buffer[index+k]['obs']
                            output_path = os.path.join(self.logs_dir, 'train_images',
                                '%d_memory_%d_%d_%d_%d.jpg' % (
                                    j, k, action, int(reward), int(is_end)))
                            cv2.imwrite(output_path, pic)

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
                [_, avg_loss, mean_q_value] = self.sess.run(
                    fetches=[self.optimizer_handle, self.avg_loss, self.mean_q_value],
                    feed_dict=feed_dict)
                avg_loss = avg_loss / self.batch_size
                losses.append(avg_loss)
                pred_values.append(mean_q_value)

                # 更新Q function
                if n_frame % self.option['option']['update_frame_freq'] == 0:
                    self.sess.run(self.copy_online_to_target)

            # 统计整个trajectory的reward
            logging.info('')
            logging.info('[epoch=%d] [frame=%d] memory_buffer: %d, epsilon: %.8f' % (
                n_epoch, n_frame+1, len(self.processor.memory_buffer), epsilon))

            max_reward = max([sum(rewards) for rewards in rewards_list])
            mean_reward = 1.0 * sum(
                [sum(rewards) for rewards in rewards_list]) / len(rewards_list)
            mean_value = 1.0 * sum(pred_values) / (len(pred_values) + 1e-8)
            avg_loss = 1.0 * sum(losses) / len(losses)
            logging.info('[epoch=%d] max reward: %d, mean reward: %.2f, '
                'avg loss: %.4f, mean value: %.4f' % (
                n_epoch, max_reward, mean_reward, mean_value, avg_loss))
            logging.info('')

            # 保存模型
            if n_epoch % self.option['option']['save_epoch_freq'] == 0:
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
                            self.option['option']['model_name'], str(n_epoch))):
                        os.mkdir(os.path.join(self.logs_dir,
                            self.option['option']['model_name'], str(n_epoch)))
                    model_path = os.path.join(self.logs_dir,
                        self.option['option']['model_name'],
                        str(n_epoch), 'tensorflow.ckpt')
                    self.saver.save(self.sess, model_path)

        self.sess.close()