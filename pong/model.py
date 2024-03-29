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
import pong.utils as utils
from pong.atari import AtariPlayer
from pong.atari_wrapper import FireResetEnv, FrameStack, LimitLength, MapState


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
        self.gamma = self.option['option']['gamma']

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
                self.avg_loss = self.network.get_regression_loss(self.place_holders)
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
            self.action_probs = self.network.get_inference(self.place_holders)

            # 构建update session
            gpu_options = tf.GPUOptions(allow_growth=True,
                visible_device_list=self.option['option']['train_gpus'])
            config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

            print('finishing initialization')

    def read_model(self, model_path):
        # 读取模型
        with self.network.graph.as_default():
            deploy_saver = tf.train.Saver(var_list=tf.global_variables(),
                write_version=tf.train.SaverDef.V2, max_to_keep=500)
            deploy_saver.restore(self.sess, model_path)

    def predict_action_probs_by_model(self, example):
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
        [action_probs] = self.sess.run(fetches=[self.action_probs], feed_dict=feed_dict)

        return action_probs[0]

    def resize_keepdims(self, im, size):
        # Opencv's resize remove the extra dimension for grayscale images. We add it back.
        ret = cv2.cv2.resize(im, size)
        if im.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, numpy.newaxis]
        return ret

    def debug(self):
        """
        debug
        """
        # 初始化环境
        self.env = AtariPlayer(self.option['option']['env_path'], frame_skip=4)
        self.env = FireResetEnv(self.env)
        self.env = MapState(self.env,
            lambda im: self.resize_keepdims(im, (self.image_y_size, self.image_x_size)))
        obs = self.env.reset()
        n = 0
        while True:
            # 存入memory
            n += 1
            action = random.choice(list(range(self.n_action)))
            next_obs, reward, is_end, info = self.env.step(action)
            if reward != 0:
                is_end = True
            path = os.path.join('/home/caory/programs/image/pong/%d.jpg' % (n))
            cv2.imwrite(path, next_obs)
            print(action, reward, is_end)
            if is_end:
                obs = self.env.reset()
                break

    def train(self):
        """
        训练模型
        """
        # 初始化环境
        self.env = AtariPlayer(self.option['option']['env_path'], frame_skip=4)
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

        # 开始训练
        n_frame, max_mean_reward = 0, -100
        reward_list = []
        max_epoch = self.option['option']['max_frame'] // \
            self.option['option']['log_frame_freq']
        for n_epoch in range(1, max_epoch+1):
            # 重置env
            obs = self.env.reset()
            recent_obs_list = []
            for _ in range(self.option['option']['n_history']-1):
                blank = numpy.zeros((self.image_y_size, self.image_x_size), dtype='uint8')
                recent_obs_list.append(blank)
            recent_obs_list.append(obs)
            episode = []

            # 开始epoch内循环
            while True:
                n_frame += 1

                # 采样action
                example = {'state': recent_obs_list, 'action': None, 'reward': None}
                action_values = self.predict_action_probs_by_model(example)
                action = numpy.random.choice(numpy.arange(0, self.n_action), p=action_values)

                # 将action传递给env
                next_obs, reward, _, _ = self.env.step(action)
                example = {'state': copy.deepcopy(recent_obs_list), 'action': action, 'reward': reward}
                episode.append(example)

                # 更新recent_obs_list
                recent_obs_list.append(next_obs)
                del recent_obs_list[0]

                # 如果episode结束，则初始化一些变量
                if reward != 0:
                    reward_list.append(reward)
                    break

            # 获取batch_phs
            batch_phs = {}
            for name in self.option['option']['data_size']:
                batch_phs[name] = []
            for example in episode:
                sample_ph = self.processor.get_sample_ph_from_example(example)
                for name in self.option['option']['data_size']:
                    batch_phs[name].append(sample_ph[name]['value'])

            # 生成feed_dict
            feed_dict = {}
            for name in self.option['option']['data_size']:
                feed_dict[self.place_holders[name]] = batch_phs[name]

            # 获取网络输出
            [_, avg_loss] = self.sess.run(
                fetches=[self.optimizer_handle, self.avg_loss],
                feed_dict=feed_dict)

            # 统计整个trajectory的reward
            logging.info('')
            mean_reward = 1.0 * sum(reward_list[-100:]) / len(reward_list[-100:])
            logging.info('[epoch=%d] mean reward: %.2f, avg loss: %.4f' % (
                n_epoch, mean_reward, avg_loss))
            logging.info('')

            # 保存模型
            if mean_reward >= max_mean_reward:
                max_mean_reward = mean_reward
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
                            self.option['option']['model_name'], 'best')):
                        os.mkdir(os.path.join(self.logs_dir,
                            self.option['option']['model_name'], 'best'))
                    model_path = os.path.join(self.logs_dir,
                        self.option['option']['model_name'],
                        'best', 'tensorflow.ckpt')
                    self.saver.save(self.sess, model_path)

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

    def play(self):
        """
        使用模型
        """
        # 初始化环境
        self.env = AtariPlayer(self.option['option']['env_path'], viz=0.5)
        self.env = FireResetEnv(self.env)
        self.env = MapState(self.env,
            lambda im: self.resize_keepdims(im, (self.image_y_size, self.image_x_size)))
        self.sess_init()
        self.read_model(model_path=self.option['option']['model_path'])

        # 重置env
        obs = self.env.reset()
        self.env.render()
        recent_obs_list = []
        for _ in range(self.option['option']['n_history']-1):
            blank = numpy.zeros((self.image_y_size, self.image_x_size), dtype='uint8')
            recent_obs_list.append(blank)
        recent_obs_list.append(obs)

        # 循环play
        while True:
            # 获取prob最大的child
            example = {'state': recent_obs_list, 'next_state': None,
                'action': None, 'reward': None, 'is_end': None}
            action_values = self.predict_action_values_by_model(example)
            action = numpy.argmax(action_values)

            # 将action传递给env
            next_obs, reward, is_end, info = self.env.step(action)
            self.env.render()

            # 更新recent_obs_list
            obs = next_obs
            recent_obs_list.append(obs)
            del recent_obs_list[0]

            if is_end:
                if info['ale.lives'] == 0:
                    break
                obs = self.env.reset()
                self.env.render()
                recent_obs_list = []
                for _ in range(self.option['option']['n_history']-1):
                    blank = numpy.zeros((self.image_y_size, self.image_x_size), dtype='uint8')
                    recent_obs_list.append(blank)
                recent_obs_list.append(obs)

        print('finish')
