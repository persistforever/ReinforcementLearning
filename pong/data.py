# -*- coding: utf8 -*-
# author: ronniecao
# time: 2021/03/22
# description: data processing of space_invaders
import os
import time
import json
import math
import random
import collections
import copy
import multiprocessing as mp
import numpy
import cv2
import pong.utils as utils


class Processor:
    """
    数据准备类：对数据进行预处理、数据增强等过程
    """
    def __init__(self, option, logs_dir):
        # 读取配置
        self.option = option
        self.logs_dir = logs_dir

        self.image_y_size = self.option['option']['image_y_size']
        self.image_x_size = self.option['option']['image_x_size']
        self.n_action = self.option['option']['n_action']
        self.n_history = self.option['option']['n_history']
        self.memory_buffer = []

    def get_sample_ph_from_example(self, example):
        """
        从example中填入sample_ph
        """
        sample_ph = collections.OrderedDict()
        for name in self.option['option']['data_size']:
            size = self.option['option']['data_size'][name]['size']
            dtype = self.option['option']['data_size'][name]['dtype']
            sample_ph[name] = {
                'size': size, 'dtype': dtype,
                'value': numpy.zeros(size, dtype=dtype)}

        # online image
        state = numpy.stack(example['state'], axis=2)
        state = 1.0 * state / 255.0
        sample_ph['online_image']['value'] = state

        # target image
        if example['next_state'] is not None:
            next_state = numpy.stack(example['next_state'], axis=2)
            next_state = 1.0 * next_state / 255.0
            sample_ph['target_image']['value'] = next_state

            # action_mask
            action = example['action']
            sample_ph['action_mask']['value'][action, 0] = 1.0

            # reward
            sample_ph['reward']['value'][0] = example['reward']

            # is end
            sample_ph['is_end']['value'][0] = example['is_end']

        # coef
        sample_ph['coef']['value'][0] = 1.0

        return sample_ph

    def put_to_memory(self, example):
        """
        将一个example存入memory
        """
        self.memory_buffer.append(example)

        while len(self.memory_buffer) > self.option['option']['max_memory_size']:
            del self.memory_buffer[0]

    def get_from_memory(self):
        """
        从memory的index位置取n_history个sample
        """
        # 索引位置
        index = random.randint(self.n_history-1, len(self.memory_buffer)-2)

        # 获取action, reward, is_end
        action = self.memory_buffer[index]['action']
        reward = self.memory_buffer[index]['reward']
        is_end = self.memory_buffer[index]['is_end']

        # 获取state
        state = []
        state.append(self.memory_buffer[index]['obs'])
        for i in range(1, self.n_history):
            if self.memory_buffer[index-i]['is_end']:
                blank = numpy.zeros((self.image_y_size, self.image_x_size), dtype='uint8')
                state.extend([blank] * (self.n_history - i))
                break
            else:
                state.append(self.memory_buffer[index-i]['obs'])
        state = state[::-1]

        # 获取next state
        next_state = state[1:]
        if is_end:
            blank = numpy.zeros((self.image_y_size, self.image_x_size), dtype='uint8')
            next_state.append(blank)
        else:
            next_state.append(self.memory_buffer[index+1]['obs'])

        example = {'index': index, 'state': state, 'action': action, 'reward': reward,
            'is_end': is_end, 'next_state': next_state}

        return example