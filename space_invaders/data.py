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
import space_invaders.utils as utils


class Processor:
    """
    数据准备类：对数据进行预处理、数据增强等过程
    """
    def __init__(self, option, logs_dir):
        # 读取配置
        self.option = option
        self.logs_dir = logs_dir

        self.n_action = self.option['option']['n_action']
        self.n_recent_frame = self.option['option']['n_recent_frame']

    def convert_image(self, image):
        """
        灰度化、缩小、裁剪
        """
        image = cv2.resize(image,
            (self.option['option']['resized_x_size'],
            self.option['option']['resized_y_size']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = numpy.reshape(image,
            (self.option['option']['resized_y_size'],
                self.option['option']['resized_x_size'], 1))
        image = image[-self.option['option']['image_y_size']:, :, :]

        return image

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
        state = numpy.concatenate(example['state'][0:self.n_recent_frame], axis=2)
        state = 1.0 * state / 255.0
        sample_ph['online_image']['value'] = state

        # target image
        if len(example['state']) >= self.n_recent_frame + 1:
            next_state = numpy.concatenate(example['state'][1:self.n_recent_frame+1], axis=2)
            next_state = 1.0 * next_state / 255.0
            sample_ph['target_image']['value'] = next_state

            # action_mask
            action = example['action']
            sample_ph['action_mask']['value'][action, 0] = 1.0

            # reward
            sample_ph['reward']['value'][action, 0] = example['reward']

            # is end
            sample_ph['is_end']['value'][action, 0] = example['is_end']

        # coef
        sample_ph['coef']['value'][0] = 1.0

        return sample_ph