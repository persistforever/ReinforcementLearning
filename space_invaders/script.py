# -*- coding: utf8 -*-
# author: ronniecao
# time: 2021/03/22
# description: start script of space_invaders
import argparse
import os
import random
import copy
import numpy
import yaml
from space_invaders.data import Processor
from space_invaders.network import Network
from space_invaders.model import Model
import space_invaders.utils as utils
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class Starter:

    def __init__(self, method, config_path):

        # 读取配置
        self.option = yaml.load(open(config_path, 'r'))

        random.seed(self.option['option']['seed'])
        numpy.random.seed(self.option['option']['seed'])

        # 实例化数据准备模块
        self.processor = Processor(
            option=self.option,
            logs_dir=os.path.join(
                self.option['option']['logs_dir'], self.option['option']['seq']))
        print('Create Processor instance process')

        # 实例化网络模块
        self.network = Network(
            option=self.option,
            name='network')
        print('Create Network instance network')

        # 实例化模型模块
        self.model = Model(
            option=self.option,
            logs_dir=os.path.join(
                self.option['option']['logs_dir'], self.option['option']['seq']),
            processor=self.processor,
            network=self.network)
        print('Create Model instance model')

    def main(self, method='train', gpus=''):

        if method == 'train':
            # 训练模型
            os.environ['CUDA_VISIBLE_DEVICES'] = self.option['option']['gpus']
            self.model.train()

        elif method == 'play':
            # 训练模型
            os.environ['CUDA_VISIBLE_DEVICES'] = self.option['option']['gpus']
            self.model.play()


if __name__ == '__main__':
    print(('current process id: %d' % (os.getpid())))
    parser = argparse.ArgumentParser(description='parsing command parameters')
    parser.add_argument('-method')
    parser.add_argument('-name')
    parser.add_argument('-config')
    arg = parser.parse_args()
    whole_method = arg.method
    whole_config_path = arg.config

    # for debug
    if True:
        whole_method = 'play'
        whole_config_path = 'E:/Github/ReinforcementLearning/scripts/space_invaders_v1/config_dqn.yaml'

    starter = Starter(method=whole_method, config_path=whole_config_path)
    starter.main(method=whole_method)