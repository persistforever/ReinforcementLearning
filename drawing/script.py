# -*- coding: utf8 -*-
# author: ronniecao
# time: 2021/03/22
# description: start script of building drawing
import argparse
import os
import random
import numpy
import yaml
from drawing.env import Env
from drawing.model import Model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class Starter:

    def __init__(self, method, config_path):

        # 读取配置
        self.option = yaml.load(open(config_path, 'r'))

        random.seed(self.option['option']['seed'])
        numpy.random.seed(self.option['option']['seed'])

        # 实例化环境模块
        self.env = Env(option=self.option)
        print('Create Env instance env')

        # 实例化模型模块
        self.model = Model(
            option=self.option,
            logs_dir=os.path.join(
                self.option['option']['logs_dir'], self.option['option']['seq']),
            env=self.env)
        print('Create Model instance model')

    def main(self, method='train', gpus=''):

        if method == 'debug':
            # debug
            os.environ['CUDA_VISIBLE_DEVICES'] = self.option['option']['gpus']
            self.model.debug()

        elif method == 'train':
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
        whole_method = 'debug'
        whole_config_path = '/home/caory/github/ReinforcementLearning/scripts/drawing_v1/config.yaml'

    starter = Starter(method=whole_method, config_path=whole_config_path)
    starter.main(method=whole_method)
