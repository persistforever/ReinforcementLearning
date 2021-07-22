# -*- coding: utf8 -*-
# author: ronniecao
# time: 2021/07/22
# description: model managering of scheduling
import os
import cv2


class Model:
    """
    模型类：控制模型训练、验证、预测和应用
    """
    def __init__(self, option, logs_dir, env):

        # 读取配置
        self.option = option
        self.logs_dir = logs_dir
        self.env = env

    def debug(self):
        """
        debug
        """
        # 初始化环境
        for fname in os.listdir(self.option['option']['train_dir']):
            wlid = fname.split('.')[0]
            path = os.path.join(self.option['option']['train_dir'], fname)
            self.env.reset(path)

            # 渲染
            if not os.path.exists(os.path.join(self.logs_dir, 'train_images')):
                os.mkdir(os.path.join(self.logs_dir, 'train_images'))
            if not os.path.exists(os.path.join(self.logs_dir, 'train_images', wlid)):
                os.mkdir(os.path.join(self.logs_dir, 'train_images', wlid))
            output_path = os.path.join(self.logs_dir, 'train_images', wlid, '0.png')
            image = self.env.render(self.env.info)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, image)