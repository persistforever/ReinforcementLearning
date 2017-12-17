# -*- coding: utf-8 -*-
# author: ronniecao
# time: 2017/12/09
# description: environment of flappy bird
from itertools import cycle
import random
import time
import os
import cv2
import numpy
import matplotlib.pyplot as plt
from matplotlib import animation
# import pygame


class Object:
    def __init__(self, pic_path, color, is_reverse=False, is_show=False):
        self.pos = [0, 0] # initial position
        image = cv2.imread(pic_path)
        self.size = [image.shape[1], image.shape[0]] # 物体的宽和高
        if is_show:
            self.surface = pygame.image.load(pic_path).convert_alpha()
            if is_reverse:
                self.surface = pygame.transform.rotate(self.surface, 180)
        image_r = numpy.zeros((self.size[1], self.size[0], 1), dtype='uint8') + color[0]
        image_g = numpy.zeros((self.size[1], self.size[0], 1), dtype='uint8') + color[1]
        image_b = numpy.zeros((self.size[1], self.size[0], 1), dtype='uint8') + color[2]
        self.image = numpy.concatenate([image_r, image_g, image_b], axis=2)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

class Background(Object):
    def __init__(self, pic_path, is_show=False):
        Object.__init__(self, pic_path, color=[35,35,35], is_show=is_show)
        self.pos = [0, 0]

class Pipe(Object):
    def __init__(self, pic_path, is_reverse=False, is_show=False):
        Object.__init__(self, pic_path, color=[67,205,128], 
            is_reverse=is_reverse, is_show=is_show)
        self.is_reverse = is_reverse # 是否是翻转的管道
        self.speed = -4 # 横向的平移量
        self.is_passed = False # 管道是否被穿过

class Ground(Object):
    def __init__(self, pic_path, is_show=False):
        Object.__init__(self, pic_path, color=[238,220,130], is_show=is_show)
        self.pos = [0, 400]
        self.speed = -100 # 横向的平移速度

class Score(Object):
    def __init__(self, pic_path, is_show=False):
        Object.__init__(self, pic_path, color=[220,220,220], is_show=is_show)
        self.pos = [120, 20]

class Bird(Object):

    def __init__(self, pic_path, is_show=False):
        Object.__init__(self, pic_path, color=[65,105,225], is_show=is_show)
        self.pos = [50, 256]
        self.max_speed = 8 # 纵向的最大下落速度
        self.accleration = 1 # 纵向的下落加速度
        self.flap_speed = -6 # 纵向的振翅上升速度
        self.speed = self.flap_speed # 纵向的速度
        self.is_flap = False # 是否振翅


class Environment:
    
    def __init__(self, is_show=True):
        self.is_show = is_show

        self.width = 288
        self.height = 512
        self.pipe_gap = 100
        self.n_frame = 0
        self.max_nps = 35
        self.pipe_queue = [] # 管道队列
        self.pipe_path = 'environment/flappy/pipe-normal.png'
        self.bird_index_gen = cycle([0, 1, 2, 1])
        self.score_paths = [
            'environment/flappy/0.png', 'environment/flappy/1.png',
            'environment/flappy/2.png', 'environment/flappy/3.png',
            'environment/flappy/4.png', 'environment/flappy/5.png',
            'environment/flappy/6.png', 'environment/flappy/7.png',
            'environment/flappy/8.png', 'environment/flappy/9.png',]
        self.score_start_x = 120 # 分数起始x
        self.score_width = 24 # 分数宽度
        
        self.reset()

    def reset(self):
        self.n_frame = 0
        self.pipe_queue = [] # 管道队列
        self.is_end = False
        self.n_score = 0
        # random.seed(0)

        # 设置画图所需变量
        if self.is_show:
            pygame.init()
            self.fps_clock = pygame.time.Clock()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Flappy Bird')

        # 初始化物体对象
        self.canvas = numpy.zeros(shape=(self.height, self.width, 3), dtype='uint8')
        self.background = Background('environment/flappy/background-day.png', 
            is_show=self.is_show)
        self.ground = Ground('environment/flappy/ground.png', is_show=self.is_show)
        self.birds = [
            Bird('environment/flappy/bluebird-downflap.png', is_show=self.is_show),
            Bird('environment/flappy/bluebird-midflap.png', is_show=self.is_show),
            Bird('environment/flappy/bluebird-upflap.png', is_show=self.is_show)]

        # 初始化pipe
        ## 初始化第一个pipe的位置
        self._push_pipe(x_pos=self.width+50)
        ## 初始化第二个pipe的位置
        self._push_pipe(x_pos=self.width+200)

        # 初始化bird
        self.bird_index = 0
        self.bird = self.birds[self.bird_index]
        self.bird.speed = self.bird.flap_speed
        self.bird_flap = False

        if self.is_show:
            self.show(n_frame=self.n_frame)
        image = self.get_image(n_frame=self.n_frame)

        return image

    def render(self, action):
        # 判断是否flap
        if self.is_show:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and \
                    (event.key == pygame.K_SPACE or event.key == pygame.K_UP):
                    self.bird_flap = True
        else:
            self.bird_flap = True if action == 'flap' else False

        # 移动pipe
        for up_pipe, down_pipe in self.pipe_queue:
            up_pipe.pos[0] += up_pipe.speed
            down_pipe.pos[0] += down_pipe.speed
        # 如果第二个pipe到达某一位置，产生新的pipe
        if self.pipe_queue[-1][0].pos[0] <= 150:
            self._push_pipe(x_pos=self.pipe_queue[-1][0].pos[0]+150)
        # 如果第一个pipe到达某一位置，移除这个pipe
        if self.pipe_queue[0][0].pos[0] <= -self.pipe_queue[0][0].size[0]:
            self._pop_pipe()

        # 获得bird的图片index
        if self.n_frame % 3 == 0:
            self.bird_index = self.bird_index_gen.next()
        self.bird = self.birds[self.bird_index]

        # 更新bird的速度和位置
        for bird in self.birds:
            bird.is_flap = self.bird_flap
            bird.pos[1] += bird.speed
            if bird.speed < bird.max_speed and not bird.is_flap:
                bird.speed += bird.accleration
            elif bird.is_flap:
                bird.speed = bird.flap_speed
            elif bird.speed >= bird.max_speed:
                bird.speed = bird.max_speed
        self.bird_flap = False

        # 碰撞检测
        is_crash = False
        self.reward = 1
        ## 判断地板碰撞
        if self._is_object_union(self.bird, self.ground):
            is_crash = True
            self.reward = -500
        ## 判断和水管碰撞
        for up_pipe, down_pipe in self.pipe_queue:
            if self._is_object_union(self.bird, up_pipe):
                is_crash = True
                self.reward = -50
            if self._is_object_union(self.bird, down_pipe):
                is_crash = True
                self.reward = -50
        if is_crash:
            self.is_end = True

        # bird通过pipe检测
        for up_pipe, down_pipe in self.pipe_queue:
            if not down_pipe.is_passed:
                if self.bird.pos[0] >= down_pipe.pos[0] + down_pipe.size[0]:
                    down_pipe.is_passed = True
                    self.n_score += 1
                    self.reward = 50

        if self.is_show:
            self.show(n_frame=self.n_frame)
        image = self.get_image(n_frame=self.n_frame)

        # 变换到下一帧
        self.n_frame += 1

        return image, self.reward, self.is_end

    def show(self, n_frame=0):
        # 画出物体
        self.window.blit(self.background.surface, self.background.pos)
        for up_pipe, down_pipe in self.pipe_queue:
            self.window.blit(up_pipe.surface, up_pipe.pos)
            self.window.blit(down_pipe.surface, down_pipe.pos)
        self.window.blit(self.ground.surface, self.ground.pos)
        self.window.blit(self.bird.surface, self.bird.pos)
        # 画出分数
        score_list = [int(t) for t in str(self.n_score)]
        for i, score in enumerate(score_list):
            s = Score(self.score_paths[score], is_show=self.is_show)
            s.pos[0] = self.score_start_x + i * self.score_width
            self.window.blit(s.surface, s.pos)

        pygame.display.flip()
        self.fps_clock.tick(self.max_nps)

    def get_image(self, n_frame=0):
        # 将各个object放置在canvas中
        self._set_object(self.canvas, self.background)
        self._set_object(self.canvas, self.bird)
        for up_pipe, down_pipe in self.pipe_queue:
            self._set_object(self.canvas, up_pipe)
            self._set_object(self.canvas, down_pipe)
        self._set_object(self.canvas, self.ground)
        
        return self.canvas

    def _set_object(self, canvas, object):
        target_left = max(0, object.pos[0])
        target_right = min(object.pos[0] + object.size[0], canvas.shape[1])
        target_top = max(0, object.pos[1])
        target_bottom = min(object.pos[1] + object.size[1], canvas.shape[0])
        source_left = target_left - object.pos[0]
        source_right = target_right - object.pos[0]
        source_top = target_top - object.pos[1]
        source_bottom = target_bottom - object.pos[1]
        if target_right >= 0 and target_left <= canvas.shape[1]:
            canvas[target_top:target_bottom, target_left:target_right, :] = \
                object.image[source_top:source_bottom, source_left:source_right, :]
        
    def _push_pipe(self, x_pos):
        pipe_y = random.randint(80, 260)
        up_pipe = Pipe(self.pipe_path, is_reverse=True, is_show=self.is_show)
        up_pipe.pos = [x_pos, pipe_y - up_pipe.size[1]]
        down_pipe = Pipe(self.pipe_path, is_reverse=False, is_show=self.is_show)
        down_pipe.pos = [x_pos, pipe_y + self.pipe_gap]
        self.pipe_queue.append([up_pipe, down_pipe])

    def _pop_pipe(self):
        del self.pipe_queue[0]

    def _is_object_union(self, objecta, objectb):
        lefta = objecta.pos[0]
        righta = objecta.pos[0] + objecta.size[0]
        topa = objecta.pos[1]
        bottoma = objecta.pos[1] + objecta.size[1]

        leftb = objectb.pos[0]
        rightb = objectb.pos[0] + objectb.size[0]
        topb = objectb.pos[1]
        bottomb = objectb.pos[1] + objectb.size[1] 

        union_left = max(lefta, leftb)
        union_right = min(righta, rightb)
        union_top = max(topa, topb)
        union_bottom = min(bottoma, bottomb)

        if union_right > union_left and union_bottom > union_top:
            is_union = True
        else:
            is_union = False

        return is_union


def main():
    env = Environment(is_show=False)
    for i in range(500):
        if not env.is_end:
            env.render()


if __name__ == '__main__':
    main()
