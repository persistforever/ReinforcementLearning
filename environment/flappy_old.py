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
import pygame


class Environment:
    
    def __init__(self, observe=True):
        # basic information
        dir = os.path.split(os.path.realpath(__file__))[0]
        # configuration
        self.height = 512
        self.width = 288
        self.fps = 200
        self.pipe_gap = 100
        self.player_width = 34
        self.player_height = 24
        self.pipe_width = 52
        self.pipe_height = 320
        self.ground_width = 336
        self.ground_height = 112
        # variables
        self.actions = ['fly', 'stay']
        self.images = {}
        self.hit_mask = {}
        
        # bird information
        self.background_list = (os.path.join(dir, 'flappy/background-day.png'), \
                                os.path.join(dir, 'flappy/background-night.png'))
        self.player_list = ((os.path.join(dir, 'flappy/redbird-upflap.png'), \
                             os.path.join(dir, 'flappy/redbird-midflap.png'), \
                             os.path.join(dir, 'flappy/redbird-downflap.png')), \
                            (os.path.join(dir, 'flappy/bluebird-upflap.png'), \
                             os.path.join(dir, 'flappy/bluebird-midflap.png'), \
                             os.path.join(dir, 'flappy/bluebird-downflap.png')), \
                            (os.path.join(dir, 'flappy/yellowbird-upflap.png'), \
                             os.path.join(dir, 'flappy/yellowbird-midflap.png'), \
                             os.path.join(dir, 'flappy/yellowbird-downflap.png')))
        self.pipe_list = (os.path.join(dir, 'flappy/pipe-green.png'), \
                          os.path.join(dir, 'flappy/pipe-red.png'))
        self.number_list = (os.path.join(dir, 'flappy/0.png'), \
                            os.path.join(dir, 'flappy/1.png'), \
                            os.path.join(dir, 'flappy/2.png'), \
                            os.path.join(dir, 'flappy/3.png'), \
                            os.path.join(dir, 'flappy/4.png'), \
                            os.path.join(dir, 'flappy/5.png'), \
                            os.path.join(dir, 'flappy/6.png'), \
                            os.path.join(dir, 'flappy/7.png'), \
                            os.path.join(dir, 'flappy/8.png'), \
                            os.path.join(dir, 'flappy/9.png'))
        self.ground_path = os.path.join(dir, 'flappy/base.png')
        
        if observe:
            pygame.init()
            self.fps_clock = pygame.time.Clock()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Flappy Bird')
        
    def reset(self, seed):
        random.seed(seed)
        # init self.hit_mask
        ## hitmask for pipe
        """
        self.hit_mask['pipe'] = (self._get_hitmask(self.pipe_width, self.pipe_height), \
                                 self._get_hitmask(self.pipe_width, self.pipe_height))
        ## hit mask for player
        self.hit_mask['player'] = (self._get_hitmask(self.player_width, self.player_height), \
                                   self._get_hitmask(self.player_width, self.player_height), \
                                   self._get_hitmask(self.player_width, self.player_height))
        """
        
        # init player information
        self.loopIter = 0
        self.playerIndex = 0
        self.player_index_gen = cycle([0, 1, 2, 1])
        self.player_point = [50, 256]
        ## bird motion information
        self.player_shm_vals = {'val': 0, 'dir': 1}
        ## player velocity, max velocity, downward accleration, accleration on flap
        self.playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerAccY    =   1   # players downward accleration
        self.playerFlapAcc =  -9   # players speed on flapping
        self.playerFlapped = False # True when player flaps
        
        # init ground information
        self.ground_point = [0, 400]
        ## amount by which base can maximum shift to left
        self.ground_shift = self.ground_width - self.width
        ## pipe movement
        self.pipeVelX = -4
        
        # init score
        self.score = 0
    
        # get 2 new pipes to add to upperPipes lowerPipes list
        new_pipe1 = self._get_random_pipe()
        new_pipe2 = self._get_random_pipe()
        ## list of upper pipes
        self.upperPipes = [
            {'x': self.width + 200, 'y': new_pipe1[0]['y']},
            {'x': self.width + 200 + (self.width / 2), 'y': new_pipe2[0]['y']},
        ]
        ## list of lowerpipe
        self.lowerPipes = [
            {'x': self.width + 200, 'y': new_pipe1[1]['y']},
            {'x': self.width + 200 + (self.width / 2), 'y': new_pipe2[1]['y']},
        ]
        
        return self._get_state()
    
    def step(self, action):
        # start iteration
        state, reward, done = None, 1, False
        """
        for event in pygame.event.get():
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                action = 'fly'
        """
        if action == 'fly':
            if self.player_point[1] > -2 * self.player_height:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                
        # check for crash here
        crashTest = self._check_crash({'x': self.player_point[0], 'y': self.player_point[1], \
                                       'index': self.playerIndex}, \
                                      self.upperPipes, self.lowerPipes)
        if crashTest[0]:
            reward = -1000
            done = True

        # check for score
        playerMidPos = self.player_point[0] + self.player_width
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + self.pipe_width
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                reward = 50

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = self.player_index_gen.next()
        self.loopIter = (self.loopIter + 1) % 30
        
        # update ground point 
        self.ground_point[0] = -((-self.ground_point[0] + 100) % self.ground_shift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        playerHeight = self.player_height
        self.player_point[1] += min(self.playerVelY, self.ground_point[1] - \
                                    self.player_point[1] - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = self._get_random_pipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < - self.pipe_width:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        state = self._get_state()
        return state, reward, done
    
    def render(self):
        
        # init self.images
        ## numbers images for score display
        self.images['number'] = []
        for number_path in self.number_list:
            self.images['number'].append(pygame.image.load(number_path).convert_alpha())
        ## ground images for ground on the bottom
        self.images['ground'] = pygame.image.load(self.ground_path).convert_alpha()
        ## random select background images
        randbg = 1 # random.randint(0,1)
        self.images['background'] = \
            pygame.image.load(self.background_list[randbg]).convert_alpha()
        ## random select player images
        self.images['player'] = []
        randpl = 1 # random.randint(0,2)
        for info_path in self.player_list[randpl]:
            self.images['player'].append(pygame.image.load(info_path).convert_alpha())
        ## random select pipe images
        self.images['pipe'] = []
        randpp = 1 # random.randint(0,1)
        self.images['pipe'] = [pygame.transform.rotate(\
            pygame.image.load(self.pipe_list[randpp]).convert_alpha(), 180), \
            pygame.image.load(self.pipe_list[randpp]).convert_alpha()]
        
        # draw sprites
        self.window.blit(self.images['background'], (0,0))
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            self.window.blit(self.images['pipe'][0], (uPipe['x'], uPipe['y']))
            self.window.blit(self.images['pipe'][1], (lPipe['x'], lPipe['y']))
        self.window.blit(self.images['ground'], (self.ground_point[0], self.ground_point[1]))
        self.window.blit(self.images['player'][self.playerIndex], \
                         (self.player_point[0], self.player_point[1]))
        
        # print score so player overlaps the score
        self._show_score(self.score)

        pygame.display.flip()
        self.fps_clock.tick(self.fps)
        
    def _get_state(self):
        to_ground_height = self.ground_point[1] - self.player_point[1] - self.player_height
        to_pipe_distance = self.upperPipes[0]['x'] - self.player_point[0] - self.player_width
        to_pipe_left_distance = self.upperPipes[0]['x'] - self.player_point[0] - self.player_width
        to_pipe_right_distance = self.upperPipes[0]['x'] + self.pipe_width - \
            (self.player_point[0] + self.player_width)
        if to_pipe_right_distance >= 0:
            to_lower_pipe_height = self.lowerPipes[0]['y'] - \
                self.player_point[1] - self.player_height
            to_upper_pipe_height = self.player_point[1] - \
                (self.upperPipes[0]['y'] + self.pipe_height) - self.player_height
        else:
            to_lower_pipe_height = self.lowerPipes[1]['y'] - \
                self.player_point[1] - self.player_height
            to_upper_pipe_height = self.player_point[1] - \
                (self.upperPipes[1]['y'] + self.pipe_height) - self.player_height
        return [to_ground_height, to_pipe_left_distance, to_pipe_right_distance, \
                to_lower_pipe_height, to_upper_pipe_height, self.playerVelY]
        
    def _set_player_shm(self, playerShm):
        """oscillates the value of playerShm['val'] between 8 and -8"""
        if abs(playerShm['val']) == 8:
            playerShm['dir'] *= -1
    
        if playerShm['dir'] == 1:
            playerShm['val'] += 1
        else:
            playerShm['val'] -= 1
            
    def _get_random_pipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gap_y = random.randrange(0, int(self.ground_point[1] * 0.6 - self.pipe_gap))
        gap_y += int(self.player_point[1] * 0.2)
        print('gap_y: %d' % (gap_y))
        pipe_height = self.pipe_height
        pipe_x = self.width + 10
    
        return [{'x': pipe_x, 'y': gap_y - pipe_height}, \
                {'x': pipe_x, 'y': gap_y + self.pipe_gap}]
        
    def _check_crash(self, player, upperPipes, lowerPipes):
        """returns True if player collders with base or pipes."""
        pi = player['index']
        player['w'] = self.player_width
        player['h'] = self.player_height
    
        # if player crashes into ground
        if player['y'] + player['h'] >= self.ground_point[1] - 1:
            return [True, True]
        else:
    
            # playerRect = pygame.Rect(player['x'], player['y'], player['w'], player['h'])
            playerRect = [player['x'], player['y'], player['w'], player['h']]
            pipeW = self.pipe_width
            pipeH = self.pipe_height
    
            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                # uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
                # lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)
                uPipeRect = [uPipe['x'], uPipe['y'], pipeW, pipeH]
                lPipeRect = [lPipe['x'], lPipe['y'], pipeW, pipeH]
    
                # player and upper/lower pipe hitmasks
                # pHitMask = self.hit_mask['player'][pi]
                # uHitmask = self.hit_mask['pipe'][0]
                # lHitmask = self.hit_mask['pipe'][1]
    
                # if bird collided with upipe or lpipe
                uCollide = self._collision_examine(playerRect, uPipeRect)#, pHitMask, uHitmask)
                lCollide = self._collision_examine(playerRect, lPipeRect)#, pHitMask, lHitmask)
    
                if uCollide or lCollide:
                    return [True, False]
    
        return [False, False]
    
    def _show_score(self, score):
        """displays score in center of screen"""
        scoreDigits = [int(x) for x in list(str(score))]
        totalWidth = 0 # total width of all numbers to be printed
        for digit in scoreDigits:
            totalWidth += self.images['number'][digit].get_width()
    
        Xoffset = (self.width - totalWidth) / 2
        for digit in scoreDigits:
            self.window.blit(self.images['number'][digit], (Xoffset, self.height * 0.1))
            Xoffset += self.images['number'][digit].get_width()

    def _collision_examine(self, rect1, rect2): #, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        # rect1 = pygame.Rect(*rect1)
        # rect2 = pygame.Rect(*rect2)
        minx = max(rect1[0], rect2[0])
        miny = max(rect1[1], rect2[1])
        maxx = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
        maxy = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
        # rect = rect1.clip(rect2)
        if (minx > maxx) or (miny > maxy):
            return False
        else:
            return True
        
        """
        if rect.width == 0 or rect.height == 0:
            return False
        else:
            return True
    
        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y
    
        for x in xrange(rect.width):
            for y in xrange(rect.height):
                if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                    return True
        return False
        """

    """
    def _get_hitmask(self, width, height):
        # returns a hitmask using an image's alpha.
        mask = []
        for x in range(width):
            mask.append([])
            for _ in range(height):
                mask[x].append(True)
        return mask
    """


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

class Env:
    
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
        random.seed(0)

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

    def render(self):
        # 判断是否flap
        if self.is_show:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and \
                    (event.key == pygame.K_SPACE or event.key == pygame.K_UP):
                    self.bird_flap = True

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
        ## 判断地板碰撞
        if self._is_object_union(self.bird, self.ground):
            is_crash = True
        ## 判断和水管碰撞
        for up_pipe, down_pipe in self.pipe_queue:
            if self._is_object_union(self.bird, up_pipe):
                is_crash = True
            if self._is_object_union(self.bird, down_pipe):
                is_crash = True
        if is_crash:
            self.is_end = True

        # bird通过pipe检测
        for up_pipe, down_pipe in self.pipe_queue:
            if not down_pipe.is_passed:
                if self.bird.pos[0] >= down_pipe.pos[0] + down_pipe.size[0]:
                    down_pipe.is_passed = True
                    self.n_score += 1

        if self.is_show:
            self.show(n_frame=self.n_frame)
        self.save_image(n_frame=self.n_frame)

        # 变换到下一帧
        self.n_frame += 1

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

    def save_image(self, n_frame=0):
        # 将各个object放置在canvas中
        self._set_object(self.canvas, self.background)
        self._set_object(self.canvas, self.bird)
        for up_pipe, down_pipe in self.pipe_queue:
            self._set_object(self.canvas, up_pipe)
            self._set_object(self.canvas, down_pipe)
        self._set_object(self.canvas, self.ground)
        print(self.canvas.shape)
        cv2.imwrite('D:/Downloads/flappy/%d.png' % (n_frame), self.canvas)

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
    env = Env(is_show=False)
    for i in range(500):
        if not env.is_end:
            env.render()

main()