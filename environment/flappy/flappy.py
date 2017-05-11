from itertools import cycle
import random
import os

import pygame
from pygame.locals import *


class Environment:
    
    def __init__(self):
        # basic information
        dir = os.path.split(os.path.realpath(__file__))[0]
        # configuration
        self.height = 512
        self.width = 288
        self.fps = 100
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
        
    def reset(self):
        pygame.init()
        self.fps_clock = pygame.time.Clock()
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Flappy Bird')
        
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
        
        # init self.hit_mask
        ## hitmask for pipe
        self.hit_mask['pipe'] = (self._get_hitmask(self.images['pipe'][0]), \
                                 self._get_hitmask(self.images['pipe'][1]))
        ## hit mask for player
        self.hit_mask['player'] = (self._get_hitmask(self.images['player'][0]), \
                                   self._get_hitmask(self.images['player'][1]), \
                                   self._get_hitmask(self.images['player'][2]))
        
        # init player information
        self.loopIter = 0
        self.playerIndex = 0
        self.player_index_gen = cycle([0, 1, 2, 1])
        self.player_point = [int(self.width * 0.2), \
                             int((self.height - self.images['player'][0].get_height()) / 2)]
        ## bird motion information
        self.player_shm_vals = {'val': 0, 'dir': 1}
        ## player velocity, max velocity, downward accleration, accleration on flap
        self.playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerAccY    =   1   # players downward accleration
        self.playerFlapAcc =  -9   # players speed on flapping
        self.playerFlapped = False # True when player flaps
        
        # init ground information
        self.ground_point = [0, self.height*0.8]
        ## amount by which base can maximum shift to left
        self.ground_shift = self.images['ground'].get_width() - self.images['background'].get_width()
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
        for event in pygame.event.get():
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                action = 'fly'
        if action == 'fly':
            if self.player_point[1] > -2 * self.images['player'][0].get_height():
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
        playerMidPos = self.player_point[0] + self.images['player'][0].get_width() / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + self.images['pipe'][0].get_width() / 2
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
        playerHeight = self.images['player'][self.playerIndex].get_height()
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
        if self.upperPipes[0]['x'] < - self.images['pipe'][0].get_width():
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        state = self._get_state()
        return state, reward, done
    
    def render(self):
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
        to_ground_height = self.ground_point[1] - self.player_point[1] - \
            self.images['player'][self.playerIndex].get_height()
        to_pipe_distance = self.upperPipes[0]['x'] - self.player_point[0] - \
            self.images['player'][self.playerIndex].get_width()
        if to_pipe_distance >= - self.images['pipe'][0].get_width():
            to_lower_pipe_height = self.player_point[1] - \
                (self.lowerPipes[0]['y']-self.images['player'][self.playerIndex].get_height()) - \
                self.images['player'][self.playerIndex].get_height()
            to_upper_pipe_height = self.player_point[1] - \
                (self.upperPipes[0]['y'] + self.images['pipe'][0].get_height())
        else:
            to_pipe_distance = self.upperPipes[1]['x'] - self.player_point[0] + \
                self.images['pipe'][1].get_width()
            to_lower_pipe_height = self.player_point[1] - \
                (self.lowerPipes[1]['y']-self.images['player'][self.playerIndex].get_height()) - \
                self.images['player'][self.playerIndex].get_height()
            to_upper_pipe_height = self.player_point[1] - \
                (self.upperPipes[1]['y'] + self.images['pipe'][1].get_height())
        return [to_ground_height, to_pipe_distance, \
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
        pipe_height = self.images['pipe'][0].get_height()
        pipe_x = self.width + 10
    
        return [{'x': pipe_x, 'y': gap_y - pipe_height}, \
                {'x': pipe_x, 'y': gap_y + self.pipe_gap}]
        
    def _check_crash(self, player, upperPipes, lowerPipes):
        """returns True if player collders with base or pipes."""
        pi = player['index']
        player['w'] = self.images['player'][0].get_width()
        player['h'] = self.images['player'][0].get_height()
    
        # if player crashes into ground
        if player['y'] + player['h'] >= self.ground_point[1] - 1:
            return [True, True]
        else:
    
            playerRect = pygame.Rect(player['x'], player['y'],
                          player['w'], player['h'])
            pipeW = self.images['pipe'][0].get_width()
            pipeH = self.images['pipe'][0].get_height()
    
            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)
    
                # player and upper/lower pipe hitmasks
                pHitMask = self.hit_mask['player'][pi]
                uHitmask = self.hit_mask['pipe'][0]
                lHitmask = self.hit_mask['pipe'][1]
    
                # if bird collided with upipe or lpipe
                uCollide = self._collision_examine(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self._collision_examine(playerRect, lPipeRect, pHitMask, lHitmask)
    
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

    def _collision_examine(self, rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)
    
        if rect.width == 0 or rect.height == 0:
            return False
    
        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y
    
        for x in xrange(rect.width):
            for y in xrange(rect.height):
                if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                    return True
        return False

    def _get_hitmask(self, image):
        """returns a hitmask using an image's alpha."""
        mask = []
        for x in range(image.get_width()):
            mask.append([])
            for y in range(image.get_height()):
                mask[x].append(bool(image.get_at((x,y))[3]))
        return mask