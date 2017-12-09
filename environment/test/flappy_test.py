# -*- encoding = utf8 -*-
import os
import time
from environment.flappy import Environment


def observe():
    print os.getcwd()
    env = Environment()
    env.reset(seed=0)
    for i in range(200):
        state, reward, done = env.step(action='fly' if i % 19 == 0 else 'stay')
        print state, reward, done
        env.render()
        # time.sleep(0.5)
        if done:
            break
        
        
observe()