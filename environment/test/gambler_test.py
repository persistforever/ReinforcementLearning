# -*- encoding = utf8 -*-
import os
from environment.gambler import Environment


def observe():
    env = Environment()
    state = env.reset()
    print state, 0, False
    for i in range(100):
        new_state, reward, done = env.step(action=i)
        print new_state, reward, done
        env.render()
        if done:
            break
        
        
observe()