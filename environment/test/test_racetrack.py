# -*- encoding = utf8 -*-
import os
from environment.racetrack import Environment


def observe():
    env = Environment()
    state, reward, done = env.reset(), 0, True
    n = 0
    for i in range(100):
        n += 1
        env.step(action=[-1 if n % 10 == 0 else 0, 1 if n % 30 == 0 else 0])
        env.render()
        
        
observe()