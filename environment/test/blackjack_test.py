# -*- encoding = utf8 -*-
from environment.blackjack import Environment


def observe():
    env = Environment()
    state = env.reset()
    print state
    for _ in range(100):
        new_state, reward, done = env.step(action='hit' if state < 17 else 'stick')
        state = new_state
        print new_state, reward, done
        # env.render()
        if done:
            break
        
        
observe()