#!/bin/sh

name=space_invaders_v1
ngpu=1
config=scripts/$name/config_dqn.yaml


## train
nohup python -u -m space_invaders.script -method train -name dqn -config $config > /home/caory/data/PDFInsight/deep_rl/logs/$name/train_logs/dqn.txt 2>&1 &