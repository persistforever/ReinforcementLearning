# 赌徒问题

## 问题描述

一个赌徒抛硬币下赌注，如果硬币正面朝上，他本局将赢得和下注数量相同的钱，如果硬币背面朝上，他本局将输掉下注的钱，当他输光所有的赌资或者赢得$100则停止赌博，硬币正面朝上的概率为p。赌博过程是一个无折扣的有限的马尔科夫决策问题。


## 问题求解

首先提取问题的重要元素

**状态** 表示当前的赌资。赢得100停止赌博，因此状态集合为[1, 2, ..., 99, 100]。

**动作** 表示本局下注的数量。设当前状态为s，则最多能下注s，又由于赢得100则获胜，那么最多只需下注100-s，因此，动作集合为[1, 2, ..., min(s, 100-s)]。

**转移概率** 设当前状态为s，动作为a，那么转移到状态s-a的概率为1-p，转移到状态s+a的概率为p，转移到其他状态的概率为0。

**奖励** 如果下一个状态为100，则奖励为1，否则为0。

**策略** 表示在当前有s赌资的状态下，下注a的概率。

然后根据值迭代算法。

1. 初始化状态值函数value_function（100个状态对应的值函数均为0）。
2. 记录当前的状态值函数为last_value_function，对每个状态，进行状态值函数的更新，更新后的状态值函数为value_function。
3. 循环步骤2，如果value_function和last_value_function的差值小于阈值delta，则跳出循环。
4. 根据状态值函数，对每个状态进行策略更新，更新后的策略为policy。

## 实验结果

### 值迭代算法

在值迭代的循环中，每步计算出的状态值函数如下。

<img width="50%" height="50%" src="https://github.com/persistforever/ReinforcementLearning/blob/master/gambler/experiments/value1.png?raw=true">

算出的最优策略如下。

<img width="50%" height="50%" src="https://github.com/persistforever/ReinforcementLearning/blob/master/gambler/experiments/policy1.png?raw=true">

### 策略迭代算法

在策略迭代的循环中，每步计算出的状态值函数如下。

<img width="50%" height="50%" src="https://github.com/persistforever/ReinforcementLearning/blob/master/gambler/experiments/value2.png?raw=true">

算出的最优策略如下。

<img width="50%" height="50%" src="https://github.com/persistforever/ReinforcementLearning/blob/master/gambler/experiments/policy2.png?raw=true">

### 效率对比

策略迭代算法耗时0.334秒，值函数计算次数为525次。

值迭代算法耗时0.044秒，值函数计算次数为27次。

而两者算出的目标策略和目标值函数几乎相同，因此得出结论，值迭代算法比策略迭代算法效率更高。
