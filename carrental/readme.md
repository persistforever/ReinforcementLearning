# 车辆租赁问题


## 问题描述
Jack在A和B两个地点拥有车辆租赁系统，每天，一些客户在两个地点分别出租一定的车辆，每辆车的出租费用为$10，如果Jack在某个地点的车被租完了，他就会流失之后租车的客户。白天，客户先租车，然后客户换车，夜晚，Jack会安排员工将车辆进行移动（从A移动到B或相反），每辆车移动费用为$2，假设每个地点借车和换车的人数服从参数为lambda的泊松分布，A地借车人数、A地还车人数、B地借车人数、B地还车人数的参数lambda分别为3、3、4、2（为了简化问题，假设A地和B地换车的人数为固定值3和4而不是泊松分布）。假设每个地点最多有20辆车，每天夜晚最多有5辆车在A、B两地移动，设该马尔科夫决策问题的参数gamma=0.9。


## 问题求解

首先提取问题的重要元素

**状态** 表示A点和B点的车辆个数。由于2地最多20辆车，因此状态集合大小为20*20=400。

**动作** 表示每天晚上从A点移动到B点的车辆个数（大于0表示从A移动到B，小于0表示从B移动到A，绝对值表示移动车辆的个数）。合法动作根据状态而定，假设每天晚上移动车辆动作之前的状态为，A地m辆车，B地n辆车，又由于最多移动5辆车，因此动作集合为[-min(20-m, n, 5), min(20-n, m, 5)]。

**转移概率** 表示在A地m辆车，B地n辆车的状态下，发生动作k，然后A地被借走a辆车（a<=m），B地被借走b辆车（b<=n），即从A地移动到B地k辆车，那么更新的状态为A地 m-k-a+min(a,3) ，B地 n+k-b+min(b,2) ，则该状态对应的转移概率为 poisson(a,3)\*poisson(b,4) 。

**奖励** 表示在A地m辆车，B地n辆车的状态下，发生动作k，然后A地被借走a辆车（a<=m），B地被借走b辆车（b<=n），即从A地移动到B地k辆车，那么更新的状态为A地m- k-a+min(a,3) ，B地 n+k-b+min(b,2) ，则该状态对应的奖励为 k\*(-2)+(a+b)\*10。

**策略** 表示在A地m辆车，B地n辆车的状态下，发生动作k的概率。

然后根据策略迭代算法。

1. 初始化状态值函数value_function（400个状态对应的值函数均为0）和策略policy（对于每个状态，动作0对应的概率为1，其他动作对应的概率均为0）。
2. 记录当前的状态值函数为last_value_function，对每个状态，进行状态值函数的更新，更新后的状态值函数为value_function。
3. 循环步骤2，如果value_function和last_value_function的差值小于阈值delta，则跳出循环。
4. 记录当前的策略为last_policy，对每个状态进行策略更新，更新后的策略为policy。
5. 循环步骤2-4，如果policy和last_policy完全相同，则跳出循环。


## 实验结果

在策略迭代的循环中，每步计算出的状态值函数如下。

<img width="40%" height="40%" src="https://github.com/persistforever/ReinforcementLearning/blob/master/carrental/experiments/value1.png?raw=true">
<img width="40%" height="40%" src="https://github.com/persistforever/ReinforcementLearning/edit/master/carrental/experiments/value2.png?raw=true">
<img width="5%" height="5%" src="https://github.com/persistforever/ReinforcementLearning/edit/master/carrental/experiments/value3.png?raw=true">
<img width="5%" height="5%" src="https://github.com/persistforever/ReinforcementLearning/edit/master/carrental/experiments/value4.png?raw=true">

每步算出的策略如下。
