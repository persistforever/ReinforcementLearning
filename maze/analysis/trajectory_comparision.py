# -*- encoding = utf8 -*-
import numpy


class Analysis:
    
    def _read_log(self, path):
        state = 'LINE'
        with open(path, 'r') as fo:
            for line in fo:
                if state == 'LINE':
                    if 'tries' in line:
                        n_try = int(line.strip().split(' ')[2])
                    elif 'seconds' in line:
                        n_time = float(line.strip().split(' ')[2])
                    elif 'try path' in line:
                        traj_list = []
                        state = 'TRAJ'
                elif state == 'TRAJ':
                    if 'win path' in line:
                        state = 'WIN'
                    else:
                        traj_list.append(line.strip())
                elif state == 'WIN':
                    win_traj = line.strip()
        return n_try, n_time, traj_list, win_traj
    
    def _analyze_one_log(self, path):
        n_try, n_time, traj_list, win_traj = self._read_log(path)
        death_rate = self._analyze_traj(traj_list, win_traj)
        
        return n_try, n_time, death_rate
    
    def _analyze_traj(self, traj_list, win_traj):
        n_death, n_path = 0, len(traj_list)
        for traj in traj_list:
            if not self._is_prefix(traj, win_traj):
                n_death += 1
                
        return 1.0 * n_death / n_path
            
    def _is_prefix(self, target, source):
        if len(target) > len(source):
            is_prefix = False
        else:
            is_prefix = True
            for i in range(len(target)):
                if target[i] != source[i]:
                    is_prefix = False
                    break
            else:
                is_prefix = True
        
        return is_prefix
    
    def analyze_all_log(self, path, mode='try_mean'):
        try_list, time_list, death_list = [], [], []
        for i in range(20):
            n_try, n_time, death_rate = self._analyze_one_log(path + str(i) + '.txt')
            try_list.append(n_try)
            time_list.append(n_time)
            death_list.append(death_rate)
        if mode == 'effective':
            print '%.0f' % (1.0 * sum(try_list) / sum(time_list))
        elif mode == 'death_rate':
            print '%.6f' % (numpy.array(death_list).mean())
        elif mode == 'try_mean':
            print '%.0f' % (numpy.array(try_list).mean())
        elif mode == 'try_var':
            print '%.0f' % (numpy.array(try_list).var())
    
    
analysis = Analysis()
analysis.analyze_all_log('../experiments/monte_carlo_control/trajectory/greedy-policy_(15_15)_')
analysis.analyze_all_log('../experiments/monte_carlo_control/trajectory/on-policy_(15_15)_')
analysis.analyze_all_log('../experiments/monte_carlo_control/trajectory/off-policy_(15_15)_')
analysis.analyze_all_log('../experiments/temporal_difference/trajectory/SARSA_(15_15)_')
analysis.analyze_all_log('../experiments/temporal_difference/trajectory/QLearning_(15_15)_')