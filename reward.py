# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 22:13:21 2025

@author: 503
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

accumulative_reward_d = np.load('accumulative_reward1.npy')
accumulative_reward = np.load('accumulative_reward0.npy')
plt.figure(figsize=(10, 8))     # 输出累计奖励图像
plt.title('Accumulative Reward',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Episodes',fontsize=20)
plt.ylabel('Accumulative Reward',fontsize=20)
plt.plot(np.arange(len(accumulative_reward_d)), accumulative_reward_d,color='red',label='QoSA-DQN')
plt.plot(np.arange(len(accumulative_reward)), accumulative_reward,color='blue',label='     DQN')
plt.rcParams.update({'font.size': 20})     #设置图例字体大小
plt.legend()
plt.grid(True)
plt.savefig('reward_com.png')