# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 22:10:16 2025

@author: 503
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

cap_out_d = np.load('cap_out3.npy')
cap_out = np.load('cap_out4.npy')
plt.figure(figsize=(10, 8))   #画容量变化图
plt.title('Capacity',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Times',fontsize=20)
plt.ylabel('Capacity',fontsize=20)
plt.plot(np.arange(len(cap_out_d)),cap_out_d,color='red',label='QoSA-DQN')
plt.plot(np.arange(len(cap_out)),cap_out,color='blue',label='    DQN')
plt.rcParams.update({'font.size': 20})     #设置图例字体大小
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('Capacity_com.png')

