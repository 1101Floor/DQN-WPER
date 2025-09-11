# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:13:26 2024

@author: Derrick-Rose
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
#
loss_list_d = np.load('loss_listRPTD.npy')
#loss_list = np.load('loss_listp2.npy')
# loss_list = np.array(loss_list).flatten()
y_d = loss_list_d
#y = loss_list[:39795]
x = np.linspace(0, len(loss_list_d), len(loss_list_d))
x_start = 0
x_end = 1000
plt.figure(figsize=(10, 8))
plt.title('Loss',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Episodes',fontsize=20)
plt.ylabel('Loss',fontsize=20)
#plt.plot(loss_list)
plt.plot(x, y_d, color='orange',label='original QoSA-DQN')
#plt.plot(x, y, color='lightblue',label='original DQN')
z_d = savgol_filter(y_d, 1003, 5, mode= 'nearest')
#z = savgol_filter(y, 1003, 5, mode= 'nearest')
# 可视化图线
plt.plot(x, z_d, color='red',label='savgol QoSA-DQN')
#plt.plot(x, z, color='blue',label='savgol DQN')
plt.savefig('loss_listRPTD.png')
#显示曲线
plt.rcParams.update({'font.size': 20})     #设置图例字体大小
plt.legend()
plt.grid(True)
plt.xlim(x_start, x_end)
plt.show()

# slope, intercept = np.polyfit(x, y, 1)
# trendline = slope * x + intercept

# # 绘制趋势线
# plt.plot(x, trendline, color='red', linestyle='--', label='Trendline')

# # 添加图例和标签
# plt.legend()
# plt.title('Data with Trendline')
# plt.xlabel("epoch")
# plt.ylabel("loss")
# # 显示图形
# plt.savefig('Loss.png')
# plt.show()
