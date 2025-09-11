# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 21:55:53 2025

@author: 503
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
#
risk_out = np.load('risk_out_1.npy')
risk_out_d = np.load('risk_out3.npy')
plt.figure(figsize=(10, 8))  # 画风险变化图
plt.title('Risk',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Times',fontsize=20)
plt.ylabel('Risk',fontsize=20)
# 只取前500个数据点
risk_out_slice = risk_out[:499]
risk_out_slice_d = risk_out_d[:499]
times_slice = np.arange(499)  # 假设np.arange本来就是从0开始，因此也取前500个
plt.plot(times_slice, risk_out_slice,color='blue',label='    DQN')  # 修改这里以匹配切片后的数据
plt.plot(times_slice, risk_out_slice_d,color='red',label='QoSA-DQN')  # 修改这里以匹配切片后的数据
plt.rcParams.update({'font.size': 20})     #设置图例字体大小
plt.legend()
plt.grid(True)
plt.savefig('Risk_out_com.png')  # 保存图片并命名以反映其包含的数据范围