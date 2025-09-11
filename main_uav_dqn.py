import matplotlib.pyplot as plt
import numpy as np
import math
import time
import random
from RL_brain import DQN

#地图：50*50
angle_phi_1 = math.radians(60)
m = -math.log(2) / math.log(math.cos(angle_phi_1))
P = 6e-2
angle_Phi_c = math.radians(60)
A = 1e-3
xi = 0.8
n_r = 1.5
height = 18
sigma_w = 1e-10

#GT位置参数
GT_1 = np.array([28,19,0]);GT_2 = np.array([27,27,0])
GT_3 = np.array([39,31,0]);GT_4 = np.array([34,36,0])
GT_5 = np.array([26,35,0])
GT_list = [GT_1,GT_2,GT_3,GT_4,GT_5]
#障碍物位置参数
OP_1 = np.array([13,8,height]);OP_2 = np.array([11,18,height])
OP_3 = np.array([10,26,height]);OP_4 = np.array([14,38,height])
OP_5 = np.array([26,9,height]);OP_6 = np.array([29,18,height])
OP_7 = np.array([20,22,height]);OP_8 = np.array([32,26,height])
OP_list = [OP_1,OP_2,OP_3,OP_4,OP_5,OP_6,OP_7,OP_8]



def calculate_incidence_angle(drone_pos, receiver_pos, normal_vector=np.array([0, 0, 1])):
    # 计算方向向量并单位化
    direction_vector = (drone_pos - receiver_pos) / np.linalg.norm(drone_pos - receiver_pos)
    # 计算两个向量的点积
    dot_product = np.dot(direction_vector, normal_vector)
    # 计算cosθ
    cos_theta = dot_product
    # 计算角度θ（弧度）
    theta = np.arccos(cos_theta)
    return theta
# 计算距离d
def distance(x,y):
    vector_difference = x - y
    d_i = np.linalg.norm(vector_difference)
    return d_i
#计算通信容量C
def capacity(s):
    result_C = 0
    for i in range(5):
        angle_phi_i = calculate_incidence_angle(s,GT_list[i])
        d_i = distance(s,GT_list[i])
        if angle_phi_i < angle_Phi_c:
            g_phi = n_r**2/(math.sin(angle_Phi_c))**2
        else:
            g_phi = 0
        h_i = ((m+1)*A/(2*math.pi*(d_i)**2))*g_phi*((math.cos(angle_phi_i))**m)*math.cos(angle_phi_i)
        C_i = 1/2*math.log(1+(math.e/(2*math.pi))*(xi*P*h_i/sigma_w)**2,2)
        result_C = result_C+C_i
    return result_C

#计算碰撞风险R
def risk(s):
    result_r = 1
    for i in range(8):
        sigma = [2, 2, 2, 2, 2, 2, 2, 2]  # 后续可以调整
        d_i = distance(s,OP_list[i])
        r_i = (1/((math.sqrt(2*math.pi))*sigma[i]))*math.exp((-(d_i)**2)/(2*(sigma[i])**2))
        result_r = result_r*(1-r_i)
    result_R =1- result_r
    return result_R
#撞上障碍物
def accident(s):
    result = any(np.array_equal(s, op) for op in OP_list)
    return result

com_num = '1'
com_num_use = int(com_num)
episodes = 100
x = np.linspace(0, 50, 101)   # 设置x轴离散点
y = np.linspace(0, 50, 101)   # 设置y轴离散点
z = np.linspace(0, 20, 20)      # 设置z轴离散点
fly_height = z[18]

accumulative_reward = np.zeros(episodes).astype('float32')      # 累计奖励函数
x_out = np.zeros(int(com_num) * 3000).astype('float32')
y_out = np.zeros(int(com_num) * 3000).astype('float32')
z_out = np.zeros(int(com_num) * 3000).astype('float32')
cap_out = np.zeros(int(com_num) * 3000).astype('float32')
risk_out = np.zeros(int(com_num) * 3000).astype('float32')

counter3 = 0
def reset():        # 重置函数
    time.sleep(0.1)
    # print('方案1重新开始!')
    X_counter = 0
    Y_counter = 0
    Z_counter = 18
    Observation = np.array([x[X_counter], y[Y_counter], z[Z_counter]])
    Game_over = 0           # 0代表学习没有结束，1代表学习结束
    return Observation, X_counter, Y_counter, Z_counter, Game_over

def step(Action, X_counter, Y_counter, Z_counter, Game_over):      #动作函数
    if Game_over == 2:
        Observation_ = np.array([x[X_counter], y[Y_counter], z[Z_counter]])
        Reward = 500
        Over = False
        return Observation_, Reward, Over, X_counter, Y_counter, Z_counter
    else:
        Action_counter_ary = [X_counter, Y_counter, Z_counter]
        Observation = np.array([x[X_counter], y[Y_counter], z[Z_counter]])
        Over = None
        hit_wall = None
        if Action == 0:     # left x减1
            if Action_counter_ary[0] == 0:
                Action_counter_ary[0] = Action_counter_ary[0] - 0
                Over = True  # 撞墙game over
                hit_wall = 1
            else:
                Action_counter_ary[0] = Action_counter_ary[0] - 1
                Over = False  # 继续学习
                hit_wall = 0
        if Action == 1:     # right x加1
            if Action_counter_ary[0] == 100:
                Action_counter_ary[0] = Action_counter_ary[0] + 0
                Over = True  # 撞墙game over
                hit_wall = 1
            else:
                Action_counter_ary[0] = Action_counter_ary[0] + 1
                Over = False  # 继续学习
                hit_wall = 0
        if Action == 2:     # forward y加1
            if Action_counter_ary[1] == 100:
                Action_counter_ary[1] = Action_counter_ary[1] + 0
                Over = True  # 撞墙game over
                hit_wall = 1
            else:
                Action_counter_ary[1] = Action_counter_ary[1] + 1
                Over = False  # 继续学习
                hit_wall = 0
        if Action == 3:     # backward y减1
            if Action_counter_ary[1] == 0:
                Action_counter_ary[1] = Action_counter_ary[1] - 0
                Over = True  # 撞墙game over
                hit_wall = 1
            else:
                Action_counter_ary[1] = Action_counter_ary[1] - 1
                Over = False  # 继续学习
                hit_wall = 0
        if Action == 4:     # hold 坐标不变
            Action_counter_ary[0] = Action_counter_ary[0]
            Action_counter_ary[1] = Action_counter_ary[1]
            Action_counter_ary[2] = Action_counter_ary[2]
            Over = False    # 继续学习
            hit_wall = 0
        X_counter = Action_counter_ary[0]
        Y_counter = Action_counter_ary[1]
        Z_counter = Action_counter_ary[2]
        Observation_ = np.array([x[Action_counter_ary[0]], y[Action_counter_ary[1]], z[Action_counter_ary[2]]])
        #奖励部分
        if hit_wall == 1 :       # 撞墙和障碍物给负奖励
            Reward = -10
        elif accident(Observation_):
            Reward = -50
            Over = True
        elif capacity(Observation_) < capacity(Observation):       # 通信容量变小给负奖励
            Reward = -2-0.1*risk(Observation_)
        elif capacity(Observation_) > capacity(Observation):       # 通信容量变大给正奖励
            Reward = 1.9-0.1*risk(Observation_)
        else:
            Reward = 0
        return Observation_, Reward, Over, X_counter, Y_counter, Z_counter    # 返回下一状态和奖励值

def run():
    run_step = 0
###########################################################
    for epi in range(episodes):      # 玩100次
        print('重新开始epi:', end=' ')
        print(epi)
        counter1 = 0
        reward_acc = 0      # 累计奖励计算
        observation, x_counter, y_counter, z_counter, game_over= reset()      # 位置重置

        while True:
            action = RL.choose_action(observation)      # 动作选择
            observation_, reward, done, x_counter, y_counter, z_counter= \
                step(action, x_counter, y_counter, z_counter, game_over)
            RL.store_transition(observation, action, reward, observation_, done)
            reward_acc = reward_acc + reward        # 计算累计奖励值
            if (run_step > 1024) and (run_step % 5 == 0):
                RL.learn()
            # print(observation)
            observation = observation_
            x_out[counter1] = observation[0]        # 绘制轨迹
            y_out[counter1] = observation[1]
            z_out[counter1] = observation[2]
            cap_out[counter1] = capacity(observation)
            risk_out[counter1] = risk(observation)
            run_step += 1
            counter1 += 1
            if done is True:        # 超出了空间位置
                observation, x_counter, y_counter, z_counter, game_over = reset()      # 位置重置
                reward_acc = 0
            if counter1 == int(com_num) * 3000:
                # print('学习收敛!')
                game_over = 1       # 为了要那个学习成功的奖励值
                observation_, reward, done, x_counter, y_counter, z_counter = \
                    step(action, x_counter, y_counter, z_counter, game_over)
                RL.store_transition(observation, action, reward, observation_, done)
                reward_acc = reward_acc + reward
                accumulative_reward[epi] = reward_acc
                break
    print('最后静止的坐标为:', end=' ')
    print(observation)

    # np.save('x_outRPTD.npy', x_out)
    # np.save('y_outRPTD.npy', y_out)
    # np.save('cap_outRPTD.npy', cap_out)
    # np.save('risk_outRPTD.npy', risk_out)
    # np.save('reward_outRP6TD.npy', accumulative_reward)


    # plt.figure(figsize=(13, 6))     # 输出累计奖励图像
    # plt.title('Accumulative Reward')
    # plt.xlabel('Episodes')
    # plt.ylabel('Accumulative Reward')
    # plt.plot(np.arange(len(accumulative_reward)), accumulative_reward)
    # plt.savefig('rewardRPTD.png')

    # plt.figure(figsize=(13, 6))   #画容量变化图
    # plt.title('Capacity')
    # plt.xlabel('Times')
    # plt.ylabel('Capacity')
    # plt.plot(np.arange(len(cap_out)),cap_out)
    # plt.savefig('CapacityRPTD.png')

    # plt.figure(figsize=(13, 6))  # 画风险变化图
    # plt.title('Risk')
    # plt.xlabel('Times')
    # plt.ylabel('Risk')
    # # 只取前500个数据点
    # risk_out_slice = risk_out[:500]
    # times_slice = np.arange(500)  # 假设np.arange本来就是从0开始，因此也取前500个
    # plt.plot(times_slice, risk_out_slice)  # 修改这里以匹配切片后的数据
    # plt.savefig('Risk_firstRPTD.png')  # 保存图片并命名以反映其包含的数据范围


RL = DQN()        # 设置机器学习参数
run()       # 运行机器学习程序
