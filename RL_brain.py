import os
import torch                                    # 导入torch
import torch.nn as nn                           # 导入torch.nn
import torch.nn.functional as F                 # 导入torch.nn.functional
import numpy as np                              # 导入numpy
import gym
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 超参数
BATCH_SIZE = 512                                # 样本数量
LR = 0.0001                                    # 学习率

EPSILON_START = 0.75  # epsilon 的初始值
EPSILON_END = 0.01  # epsilon 的最终值
EPSILON_DECAY = 1000 # epsilon 的衰减步数
EPSILON_MAX = 0.9                               # greedy policy

GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = pow(2, 15)                # 目标网络更新频率
# TARGET_REPLACE_ITER = 10000                     # 目标网络更新频率
MEMORY_CAPACITY = pow(2, 15)                    # 记忆库容量
N_ACTIONS = 5                                   # 无人船动作个数 (5个)
N_STATES = 3                                    # 无人船状态个数 (3个)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
loss_list = []

# =============================================================================
# SumTree
class SumTree:
    def __init__(self, capacity: int):
        # 初始化SumTree，设定容量
        self.capacity = capacity
        # 数据指针，指示下一个要存储数据的位置
        self.data_pointer = 0
        # 数据条目数
        self.n_entries = 0
        # 构建SumTree数组，长度为(2 * capacity - 1)，用于存储树结构
        self.tree = np.zeros(2 * capacity - 1)
        # 数据数组，用于存储实际数据
        self.data = np.zeros(capacity, dtype=object)

    def update(self, tree_idx, p):#更新采样权重
        # 计算权重变化
        change = p - self.tree[tree_idx]
        # 更新树中对应索引的权重
        self.tree[tree_idx] = p

        # 从更新的节点开始向上更新，直到根节点
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, p, data):#向SumTree中添加新数据
        # 计算数据存储在树中的索引
        tree_idx = self.data_pointer + self.capacity - 1
        # 存储数据到数据数组中
        self.data[self.data_pointer] = data
        # 更新对应索引的树节点权重
        self.update(tree_idx, p)

        # 移动数据指针，循环使用存储空间
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        # 维护数据条目数
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, v):#采样数据
        # 从根节点开始向下搜索，直到找到叶子节点
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            # 如果左子节点超出范围，则当前节点为叶子节点
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                # 根据采样值确定向左还是向右子节点移动
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        # 计算叶子节点在数据数组中的索引
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total(self):
        return int(self.tree[0])

# =============================================================================

# =============================================================================
# 单独TD误差的PER_ReplayTree
# ReplayTree
class ReplayTree:#ReplayTree for the per(Prioritized Experience Replay) DQN. 
    def __init__(self, capacity):
        self.capacity = capacity # 记忆回放的容量
        self.tree = SumTree(capacity)  # 创建一个SumTree实例
        self.abs_err_upper = 1.  # 绝对误差上限
        self.epsilon = 0.01
        ## 用于计算重要性采样权重的超参数
        self.beta_increment_per_sampling = 0.001
        self.alpha = 0.6
        self.beta = 0.4 
        #self.abs_err_upper = 1.

    def __len__(self):# 返回存储的样本数量
        return self.tree.total()

    def push(self, error, sample):#Push the sample into the replay according to the importance sampling weight
        p = (np.abs(error.detach().numpy()) +self.epsilon) ** self.alpha
        self.tree.add(p, sample)         

    def sample(self, batch_size):
        pri_segment = self.tree.total() / batch_size
        priorities = []
        batch = []
        idxs = []
        is_weights = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = pri_segment * i
            b = pri_segment * (i+1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return zip(*batch), idxs, is_weights
    
    def batch_update(self, tree_idx, abs_errors):#Update the importance sampling weight
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

# =============================================================================

# =============================================================================
# 混合加权TD误差和时间衰减优先级的PER_ReplayTree
# class ReplayTree:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.tree = SumTree(capacity)
#         self.abs_err_upper = 1.
#         self.epsilon = 0.01
#         self.beta_increment_per_sampling = 0.001
#         self.alpha = 0.6
#         self.beta = 0.4
#         #时间衰减
#         self.decay_rate = 0.9  # 时间衰减因子
        
#         self.td_weight = 0.85      # TD误差权重
#         self.time_weight = 0.15    # 时间衰减权重
        
#         self.time_stamps = {}     # 记录每个样本的时间戳
#         self.current_time = 0


#     def __len__(self):
#         return self.tree.total()

#     def push(self, error, sample):
#         td_priority = (np.abs(error.detach().numpy()) + self.epsilon) ** self.alpha
#         self.time_stamps[self.current_time] = td_priority
#         self.tree.add(td_priority, sample)
#         self.current_time += 1

#     def sample(self, batch_size):      
#         pri_segment = self.tree.total() / batch_size
#         priorities = []
#         batch = []
#         idxs = []
#         self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

#         for i in range(batch_size):
#             a = pri_segment * i
#             b = pri_segment * (i+1)
#             s = random.uniform(a, b)
#             idx, p, data = self.tree.get_leaf(s)

#             priorities.append(p)
#             batch.append(data)
#             idxs.append(idx)
#         sampling_probabilities = np.array(priorities) / self.tree.total()
#         is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
#         is_weights /= is_weights.max()
#         # print('batch=', batch)
#         return zip(*batch), idxs, is_weights
    
#     def _update_priority(self, idx, error=None):
#         if error is not None:
#             # TD误差更新逻辑
#             p = (np.abs(error) + self.epsilon) ** self.alpha
#             self.time_stamps[idx] = p
#             return p
#         else:
#             # 混合优先级更新逻辑
#             age = self.current_time - idx
#             time_component = (self.decay_rate ** age)
#             td_component = self.time_stamps[idx]
#             return (self.td_weight * td_component + 
#                     self.time_weight * time_component)
        
#     def batch_update(self, tree_idx, abs_errors):
#         for ti, error in zip(tree_idx, abs_errors):
#             p = self._update_priority(ti, error)
#             self.tree.update(ti, p)

#     def update_combined_priority(self):
#         for idx in self.time_stamps:
#             p = self._update_priority(idx)
#             self.tree.update(idx, p)
#             self.time_stamps[idx] = p
            
# 1. batch_update 方法：
# - 主要用于批量更新TD误差相关的优先级
# - 只更新传入的tree_idx对应的节点
# - 更新逻辑较简单，仅基于TD误差
# 2. update_combined_priority 方法：
# - 更新所有存储样本的优先级
# - 结合了时间衰减(time_component)和TD误差(td_component)
# - 计算更复杂的混合优先级

# 主要区别：
# 1. 更新范围不同：batch_update是局部更新，update_combined_priority是全局更新
# 2. 计算逻辑不同：batch_update只考虑TD误差，update_combined_priority考虑双重因素
# 3. 调用场景不同：batch_update在训练后调用，update_combined_priority在采样前调用


    # def batch_update(self, tree_idx, abs_errors):
    #     abs_errors += self.epsilon
    #     clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
    #     ps = np.power(clipped_errors, self.alpha)
    #     for ti, p in zip(tree_idx, ps):
    #         self.tree.update(ti, p)
    #         self.time_stamps[ti] = p

    # def update_combined_priority(self):
    #     for idx in self.time_stamps:
    #         #self.current_time_t = self.current_time + 1
    #         age = self.current_time_t - idx
    #         time_component = (self.decay_rate ** age)
    #         td_component = self.time_stamps[idx]
    #         combined_priority = (self.td_weight * td_component + 
    #                             self.time_weight * time_component)
    #         self.tree.update(idx, combined_priority)
    #         self.time_stamps[idx] = combined_priority


# 时间衰减机制在ReplayTree类中的工作原理如下：
# 1. 时间戳记录 ：
# - 每次调用 push() 方法存储新经验时，会记录当前时间戳 current_time 和初始TD误差优先级
# - 时间戳作为键存储在 time_stamps 字典中，值为对应的优先级值
# 2. 时间衰减计算 ：
# - 在 update_combined_priority() 方法中，计算每个经验样本的"年龄"： age = current_time - idx
# - 时间衰减分量计算公式： time_component = (decay_rate ** age)
# - 这意味着越早的经验(age越大)，其时间衰减分量越小
# 3. 混合优先级计算 ：
# - 结合TD误差分量和时间衰减分量： combined_priority = (td_weight * td_component + time_weight * time_component)
# - 默认权重设置为TD误差占70%，时间衰减占30%
# 4. 优先级更新 ：
# - 每次采样( sample() )前会自动调用 update_combined_priority() 更新所有经验的优先级
# - 这样较新的经验会保持较高优先级，而旧经验的优先级会随时间衰减
# 5. 参数控制 ：
# - decay_rate 控制衰减速度(默认0.995，值越小衰减越快)
# - time_weight 和 td_weight 可调整两部分的影响比例
# 这种机制确保了：
# 1. 重要的TD误差经验仍会被优先采样
# 2. 新经验不会被旧经验完全淹没
# 3. 通过调整参数可以灵活控制新旧经验的平衡
# =============================================================================

# =============================================================================
#  TD-error,Time,Reward三种加权回放项
# class ReplayTree:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.tree = SumTree(capacity)
#         self.abs_err_upper = 1.
#         self.epsilon = 0.01
#         self.beta_increment_per_sampling = 0.001
#         self.alpha = 0.6
#         self.beta = 0.4
#         # 权重参数（总和应为1）
#         self.td_weight = 0.6      # TD误差权重
#         self.time_weight = 0.2    # 时间衰减权重
#         self.reward_weight = 0.2  # 奖励权重
#         # 记录系统
#         self.decay_rate = 0.9     # 时间衰减因子
#         self.time_stamps = {}     # 记录时间戳和TD误差
#         self.rewards = {}         # 记录奖励值
#         self.current_time = 0     # 全局时间计数器

#     def __len__(self):
#         return self.tree.total()

#     def push(self, error, sample):
#         s, a, r, s_ = sample  # 解包样本
#         # 计算初始TD优先级
#         td_priority = (np.abs(error.detach().numpy()) + self.epsilon) ** self.alpha
#         # 记录时间戳、TD误差和奖励
#         self.time_stamps[self.current_time] = td_priority
#         self.rewards[self.current_time] = r
#         # 计算初始混合优先级（仅TD误差）
#         self.tree.add(td_priority, sample)
#         self.current_time += 1

#     def sample(self, batch_size):
        
#         # 按优先级比例采样
#         pri_segment = self.tree.total() / batch_size
#         priorities = []
#         batch = []
#         idxs = []
#         self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

#         for i in range(batch_size):
#             a = pri_segment * i
#             b = pri_segment * (i+1)
#             s = random.uniform(a, b)
#             idx, p, data = self.tree.get_leaf(s)
#             priorities.append(p)
#             batch.append(data)
#             idxs.append(idx)

#         # 计算重要性采样权重
#         sampling_probabilities = np.array(priorities) / self.tree.total()
#         is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
#         is_weights /= is_weights.max()
#         return zip(*batch), idxs, is_weights

#     def _update_priority(self, idx, error=None):
#         if error is not None:
#             # TD误差更新逻辑
#             p = (np.abs(error) + self.epsilon) ** self.alpha
#             self.time_stamps[idx] = p
#             return p
#         else:
#             # 混合优先级更新逻辑
#             age = self.current_time - idx
#             # 1. 时间衰减分量
#             time_component = (self.decay_rate ** age)
#             # 2. TD误差分量
#             td_component = self.time_stamps[idx]
#             # 3. 奖励分量（归一化处理）
#             current_reward = self.rewards[idx]
#             avg_reward = np.mean(list(self.rewards.values()))
#             reward_diff = current_reward - avg_reward
#             max_r = max(self.rewards.values())
#             min_r = min(self.rewards.values())
#             reward_component = (reward_diff - min_r) / (max_r - min_r + 1e-5)
            
#             # 三因素混合优先级
#             return (self.td_weight * td_component +
#                     self.time_weight * time_component +
#                     self.reward_weight * reward_component)

#     def batch_update(self, tree_idx, abs_errors):
#         # 批量更新TD误差部分
#         for ti, error in zip(tree_idx, abs_errors):
#             p = self._update_priority(ti, error)
#             self.tree.update(ti, p)

#     def update_combined_priority(self):
#         # 全局更新所有样本的混合优先级
#         for idx in self.time_stamps:
#             p = self._update_priority(idx)
#             self.tree.update(idx, p)
#             self.time_stamps[idx] = p
# =============================================================================

# =============================================================================
# 混合加权TD误差和时间衰减优先级的PER_ReplayTree
# 创新：
# 1. 新增了 time_weight 和 td_weight 参数控制两部分权重
# 2. 添加了 time_steps 字典记录每个样本的时间步
# 3. 实现了新的时间衰减优先级计算： time_decay_priority = 1.0 / (step + 1)
# 4. 在 update_combined_priority() 方法中实现了混合优先级计算
# 影响：
# 1. 较新的经验样本保持较高的优先级
# 2. 较早的经验样本优先级会随时间逐渐降低
# 3. 通过权重参数可以灵活调整两部分的影响程度
# 4. 保留了优先经验回放对重要经验的学习优势

# class ReplayTree:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.tree = SumTree(capacity)
#         self.abs_err_upper = 1.
#         self.epsilon = 0.01
#         self.beta_increment_per_sampling = 0.001
#         self.alpha = 0.6
#         self.beta = 0.4
#         self.time_weight = 0.3    # 时间衰减权重
#         self.td_weight = 0.7      # TD误差权重
#         self.time_steps = {}      # 记录每个样本的时间步
#         self.current_step = 0     # 当前时间计数器


        
#     def __len__(self):
#         return self.tree.total()

#     def push(self, error, sample):
#         # 计算TD误差部分优先级
#         td_priority = (np.abs(error.detach().numpy()) + self.epsilon) ** self.alpha
#         self.time_steps[self.current_step] = td_priority
#         # 记录当前时间步
#         self.current_step += 1
#         for idx in self.time_steps:
#             step = self.current_step - idx
#         #         # 计算时间衰减优先级（step越大，优先级越低）
#             time_decay_priority = 1.0 / (step + 1) if step >= 0 else 1.0
#         # 初始优先级为纯TD误差
#             priority_final = (self.td_weight * td_priority + self.time_weight * time_decay_priority) #td_weight=1，完全TD误差。time_weight=1，完全时间衰减。
#             # self.tree.update(idx, priority_final)
#             # self.time_steps[idx] = priority_final
#         self.tree.add(priority_final, sample)       


#     # def push(self, error, sample):
#     #     # 计算TD误差部分优先级
#     #     td_priority = (np.abs(error.detach().numpy()) + self.epsilon) ** self.alpha
#     #     # 记录当前时间步
#     #     self.time_steps[self.current_step] = td_priority
#     #     # 初始优先级为纯TD误差
#     #     self.tree.add(td_priority, sample)
#     #     self.current_step += 1
#     # def update_combined_priority(self):
#     #     # 计算混合优先级
#     #     for idx in self.time_steps:
#     #         step = self.current_step - idx
#     #         # 计算时间衰减优先级（step越大，优先级越低）
#     #         time_decay_priority = 1.0 / (step + 1) if step >= 0 else 1.0
#     #         # 计算TD误差优先级
#     #         priority_td = self.time_steps[idx]
#     #         # 混合优先级（时间衰减 + TD误差）
#     #         priority_final = (self.td_weight * priority_td + self.time_weight * time_decay_priority) #td_weight=1，完全TD误差。time_weight=1，完全时间衰减。td_weight+time_weight=1。
#     #         self.tree.update(idx, priority_final)
#     #         self.time_steps[idx] = priority_final
 

#     def sample(self, batch_size):
#         #self.update_combined_priority()
#         pri_segment = self.tree.total() / batch_size
#         priorities = []
#         batch = []
#         idxs = []
#         is_weights = []
#         self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
#         for i in range(batch_size):
#             a = pri_segment * i
#             b = pri_segment * (i+1)
#             s = random.uniform(a, b)
#             idx, p, data = self.tree.get_leaf(s)
#             priorities.append(p)
#             batch.append(data)
#             idxs.append(idx)

#         sampling_probabilities = np.array(priorities) / self.tree.total()
#         is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
#         is_weights /= is_weights.max()
#         return zip(*batch), idxs, is_weights
    
#     def batch_update(self, tree_idx, abs_errors):#Update the importance sampling weight
#         abs_errors += self.epsilon
#         clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
#         ps = np.power(clipped_errors, self.alpha)

#         for ti, p in zip(tree_idx, ps):
#             self.tree.update(ti, p)

# 1. 初始化参数 ：
# - time_weight 和 td_weight 分别控制时间衰减和TD误差的权重比例
# - time_steps 字典记录每个样本的时间步信息
# - current_step 作为全局时间计数器
# 2. 存储经验(push方法) ：
# - 计算初始TD误差优先级： (np.abs(error) + epsilon) ** alpha
# - 记录当前时间步和初始优先级
# - 将经验存入SumTree中
# 3. 优先级更新(update_combined_priority方法) ：
# - 对每个存储的经验：
#   - 计算"年龄"(当前步-存储步)
#   - 时间衰减优先级： 1.0/(step + 1) （越旧的经验优先级越低）
#   - TD误差优先级：直接从字典获取
#   - 混合优先级： td_weight*TD优先级 + time_weight*时间衰减优先级
# 4. 采样(sample方法) ：
# - 先调用 update_combined_priority 更新所有优先级
# - 然后按优先级比例进行采样
# - 计算重要性采样权重(IS weights)用于抵消优先级偏差
# 5. 批量更新(batch_update方法) ：
# - 根据新的TD误差更新样本优先级
# - 但不更新时间衰减部分（这部分在下一次采样前统一更新）
# 关键创新点：

# 1. 双重考量：同时考虑TD误差（重要性）和时间衰减（新鲜度）
# 2. 可调权重：通过调整time_weight和td_weight改变两部分影响
# 3. 动态衰减：旧经验优先级随时间逐步降低（1/(step+1)）
# 4. 重要性采样：通过IS权重保证学习无偏性
# 这种机制能：

# - 保留重要经验的学习优先级
# - 防止旧经验完全淹没新经验
# - 通过参数灵活调整新旧经验平衡
# =============================================================================


# =============================================================================
# TD-error,Time,Reward三种加权回放项
# class ReplayTree:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.tree = SumTree(capacity)
#         self.abs_err_upper = 1.
#         self.epsilon = 0.01
#         self.beta_increment_per_sampling = 0.001
#         self.alpha = 0.6
#         self.beta = 0.4
#         # 调整权重参数（总和为1）
#         self.td_weight = 0.8      # TD误差权重
#         self.time_weight = 0.1    # 时间衰减权重
#         self.reward_weight = 0.1  # 新增奖励函数权重
#         self.time_steps = {}      
#         self.rewards = {}        # 新增奖励记录字典
#         self.current_step = 0     
  
#     def __len__(self):
#         return self.tree.total()

#     def push(self, error, sample):
#         s, a, r, s_ = sample  # 解包样本获取奖励值
#         td_priority = (np.abs(error.detach().numpy()) + self.epsilon) ** self.alpha
#         # 记录时间和奖励
#         self.time_steps[self.current_step] = td_priority
#         self.rewards[self.current_step] = r
#         self.current_step += 1
#         # 记录当前时间步
#         min_reward = min(self.rewards.values())
#         max_reward = max(self.rewards.values())
#         for idx in self.time_steps:
#             step = self.current_step - idx
#             # 计算时间衰减优先级
#             #time_decay_priority = 1.0 / (self.current_step - 0 + 1)  # 简化计算
#             time_decay_priority = 1.0 / (step + 1) if step >= 0 else 1.0
#             reward = self.rewards[idx]
#             reward_priority = (reward - min_reward) / (max_reward - min_reward + 1e-5)
#         # 计算奖励优先级（归一化处理）
#             #reward_priority = (r - min(self.rewards.values())) / \
#             #           (max(self.rewards.values()) - min(self.rewards.values()) + 1e-5)
        
#         # 三因素混合优先级
#             priority_final = (self.td_weight * td_priority +
#                           self.time_weight * time_decay_priority +
#                           self.reward_weight * reward_priority)
        
#         self.tree.add(priority_final, sample)


#     def sample(self, batch_size):
#         #self._update_priorities()
#         pri_segment = self.tree.total() / batch_size
#         priorities = []
#         batch = []
#         idxs = []
#         is_weights = []
#         self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        
#         for i in range(batch_size):
#             a = pri_segment * i
#             b = pri_segment * (i+1)
#             s = random.uniform(a, b)
#             idx, p, data = self.tree.get_leaf(s)
#             priorities.append(p)
#             batch.append(data)
#             idxs.append(idx)

#         sampling_probabilities = np.array(priorities) / self.tree.total()
#         is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
#         is_weights /= is_weights.max()
#         return zip(*batch), idxs, is_weights
    
#     # def _update_priorities(self):
#     #     # 更新所有存储样本的优先级
#     #     min_reward = min(self.rewards.values())
#     #     max_reward = max(self.rewards.values())
        
#     #     for idx in self.time_steps:
#     #         step = self.current_step - idx
#     #         time_decay = 1.0 / (step + 1)
#     #         reward = self.rewards[idx]
            
#     #         # 归一化奖励
#     #         reward_norm = (reward - min_reward) / (max_reward - min_reward + 1e-5)
            
#     #         new_priority = (self.td_weight * self.time_steps[idx] +
#     #                        self.time_weight * time_decay +
#     #                        self.reward_weight * reward_norm)
            
#     #         self.tree.update(idx, new_priority)
#     #         self.time_steps[idx] = new_priority

#     def batch_update(self, tree_idx, abs_errors):
#         abs_errors += self.epsilon
#         clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
#         ps = np.power(clipped_errors, self.alpha)

#         for ti, p in zip(tree_idx, ps):
#             self.tree.update(ti, p)
# =============================================================================


# 定义Net类 (定义网络)
class Net(nn.Module):
    def __init__(self):                                                         # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()                                             # 等价与nn.Module.__init__()
        self.fc1 = nn.Linear(N_STATES, 512)                                  # 设置第一个全连接层: 256个神经元到动作数个神经元
        self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.fc2 = nn.Linear(512, 512)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(512, 512)
        self.fc3.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(512, N_ACTIONS)                                   # 设置第二个全连接层(隐藏层到输出层): 256个神经元到动作数个神经元
        self.out.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        # self.BatchNorm2d = nn.BatchNorm2d(96)                                   # 若要使用GPU，必须在init初始化时定义nn网络层
        
    def forward(self, x):                                                       # 定义forward函数 (x为状态)
        x = F.relu(self.fc1(x))                                                 # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        actions_value = self.out(x)                                             # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value


# 定义DQN类 (定义两个网络)
class DQN(object):
    def __init__(self):                                                         # 定义DQN的一系列属性
        self.eval_net, self.target_net = Net(), Net()     # 利用Net创建两个神经网络: 评估网络和目标网络
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.epsilon = EPSILON_START
        # self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))            # 初始化记忆库，一行代表一个transition
        self.memory = ReplayTree(capacity=MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
        # self.BatchNorm2d = nn.BatchNorm2d(96)                                 # 若要使用GPU，必须在init初始化时定义nn网络层
        self.loss_func = nn.MSELoss()                                # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)

    def choose_action(self, x):                                                 # 定义动作选择函数 (x为状态)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)                            # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        # x = self.BatchNorm2d(x)                                                 # 此句话以便顺利使用.to('cuda:0')
        if np.random.uniform() < 1-self.epsilon:                                   # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            actions_value = self.eval_net.forward(x)                            # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[0]                                                  # 输出action的第一个数
            
        else:                                                                   # 随机选择动作
            action = np.random.randint(0, N_ACTIONS)                            # 这里action随机
        return action                                                           # 返回选择的动作

    # def store_transition(self, s, a, r, s_):                                    # 定义记忆存储函数 (这里输入为一个transition)
    #     transition = np.hstack((s, [a, r], s_))                                 # 在水平方向上拼接数组
    #     # 如果记忆库满了，便覆盖旧的数据
    #     index = self.memory_counter % MEMORY_CAPACITY                           # 获取transition要置入的行数
    #     self.memory[index, :] = transition                                      # 置入transition
    #     self.memory_counter += 1   
        
    def store_transition(self, s, a, r, s_, done):
        policy_val =self.eval_net(torch.tensor(s, dtype=torch.float))[a]
        #torch.tensor([1, 2, 3], dtype=torch.float)
        target_val =self.target_net(torch.tensor(s_, dtype=torch.float))
        transition = (s, a, r, s_)

        if done:
            error = abs(policy_val-r)
        else:
            error = abs(policy_val-r-GAMMA*torch.max(target_val))
        self.memory.push(error, transition)  # 添加经验和初始优先级
        self.memory_counter += 1        # memory_counter自加1

    def learn(self):                                                            # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络
            print('target_params_replaced')
        self.learn_step_counter += 1                                            # 学习步数自加1
        
        batch, tree_idx, is_weights = self.memory.sample(BATCH_SIZE)
        #print('1=',batch)
        b_s, b_a, b_r, b_s_ = batch
        b_s = torch.FloatTensor(b_s)
        b_a = torch.unsqueeze(torch.LongTensor(b_a), 1)
        b_r = torch.unsqueeze(torch.FloatTensor(b_r),1)
        b_s_ = torch.FloatTensor(b_s_)
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        
        
        
        
        
        loss = (torch.FloatTensor(is_weights) * self.loss_func(q_eval, q_target)).mean()    
        self.optimizer.zero_grad()
        loss.backward()
        loss_list.append(loss.item())
        self.optimizer.step()                                           # 更新评估网络的所有参数
        #np.save('loss_listRPTD.npy', loss_list)
        
        # # 抽取记忆库中的批数据
        # sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)            # 在记忆容量内随机抽取256个数，可能会重复
        # b_memory = self.memory[sample_index, :]                                 # 抽取256个索引对应的256个transition，存入b_memory
        # b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(DEVICE)
        # # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        # b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).to(DEVICE)
        # # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        # b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).to(DEVICE)
        # # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        # b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(DEVICE)
        # # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

        # # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        # q_eval = self.eval_net(b_s).gather(1, b_a)
        # # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        # q_next = self.target_net(b_s_).detach()
        # # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        # q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        # loss = self.loss_func(q_eval, q_target)
        # # 输入32个评估值和32个目标值，使用均方损失函数
        # self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        # loss.backward()                                                 # 误差反向传播, 计算参数更新值
        # loss_list.append(loss.item())
        # self.optimizer.step()                                           # 更新评估网络的所有参数
        # np.save('loss_list.npy', loss_list)
        abs_errors = torch.abs(q_eval - q_target).detach().numpy().squeeze()
        self.memory.batch_update(tree_idx, abs_errors)  # 更新经验的优先级
        # 更新 epsilon
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
            np.exp(-1.0 * self.learn_step_counter / EPSILON_DECAY) if EPSILON_END + (EPSILON_START - EPSILON_END) * \
            np.exp(-1.0 * self.learn_step_counter / EPSILON_DECAY) > EPSILON_END else EPSILON_END
        self.learn_step_counter += 1
