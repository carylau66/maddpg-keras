# 定义环境的状态空间、动作空间、奖励函数以及智能体之间的交互规则
import math
import random

import numpy as np

from config import NUM_STEPS, NUM_EPISODES

#Dimension of State Space for single agent ------
DIM_AGENT_STATE = 3 
DIM_AGENT_STATE_E = 3 #为了少改代码，保持一致
DIM_AGENT_STATE_G = 3

# Number of agents 
NUM_AGENTS = 3
NUM_AGENTS_E = 2
NUM_AGENTS_G = 1
# 动作维度
DIM_ACTION = 3
#Dimension of State Space
dim_state = DIM_AGENT_STATE * NUM_AGENTS

#Constant 
a = 20

#Number of Episodes
num_episodes = NUM_EPISODES

#Number of Steps
num_steps = NUM_STEPS




class ENVIRONMENT:
  def __init__(self):
    self.e1_1 = random.uniform(0.0, 1.0 + 1e-10) * 100 # 报价P，0-100, 因为生成的随机数不含上界 为了可以取到1
    self.e1_2 = random.uniform(0.0, 1.0 + 1e-10) + 1 # 质量Q
    self.e1_3 = random.uniform(0.0, 1.0 + 1e-10)     # 采购比例

    self.e2_1 = random.uniform(0.0, 1.0 + 1e-10) * 100
    self.e2_2 = random.uniform(0.0, 1.0 + 1e-10) + 1
    self.e2_3 = 1 - self.e1_3

    self.g_1 = random.uniform(0.0, 1.0 + 1e-10) # 对1的采购比例
    self.g_2 = random.uniform(0.0, 1.0 + 1e-10) * 0.25 # 激励1
    self.g_3 = random.uniform(0.0, 1.0 + 1e-10) * 0.25 # 激励2
    # self.g_3 = random.uniform(0.0, 1.0 + 1e-10)

  
  def initial_obs(self):
    # 所有智能体的状态列出来
    obs = [self.e1_1, self.e1_2, self.e1_3,
           self.e2_1, self.e2_2, self.e2_3,
           self.g_1, self.g_2, self.g_3]
    return obs

  # def state_step(self, actions):
  #   self.e1_1 = actions[0][0] *100
  #   self.e1_2 = actions[0][1] + 1
  #   self.e1_3 = actions[2][0]

  #   self.e2_1 = actions[1][0] *100
  #   self.e2_2 = actions[1][1] + 1
  #   self.e2_3 = 1 - self.e1_3

  #   self.g_1 = actions[2][0]
  #   self.g_2 = actions[2][1] * 0.25
  #   self.g_3 = actions[2][2] * 0.25


  #   state_e1 = [self.e1_1, self.e1_2, self.e1_3]
  #   state_e2 = [self.e2_1, self.e2_2, self.e2_3]
  #   state_g1 = [self.g_1, self.g_2, self.g_3]

  #   return state_e1, state_e2

  def step(self, actions):
    # state_e1, state_e2 = self.state_step(actions)
    self.e1_1 = actions[0] *100  # 0 - 100
    self.e1_2 = actions[1] + 1   # 1 - 2
    self.e1_3 = actions[2]      # 0 - 1

    self.e2_1 = actions[3] *100
    self.e2_2 = actions[4] + 1
    self.e2_3 = 1 - self.e1_3       # 0 - 1

    self.g_1 = actions[6]        # 0 - 1
    self.g_2 = actions[7] * 0.25 # 0-0.25
    self.g_3 = actions[8] * 0.25 # 0-0.25

    obs = [self.e1_1, self.e1_2, self.e1_3,
           self.e2_1, self.e2_2, self.e2_3,
           self.g_1, self.g_2, self.g_3]

    return obs
  
#质量函数
def ceq(q):
   return 5*q**2+5*q+2   
#Function for generating sigmoid output of Input Function
def sigmoid(x):
    val = 1/(1+np.exp(-x))
    return val
def uq(q):
    q0 = 1
    L = 2
    k = 5
    return L/(1 + math.exp(-k*(q-q0)))

#Reward Calculator
def reward(state):
  rewards = []
  for i in range(NUM_AGENTS):
    # 企业1
    if i == 2:
      r1 = a * uq(state[1]) - (0.75 + state[5]) * state[0]
      r2 = a * uq(state[3]) - (0.75 + state[6]) * state[2]
      r = (r1**2)/(r1+r2) + (r2**2)/(r1+r2)
      rewards.append(r)
    else:
      r = ((0.75 + state[2+i*3]) * state[i*3] - ceq(state[1+i*3])) * state[2+i*3]
      rewards.append(r)

  return rewards


