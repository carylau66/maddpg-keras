# 专注于预测阶段，即智能体应用所学策略进行决策的过程。预测环境通常不包含训练时的某些复杂性，如随机性和噪声，以确保智能体的决策过程尽可能地稳定和可预测。

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import math
import random

#Dimension of State Space for single agent
dim_agent_state = 5

#Number of Episodes
num_episodes = 3000

#Number of Steps

num_steps = 400

import math
import random

import numpy as np

from config import NUM_STEPS, NUM_EPISODES

#Dimension of State Space for single agent ------
# DIM_AGENT_STATE = 5 
DIM_AGENT_STATE_E = 2
DIM_AGENT_STATE_G = 3

# Number of agents 
NUM_AGENTS = 3
NUM_AGENTS_E = 2
NUM_AGENTS_G = 1

#Constant 
a = 20

#Number of Episodes
num_episodes = NUM_EPISODES

#Number of Steps
num_steps = NUM_STEPS

#Dimension of State Space
dim_state = DIM_AGENT_STATE_E*NUM_AGENTS_E + DIM_AGENT_STATE_G*NUM_AGENTS_G


class ENVIRONMENT:
  def __init__(self):
    self.e1_1 = random.uniform(0.0, 1.0 + 1e-10) * 100 # 报价P，0-100, 因为生成的随机数不含上界 为了可以取到1
    self.e1_2 = random.uniform(0.0, 1.0 + 1e-10) + 1 # 质量Q
    # self.e1_3 = random.uniform(0.0, 1.0 + 1e-10)
    self.e2_1 = random.uniform(0.0, 1.0 + 1e-10) * 100
    self.e2_2 = random.uniform(0.0, 1.0 + 1e-10) + 1
    # self.e2_3 = random.uniform(0.0, 1.0 + 1e-10)

    self.g_1 = random.uniform(0.0, 1.0 + 1e-10) # 对1的采购比例
    self.g_2 = random.uniform(0.0, 1.0 + 1e-10) * 0.25 # 激励1
    self.g_3 = random.uniform(0.0, 1.0 + 1e-10) * 0.25 # 激励2
    # self.g_3 = random.uniform(0.0, 1.0 + 1e-10)

  
  def initial_obs(self):
    # 所有智能体的状态列出来
    obs = [self.e1_1, self.e1_2,
           self.e2_1, self.e2_2, 
           self.g_1, self.g_2, self.g_3]
    return obs

  def state_step(self, actions):
    self.e1_1 = actions[0][0] *100
    self.e1_2 = actions[0][1] + 1
    self.e2_1 = actions[1][0] *100
    self.e2_2 = actions[1][1] + 1

    self.g_1 = actions[3][0]
    self.g_2 = actions[3][1] * 0.25
    self.g_3 = actions[3][2] * 0.25


    state_e1 = [self.e1_1, self.e1_2]
    state_e2 = [self.e2_1, self.e2_2]
    state_g1 = [self.g_1, self.g_2, self.g_3]

    return state_e1, state_e2

  def step(self, actions):
    # state_e1, state_e2 = self.state_step(actions)
    self.e1_1 = actions[0][0] *100  # 0 - 100
    self.e1_2 = actions[0][1] + 1   # 1 - 2
    self.e2_1 = actions[1][0] *100
    self.e2_2 = actions[1][1] + 1

    self.g_1 = actions[3][0]        # 0 - 1
    self.g_2 = actions[3][1] * 0.25 # 0-0.25
    self.g_3 = actions[3][2] * 0.25 # 0-0.25

    obs = [self.e1_1, self.e1_2,
           self.e2_1, self.e2_2, 
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
    return L/(1 + math.exp(-k(q-q0)))

#Reward Calculator
def reward(state):
  rewards = []
  for i in range(NUM_AGENTS):
    # 企业1
    if i == 0:
      r = (0.75 + state[5]) * state[0] - ceq(state[1]) * state[4]
      rewards.append(r)
    elif i == 1:
      r = (0.75 + state[6]) * state[2] - ceq(state[3]) * (1 - state[4])
      rewards.append(r)
    else:
      r1 = a * uq(state[1]) - (0.75 + state[5]) * state[0]
      r2 = a * uq(state[3]) - (0.75 + state[6]) * state[2]
      r = r1**2/(r1+r2) + r2**2/(r1+r2)
      rewards.append(r)

  return rewards


