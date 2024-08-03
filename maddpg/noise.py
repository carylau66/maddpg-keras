# OUActionNoise类用于生成Ornstein-Uhlenbeck噪声，这通常用于连续动作空间的探索。
# 如果动作空间的范围或类型发生变化，你可能需要调整噪声的生成方式，以确保其仍然适合新的动作范围。
# OU过程是一种时间相关的噪声过程，适用于模拟有惯性的系统。它有助于生成平滑的、具有时间相关性的噪声序列。
        # 尝试高斯噪声=======
import numpy as np

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=0.4, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self): # __call__ 是一个特殊方法（或称为魔术方法），允许对象的实例像函数一样被调用
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
            