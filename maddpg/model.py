# 在get_actor()函数中，你可以看到定义的输入层大小匹配了状态空间的维度，而输出层则根据动作空间进行了设计。
# get_critic()函数定义了Critic网络，它接收状态和动作作为输入。如果状态或动作空间发生变化，你需要修改输入层的大小，以确保网络可以处理更新后的信息。
import tensorflow as tf
from tensorflow.keras import layers

from env.env import DIM_AGENT_STATE, NUM_AGENTS

def get_actor():

    """
    Creates actor model

    Returns
    -------
    model: tf.keras.Model
        keras actor model
    """
    # Initialize weights between -3e-5 and 3-e5
    last_init = tf.random_uniform_initializer(minval=-0.00003, maxval=0.00003)

    # Actor will get observation of the agent
    # not the observation of other agents
    inputs = layers.Input(shape=(DIM_AGENT_STATE,))
    out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(inputs)
    out = layers.Dropout(rate=0.5)(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(out)
    out = layers.Dropout(rate=0.5)(out)
    out = layers.BatchNormalization()(out)
    
    # Using tanh activation as action values for
    # for our environment lies between -1 to +1
    # outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out) ---------------
    outputs = layers.Dense(3, activation="sigmoid", kernel_initializer=last_init)(out)
    
    outputs = outputs 
    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic(dim_state):

    """
    Creates and returns critic model

    Parameters
    ----------
    dim_state: int
        sum of dimension of state of each agents
    
    Returns
    -------
    model: tf.keras.Model
        keras critic model
    """
    
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    
    # State as input, here this state is
    # observation of all the agents
    # hence this state will have information
    # of observation of all the agents
    state_input = layers.Input(shape=(dim_state))
    state_out = layers.Dense(16, activation="selu", kernel_initializer="lecun_normal")(state_input)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Dense(32, activation="selu", kernel_initializer="lecun_normal")(state_out)
    state_out = layers.BatchNormalization()(state_out)

    # 
    agents_action_input = [layers.Input(shape=(1)) for i in range(NUM_AGENTS)]
    action_input = layers.Concatenate()(agents_action_input)
    
    action_out = layers.Dense(32, activation="selu", kernel_initializer="lecun_normal")(action_input)
    action_out = layers.BatchNormalization()(action_out)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(512, activation="selu", kernel_initializer="lecun_normal")(concat)
    out = layers.Dropout(rate=0.5)(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(512, activation="selu", kernel_initializer="lecun_normal")(out)
    out = layers.Dropout(rate=0.5)(out)
    out = layers.BatchNormalization()(out)
    
    outputs = layers.Dense(1)(out)
    
    # Outputs single value for give state-action
    model = tf.keras.Model([state_input]+agents_action_input, outputs)

    return model