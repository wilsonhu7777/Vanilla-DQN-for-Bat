import random
import keras
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

class BESSAgent:
    def __init__(self, state_count, action_count, iteration):
        self.state_count = state_count
        self.action_count = action_count
        self.iteration = iteration
        self.epsilon = 0.05
        self.q_net = self._build_dqn_model()
        self.target_q_net = self._build_dqn_model()

    def _build_dqn_model(self):
        q_net = Sequential()
        q_net.add(Dense(64, input_dim=self.state_count, activation='relu', kernel_initializer='he_uniform'))  # initializer sets the initial weights of the layer
        q_net.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(Dense(self.action_count, activation='linear', kernel_initializer='he_uniform'))
        q_net.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss='mean_squared_error', metrics=['MeanSquaredError'])
        return q_net

    def random_action(self, current_obs):
        """
        returns a random action
        :param current_obs: current observation of the environment
        :return: action corresponding to the observation
        """
        action = random.randint(0,2)
        return action

    def collect_action(self, current_obs):
        """
        trade-off between explore and exploit of action space controllable by epsilon
        :param state: observation of the environment
        :return: action
        """
        if random.random() < self.epsilon:
            return self.random_action(current_obs)
        return self.action(current_obs)

    def action(self,state):
        """
        takes a state from the env and returns an action that has the highest Q value and
        should be taken next step.
        :param state: the current env state
        :return: an action (the index)
        """
        state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        action_q = self.q_net(state_input)
        action = np.argmax(action_q.numpy(), axis=1)
        return action[0]

    def update_target_network(self):
        """
        updates the current dqn network with the q_net
        :return: None
        """
        self.target_q_net.set_weights(self.q_net.get_weights())

    def train(self, batch):
        current_obs_batch, next_obs_batch, action_batch, reward_batch, done_batch = batch
        current_q = self.q_net(current_obs_batch).numpy()
        target_q = np.copy(current_q)
        next_q = self.target_q_net(next_obs_batch).numpy()
        max_next_q = np.amax(next_q, axis=1)
        for i in range(current_obs_batch.shape[0]):
            target_q_val = reward_batch[i]
            if not done_batch[i]:
                target_q_val += 0.95 * max_next_q[i]
            target_q[i][action_batch[i]] = target_q_val
        training_history = self.q_net.fit(x=current_obs_batch, y=target_q, verbose=0)
        loss = training_history.history['loss']
        return loss
