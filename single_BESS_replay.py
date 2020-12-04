import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    replay bufferstores and retrieves gameplay experiences
    """

    def __init__(self):
        self.gameplay_experiences = deque(maxlen=1000000)

    def store_gameplay_experience(self, current_obs, next_obs, action, reward, done):
        """
        records a single step (state transition) of gameplay experiences.
        :param state: the current game state
        :param next_state: the game state after taking action
        :param reward: the reward taking action at the current state brings
        :param action: the action taken at the current state
        :param done: a boolean indicating if the game is finished after taking the action
        :return: None
        """
        self.gameplay_experiences.append((current_obs, next_obs, action, reward, done))

    def sample_gameplay_batch(self):
        """
        samples a batch of gameplay experiences for training
        :return: a list of game experiences
        """
        batch_size = min(24*7, len(self.gameplay_experiences))
        sampled_gameplay_batch = random.sample(self.gameplay_experiences, batch_size)
        current_obs_batch = []
        next_obs_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        for gameplay_experience in sampled_gameplay_batch:
            current_obs_batch.append(gameplay_experience[0])
            next_obs_batch.append(gameplay_experience[1])
            action_batch.append(gameplay_experience[2])
            reward_batch.append(gameplay_experience[3])
            done_batch.append(gameplay_experience[4])
        return np.array(current_obs_batch), np.array(next_obs_batch), np.array(action_batch), np.array(reward_batch), np.array(done_batch)
