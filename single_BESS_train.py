"""
Training loop
This module trains the DQN agent by trial and error. In this module the DQN
agent will play the game episode by episode, store the gameplay experiences
and then use the saved gameplay experiences to train the underlying model.
"""
from single_BESS_agent import BESSAgent
from single_BESS_replay import ReplayBuffer
from single_BESS_env import BESSEnv
import pickle


def evaluate_training_result(env, agent, num_episode):
    """
    Evaluates the performance of the current DQN agent by using it to play a
    few episodes of the game and then calculates the average reward it gets.
    The higher the average reward is the better the DQN agent performs.
    :param num_episode: number of episodes
    :param env: the game environment
    :param agent: the DQN agent
    :return: average reward across episodes
    """
    total_reward = 0
    episodes_to_play = num_episode
    for i in range(episodes_to_play):
        current_obs = env.reset(day=i)
        finish = False
        episode_reward = 0
        while not finish:
            action = agent.action(current_obs)
            current_obs, next_obs, reward, finish = env.step(action)
            episode_reward += reward
            current_obs = next_obs
        total_reward += episode_reward
    average_reward = total_reward / episodes_to_play
    return average_reward


def collect_gameplay_experiences(env, agent, buffer, num_episodes=60):
    """
    Collects gameplay experiences by playing env with the instructions
    produced by agent and stores the gameplay experiences in buffer.
    :param env: the game environment
    :param agent: the DQN agent
    :param buffer: the replay buffer
    :return: None
    """
    for xday in range(num_episodes):
        current_obs = env.reset(xday)
        finish = False
        while not finish:
            action = agent.collect_action(current_obs)
            current_obs, next_obs, reward, finish = env.step(action)
            buffer.store_gameplay_experience(current_obs, next_obs, action, reward, finish)
            current_obs = next_obs


def train_model(max_episodes=60):
    """
    Trains a DQN agent to play the CartPole game by trial and error
    :return: None
    """
    agent = BESSAgent(state_count=25, action_count=3, iteration=24)
    buffer = ReplayBuffer()
    env = BESSEnv()
    collect_gameplay_experiences(env, agent, buffer)
    for episode_cnt in range(max_episodes):
        collect_gameplay_experiences(env, agent, buffer)
        gameplay_experience_batch = buffer.sample_gameplay_batch()
        loss = agent.train(gameplay_experience_batch)
        avg_reward = evaluate_training_result(env, agent, max_episodes)
        print(f'Episode {episode_cnt+1}/{max_episodes} and so far the performance is {avg_reward} and loss is {loss[0]}')
        if (episode_cnt+1) % 15 == 0:
            agent.update_target_network()
        with open("REWARDS_DQN.pkl", 'wb') as f:
            pickle.dump(avg_reward, f, pickle.HIGHEST_PROTOCOL)
    print('running smooth, no bugs!')


if __name__ == '__main__':
    train_model(max_episodes=75)
