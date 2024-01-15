import os
import d3rlpy
import gym
from d3rlpy.datasets import MDPDataset
import numpy as np
import pickle


def save_batch(dataset_path, observations, actions, rewards, terminals):
    """
    Save a batch of data to the dataset file.

    Args:
        observations (list): List of observations.
        actions (list): List of actions.
        rewards (list): List of rewards.
        terminals (list): List of terminal flags.

    Returns:
        None
    """
    # Data structure for the batch
    batch_data = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'terminals': terminals
    }

    # Check if the dataset file already exists
    if os.path.exists(dataset_path):
        with open(dataset_path, 'rb') as f:
            existing_data = pickle.load(f)
        # Append new batch data to the existing data
        for key in batch_data:
            existing_data[key].extend(batch_data[key])
        with open(dataset_path, 'wb') as f:
            pickle.dump(existing_data, f)
    else:
        # Create a new dataset file with the current batch
        with open(dataset_path, 'wb') as f:
            pickle.dump(batch_data, f)


def collect_data(env, dataset_path, num_episodes=50, batch_size=25):
    """
    Collects data by running episodes in the environment.

    Args:
        num_episodes (int): The number of episodes to run.
        batch_size (int): The size of each batch to save.

    Returns:
        None
    """
    # Prepare a temporary storage for batch data
    observations, actions, rewards, terminals = [], [], [], []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminals.append(done)

            obs = next_obs

        # Save batch when it reaches batch_size or at the end of data collection
        if (episode + 1) % batch_size == 0 or episode == num_episodes - 1:
            save_batch(dataset_path, observations, actions, rewards, terminals)
            # Clear lists for the next batch
            observations, actions, rewards, terminals = [], [], [], []


def load_dataset(env_name='Seaquest-v4', dataset_path='data/SeaQuestdataset.pkl', seed=42):
    """
    Loads the dataset from a file or collects new data if the file doesn't exist.

    Returns:
        dataset (MDPDataset): The loaded dataset.
    """
    env = gym.make(env_name)
    env.seed(seed)
    # Check if dataset file exists
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        dataset = MDPDataset(
            observations=np.array(data['observations']), 
            actions=np.array(data['actions']), 
            rewards=np.array(data['rewards']), 
            terminals=np.array(data['terminals']), 
            discrete_action=True
        )
    else:
        print("Collecting new data...")
        collect_data(env, dataset_path)  # This will save batches incrementally
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        dataset = MDPDataset(
            observations=np.array(data['observations']), 
            actions=np.array(data['actions']), 
            rewards=np.array(data['rewards']), 
            terminals=np.array(data['terminals']), 
            discrete_action=True
        )
    print(f"Dataset size: {len(dataset)}")
    return dataset
