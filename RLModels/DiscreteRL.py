import os
import d3rlpy
from d3rlpy.algos import DiscreteSAC as DSC
import gym
from d3rlpy.datasets import MDPDataset
import numpy as np
import pickle

class DiscreteModel:
    """
    A class representing a discrete reinforcement learning model.

    This class provides methods for collecting data, saving data to a dataset file,
    loading the dataset, initializing the model, training the model, and saving the trained policy model.

    Attributes:
    - env_name (str): Name of the environment.
    - actor_lr (float): Learning rate for the actor network.
    - critic_lr (float): Learning rate for the critic network.
    - seed (int): Seed value for reproducibility.
    - use_gpu (bool): Flag indicating whether to use GPU for training.
    - model: The reinforcement learning model.
    - env: The environment.
    - dataset_path (str): Path to the dataset file.

    Methods:
    - __init__(self, env_name, actor_lr=0.0003, critic_lr=0.0003, seed=42, use_gpu=True, dataset_path='SeaQuestdataset.pkl'): Initializes the DiscreteModel class.
    - collect_data(self, num_episodes=50, batch_size=25): Collects data by running episodes in the environment.
    - save_batch(self, observations, actions, rewards, terminals): Save a batch of data to the dataset file.
    - load_dataset(self): Loads the dataset from a file or collects new data if the file doesn't exist.
    - initialize_model(self): Initializes the model for reinforcement learning.
    - train(self, n_epochs=256): Trains the model using the specified number of epochs.
    - save_model(self, save_path): Saves the trained policy model at the specified save path.
    """
    
    def __init__(self, env_name, actor_lr=0.0003, critic_lr=0.0003, seed=42, use_gpu=True, dataset_path='SeaQuestdataset.pkl'):
        """
        Initializes the DiscreteModel class.

        Parameters:
        - env_name (str): Name of the environment.
        - actor_lr (float): Learning rate for the actor network. Default is 0.0003.
        - critic_lr (float): Learning rate for the critic network. Default is 0.0003.
        - seed (int): Seed value for reproducibility. Default is 42.
        - use_gpu (bool): Flag indicating whether to use GPU for training. Default is True.
        - dataset_path (str): Path to the dataset file. Default is 'SeaQuestdataset.pkl'.
        """
        self.env_name = env_name
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.seed = seed
        self.use_gpu = use_gpu
        self.model = None
        self.env = gym.make(self.env_name)
        self.env.seed(self.seed)
        self.dataset_path = dataset_path

    def collect_data(self, num_episodes=50, batch_size=25):
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
            obs = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_obs, reward, done, _ = self.env.step(action)

                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                terminals.append(done)

                obs = next_obs

            # Save batch when it reaches batch_size or at the end of data collection
            if (episode + 1) % batch_size == 0 or episode == num_episodes - 1:
                self.save_batch(observations, actions, rewards, terminals)
                # Clear lists for the next batch
                observations, actions, rewards, terminals = [], [], [], []

    def save_batch(self, observations, actions, rewards, terminals):
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
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, 'rb') as f:
                existing_data = pickle.load(f)
            # Append new batch data to the existing data
            for key in batch_data:
                existing_data[key].extend(batch_data[key])
            with open(self.dataset_path, 'wb') as f:
                pickle.dump(existing_data, f)
        else:
            # Create a new dataset file with the current batch
            with open(self.dataset_path, 'wb') as f:
                pickle.dump(batch_data, f)


    def load_dataset(self):
        """
        Loads the dataset from a file or collects new data if the file doesn't exist.

        Returns:
            dataset (MDPDataset): The loaded dataset.
        """
        # Check if dataset file exists
        if os.path.exists(self.dataset_path):
            print(f"Loading dataset from {self.dataset_path}")
            with open(self.dataset_path, 'rb') as f:
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
            self.collect_data()  # This will save batches incrementally
            with open(self.dataset_path, 'rb') as f:
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


    def initialize_model(self):
        """
        Initializes the model for reinforcement learning.

        This method creates an instance of the DSC class with the specified actor and critic learning rates,
        and sets the use_gpu flag based on the value provided.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None
        """
        self.model = DSC(actor_learning_rate=self.actor_lr, critic_learning_rate=self.critic_lr, use_gpu=self.use_gpu)

    def train(self, n_epochs=256):
        """
        Trains the model using the specified number of epochs.

        Parameters:
        - n_epochs (int): The number of epochs to train the model (default: 256).
        """
        # Load the dataset
        dataset = self.load_dataset()
        
        # Initialize the model
        self.initialize_model()

        # Start training
        self.model.fit(
            dataset,
            n_epochs=n_epochs,
            show_progress=True
        )

        # Save the trained model
        self.save_model('trained_policy.pth')

    def save_model(self, save_path):
        """
        Saves the trained policy model at the specified save path.

        Args:
            save_path (str): The path where the trained policy model will be saved.

        Returns:
            None
        """
        self.model.save_policy(save_path)
        print(f"Trained policy saved at: {save_path}")
