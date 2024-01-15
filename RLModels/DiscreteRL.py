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

    This class provides methods for initializing the model, training the model, and saving the trained policy model.

    Attributes:
    - actor_lr (float): Learning rate for the actor network.
    - critic_lr (float): Learning rate for the critic network.
    - seed (int): Seed value for reproducibility.
    - use_gpu (bool): Flag indicating whether to use GPU for training.
    - model (object): The instance of the DSC class used for reinforcement learning.

    Methods:
    - __init__(self, actor_lr=0.0003, critic_lr=0.0003, seed=42, use_gpu=True): Initializes the DiscreteModel class.
    - initialize_model(self): Initializes the model for reinforcement learning.
    - train(self, dataset, n_epochs=256): Trains the model using the specified number of epochs.
    - save_model(self, save_path): Saves the trained policy model at the specified save path.
    """

    def __init__(self, actor_lr=0.0003, critic_lr=0.0003, seed=42, use_gpu=True):
        """
        Initializes the DiscreteModel class.

        Parameters:
        - actor_lr (float): Learning rate for the actor network. Default is 0.0003.
        - critic_lr (float): Learning rate for the critic network. Default is 0.0003.
        - seed (int): Seed value for reproducibility. Default is 42.
        - use_gpu (bool): Flag indicating whether to use GPU for training. Default is True.
        """

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.seed = seed
        self.use_gpu = use_gpu
        self.model = None


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

    def train(self, dataset, n_epochs=256):
        """
        Trains the model using the specified number of epochs.

        Parameters:
        - dataset (object): The dataset used for training.
        - n_epochs (int): The number of epochs to train the model (default: 256).
        """

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
