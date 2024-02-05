import torch
import pickle
import numpy as np
import random
from mk_patch_decision_transformer_atari import GPTConfig, GPT
from mk_patch_trajectory_transformer import GPT as TrajectoryGPT

import sys
sys.path.insert(0, 'Seaquest_HalfCheetah/trajectory_transformer')

def set_seed(seed):
    """Set all seeds to make results reproducible (deterministic mode).

    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_seaquest_model(checkpoint_path="decision_transformer_atari\checkpoints\Seaquest_123.pth", seed=0):
    """
    Load the Seaquest model from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file. Default is "decision_transformer_atari\checkpoints\Seaquest_123.pth".

    Returns:
        model (GPT): Loaded Seaquest model.
    """
    set_seed(seed)

    vocab_size = 18 # 18 for Seaquest
    block_size = 90
    model_type = "reward_conditioned"
    timesteps = 2719 # 2719 for Seaquest

    mconf = GPTConfig(
        vocab_size,
        block_size,
        n_layer=6,
        n_head=8,
        n_embd=128,
        model_type=model_type,
        max_timestep=timesteps,
    )
    model = GPT(mconf)

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    return model

def load_halfcheetah_model(model_path, seed=0):
    """
    Loads a pre-trained transformer model from the given path.

    Args:
        model_path (str): The path to the directory containing the model files.

    Returns:
        The loaded transformer model.
    """
    set_seed(seed)
    # Load the model configurations
    model_config_path = model_path + '/model_config.pkl'
    print(model_config_path)
    with open(model_config_path, 'rb') as config_file:
        model_config = pickle.load(config_file)
    # Initialize the model from the loaded configuration
    model = TrajectoryGPT(model_config)

    # Load the pre-trained weights
    # Choose the specific state file you want to load, e.g., state_0.pt, state_16.pt, etc.
    state_dict_path = model_path + '/state_48.pt'
    model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))

    # If you're using a GPU, you can move the model to GPU
    # model = model.to('cuda')
    return model