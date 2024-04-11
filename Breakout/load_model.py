import torch
import pickle
import numpy as np
import random
from mk_patch_decision_transformer_atari import GPTConfig, GPT

# import sys
# sys.path.insert(0, 'Breakout/decision_transformer_atari')

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

def load_seaquest_model(checkpoint_path="decision_transformer_atari\checkpoints\Seaquest_123.pth", vocab_size= 4, block_size=90, timesteps=2654, seed=0):
    """
    Load the Seaquest model from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file. Default is "decision_transformer_atari\checkpoints\Seaquest_123.pth".

    Returns:
        model (GPT): Loaded Seaquest model.
    """
    set_seed(seed)

    vocab_size = vocab_size
    block_size = block_size
    model_type = "reward_conditioned"
    timesteps = timesteps

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