"""
This script performs training and analysis on the Seaquest and HalfCheetah datasets.
It loads the datasets, creates trajectories, obtains trajectory embeddings, performs clustering,
trains models using different algorithms, and computes attributions for the test observations.
The results are printed in tables.
"""
import d3rlpy
import torch
import numpy as np
from matplotlib import pyplot as plt
from makeSeaquestdata import load_seaquest_dataset, make_seaquest_testset
from load_model import load_seaquest_model
from model import make_episodes, fit_discrete_sac, fit_per_cluster
from utils import create_trajectories, get_trajectory_embedding, perform_clustering_and_plot, trajectory_attributions_sq, set_seeds, print_results_sq
from encoder import CustomCNNFactory

if __name__ == "__main__":
    # Set device and random seed
    device = set_seeds(seed=0)

    # Set data and test sizes
    data_size, test_size = 617, 100

    # Load Seaquest dataset
    seaquestdata, sq_env = load_seaquest_dataset(env_name='Breakout', size=data_size+test_size)

    # Load pre-trained encoders
    pre_trained_encoder_seaquest = load_seaquest_model("decision_transformer_atari/checkpoints/Breakout_123.pth", seed=0)

    # Create Seaquest test set
    seaquestdata, test_observation_sq = make_seaquest_testset(seaquestdata, data_size, test_size)

    # Create trajectories and obtain embeddings for Seaquest data
    final_obs_sq, final_act_sq, final_rew_sq, _ = create_trajectories(
        seaquestdata["observations"], seaquestdata["actions"], 
        seaquestdata["rewards"], seaquestdata["terminals"], trajectory_length=30
    )
    trajectory_embedding_seaquest = get_trajectory_embedding(
        pre_trained_encoder_seaquest, final_obs_sq, final_act_sq, final_rew_sq, 
        is_seaquest=True, device=device
    )

    # Perform clustering on Seaquest data
    clusters_seaquest, _ = perform_clustering_and_plot(trajectory_embedding_seaquest.detach().cpu().numpy(), 2, 8, ccore=True, plot=False)

    # Train Seaquest model with DiscreteSAC
    print('#' * 100)
    print("Training Seaquest with DiscreteSAC...")
    list_episodes_sq = make_episodes(final_obs_sq, final_act_sq, final_rew_sq, 4)
    sac_sq = fit_discrete_sac(list_episodes_sq, n_steps=10, n_steps_per_epoch=10, device=device)


    # Compute all possible permutations of clusters
    print('#' * 100)
    print("Compute all possible permutations of clusters")
    feature_size = 128
    custom_encoder_actor = CustomCNNFactory(feature_size)
    custom_encoder_critic = CustomCNNFactory(feature_size)
    
    model_params_sq = {
        'actor_learning_rate': 3e-4,
        'critic_learning_rate': 3e-4,
        'temp_learning_rate': 3e-4,
        'batch_size': 256,
        'actor_encoder_factory': custom_encoder_actor,
        'critic_encoder_factory': custom_encoder_critic,
        'scaler': 'pixel',
        'use_gpu': True if device == 'cuda' else False
    }
    
    # Fit models per cluster for Seaquest data
    models_sq, result_data_combinations_sq = fit_per_cluster(
        model=sac_sq, 
        model_class=d3rlpy.algos.DiscreteSAC,
        model_params=model_params_sq,
        data_embedding=trajectory_embedding_seaquest,
        list_episodes=list_episodes_sq,
        clusters=clusters_seaquest,
        trajectory_embedding=trajectory_embedding_seaquest,
        test_observations=test_observation_sq
    )
    
    # Compute attributions for Seaquest data
    print('#' * 100)
    print("Attributions Seaquest")
    action_dict = {	
        0: 'NOOP',
        1: 'FIRE',
        2: 'UP',
        3: 'RIGHT',
        4: 'LEFT',
        5: 'DOWN',
        6: 'UPRIGHT',
        7: 'UPLEFT',
        8: 'DOWNRIGHT',
        9: 'DOWNLEFT',
        10: 'UPFIRE',
        11: 'RIGHTFIRE',
        12: 'LEFTFIRE',
        13: 'DOWNFIRE',
        14: 'UPRIGHTFIRE',
        15: 'UPLEFTFIRE',
        16: 'DOWNRIGHTFIRE',
        17: 'DOWNLEFTFIRE'
    }
    attributions_sq = trajectory_attributions_sq(test_observation_sq, models_sq, trajectory_embedding_seaquest, clusters_seaquest)
    
    # Print table 2 for Seaquest data
    print('#' * 100)
    print("Print table 2")
    print_results_sq(result_data_combinations_sq, test_observation_sq, models_sq, attributions_sq)
