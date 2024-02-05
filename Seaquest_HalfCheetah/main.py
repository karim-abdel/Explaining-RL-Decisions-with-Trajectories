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
from data.makeSeaquestdata import load_seaquest_dataset, make_seaquest_testset
from data.makeHalfcheetahdata import load_halfcheetah_dataset, make_halfcheetah_testset
from load_model import load_seaquest_model, load_halfcheetah_model
from model import make_episodes, fit_discrete_sac, fit_sac, fit_per_cluster
from utils import create_trajectories, get_trajectory_embedding, perform_clustering_and_plot, trajectory_attributions_hc, trajectory_attributions_sq, set_seeds, print_results_hc, print_results_sq
from model import make_episodes, fit_discrete_sac, fit_per_cluster, fit_sac
from encoder import CustomCNNFactory

if __name__ == "__main__":
    # Set device and random seed
    device = set_seeds(seed=0)

    # Set data and test sizes
    data_size, test_size = 25, 2

    # Load Seaquest dataset
    seaquestdata, sq_env = load_seaquest_dataset(env_name='Seaquest', size=data_size+test_size)

    # Load HalfCheetah dataset
    halfcheetahdata, hc_env = load_halfcheetah_dataset(env_name='halfcheetah-medium-v2', size=100*100, seed=0)

    # Load pre-trained encoders
    pre_trained_encoder_seaquest = load_seaquest_model("Seaquest_HalfCheetah/decision_transformer_atari/checkpoints/Seaquest_123.pth", seed=0)
    pre_trained_encoder_halfcheetah = load_halfcheetah_model("Seaquest_HalfCheetah/trajectory_transformer/logs/halfcheetah-medium-v2/gpt/pretrained", seed=0)

    # Create Seaquest test set
    seaquestdata, test_observation_sq = make_seaquest_testset(seaquestdata, data_size, test_size)

    # Create HalfCheetah test set
    halfcheetahdata, test_observation_hc = make_halfcheetah_testset(halfcheetahdata, test_size=2)
    
    # Create trajectories and obtain embeddings for Seaquest data
    final_obs_sq, final_act_sq, final_rew_sq, _ = create_trajectories(
        seaquestdata["observations"], seaquestdata["actions"], 
        seaquestdata["rewards"], seaquestdata["terminals"], trajectory_length=30
    )
    trajectory_embedding_seaquest = get_trajectory_embedding(
        pre_trained_encoder_seaquest, final_obs_sq, final_act_sq, final_rew_sq, 
        is_seaquest=True, device=device
    )

    # Create trajectories and obtain embeddings for HalfCheetah data
    final_obs_hc, final_act_hc, final_rew_hc, final_ter_hc = create_trajectories(
        halfcheetahdata["observations"], halfcheetahdata["actions"], 
        halfcheetahdata["rewards"], halfcheetahdata["terminals"], trajectory_length=25
    )
    trajectory_embedding_halfcheetah = get_trajectory_embedding(
        pre_trained_encoder_halfcheetah, final_obs_hc, final_act_hc, final_rew_hc, is_seaquest=False, device=device
    )

    # Perform clustering on Seaquest data
    clusters_seaquest, _ = perform_clustering_and_plot(trajectory_embedding_seaquest.detach().cpu().numpy(), 2, 8, ccore=True, plot=False)

    # Perform clustering on HalfCheetah data
    clusters_halfcheetah, _ = perform_clustering_and_plot(trajectory_embedding_halfcheetah.detach().cpu().numpy(), 2, 10, ccore=True, plot=False)

    # Train Seaquest model with DiscreteSAC
    print('#' * 100)
    print("Training Seaquest with DiscreteSAC...")
    list_episodes_sq = make_episodes(final_obs_sq, final_act_sq, final_rew_sq, 18)
    sac_sq = fit_discrete_sac(list_episodes_sq, n_steps=1, n_steps_per_epoch=1, device=device)

    # Train HalfCheetah model with SAC
    print('#' * 100)
    print("Training HalfCheetah with SAC...")
    list_episodes_hc = make_episodes(final_obs_hc, final_act_hc, final_rew_hc, 6)
    sac_hc = fit_sac(list_episodes_hc, n_steps=100, n_steps_per_epoch=10, device=device)

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
    
    model_params_hc = {
        'actor_learning_rate': 3e-4,
        'critic_learning_rate': 3e-4,
        'temp_learning_rate': 3e-4,
        'batch_size': 256,
        'scaler': 'pixel',
        'use_gpu': True if device == 'cuda' else False
    }

    # Fit models per cluster for HalfCheetah data
    models_hc, result_data_combinations_hc = fit_per_cluster(
        model=sac_hc, 
        model_class=d3rlpy.algos.SAC, 
        model_params=model_params_hc,
        data_embedding=trajectory_embedding_halfcheetah,
        list_episodes=list_episodes_hc, 
        clusters=clusters_halfcheetah, 
        trajectory_embedding=trajectory_embedding_halfcheetah,
        test_observations=np.expand_dims(test_observation_hc, axis=0)
    )

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

    # Compute attributions for HalfCheetah data
    print('#' * 100)
    print("Attributions HalfCheetah")
    action_dict = {	
        0: 'bthigh',
        1: 'bshin',
        2: 'bfoot',
        3: 'fthigh',
        4: 'fshin',
        5: 'ffoot',            
    }
    attributions_hc = trajectory_attributions_hc(np.expand_dims(test_observation_hc, axis=0), models_hc, trajectory_embedding_halfcheetah, clusters_halfcheetah)
    
    # Print table 2 for HalfCheetah data
    print("Print table 2")
    print_results_hc(result_data_combinations_hc, np.expand_dims(test_observation_hc, axis=0), models_hc, attributions_hc)
    