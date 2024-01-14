import numpy as np
import d3rlpy as d3
import torch
import gym
from RLModels.DiscreteRL import DiscreteModel
from SeqEncoders.encoder import encodeTrajectories
from Clustering.XMeans import XMeans
from Embedders.trainExpPolicy import trainExpPolicies
from Clustering.ClusterAttribution import generateClusterAttribution
from decision_transformer_atari.decision_transformer_atari import GPTConfig, GPT



if __name__ == "__main__":
    # Example usage:


    # Pre-trained Encoder
    vocab_size = 18 #Need to change this otherwise crashes but is this correct??
    block_size = 90
    model_type = "reward_conditioned"
    timesteps = 2719 #Need to change this otherwise crashes but is this correct??

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

    checkpoint_path = "decision_transformer_atari\checkpoints\Seaquest_123.pth"  # or Pong, Qbert, Seaquest
    
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        print("Be careful you are running on cpu")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint)

    # Create an instance of the OfflineRLTrainer class
    discreteSAC = DiscreteModel(env_name="Seaquest-v4")
    # Train the model:
    discreteSAC.train()

    #=============================================================

    # 3. Cluster the trajectories using XMeans.
    XMeans = XMeans()
    clusters = XMeans.clusterTrajectories(trajectory_embeddings)

    # 4 and 5. Generate data embeddings using generateDataEmbedding function Train the offline RL algorithm using trainOfflineRL function

    # Placeholder function for offlineRLAlgo
    def offlineRLAlgo(complementary_data):
        # Return a dummy policy (could be a more complex object or model in practice)
        return {'policy': 'dummy_policy'}

    # Fake offline data with two trajectories
    offline_data = [
        [([0.1, 0.2], [1], 0.5), ([0.2, 0.3], [0], 0.6)],  # First trajectory
        [([0.3, 0.4], [1], 0.4), ([0.4, 0.5], [0], 0.7)]   # Second trajectory
    ]

    # Fake trajectory embeddings
    # Assuming we have an embedding of size 3 for each trajectory
    trajectory_embeddings = [np.random.rand(3) for _ in offline_data]

    # Fake clusters of trajectories
    clusters = [[0], [1]]

    # Now we call the trainExpPolicies function with the fake data
    explanation_policies, data_embeddings = trainExpPolicies(
        offline_data, trajectory_embeddings, clusters, offlineRLAlgo)


    # Perform cluster attribution for a specific state using generateClusterAttribution function
    state = np.array([0.1, 0.2, 0.3]) # Random state

    def mock_policy(state):
    # Return a dummy action that is the same shape as the state
        return state * 0.5  # Simple transformation for the sake of example
    
    
    original_policy=mock_policy
    original_data_embedding = np.array([0.5, 0.5, 0.5])
    explanation_policies = [mock_policy, mock_policy, mock_policy]

    chosen_cluster = generateClusterAttribution(
        state, original_policy, explanation_policies, original_data_embedding, data_embeddings)

    print("Attributed cluster:", chosen_cluster)

