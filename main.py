import numpy as np
import d3rlpy as d3
import torch
import gym
from RLModels.DiscreteRL import DiscreteModel
from Clustering.XMeans import XMeans
from Embedders.trainExpPolicy import trainExpPolicies
from Clustering.ClusterAttribution import generateClusterAttribution
from decision_transformer_atari.load_model import load_model
from data.makeSeaquestdata import load_dataset



if __name__ == "__main__":
    #Pre trained model from hugface
    seaquestdata = load_dataset(env_name='Seaquest-v4', dataset_path='data/SeaQuestdataset.pkl', seed=42)
    pre_trained_encoder = load_model()

    for trajectory in seaquestdata:
        print(trajectory)
        trajectory_embedding = pre_trained_encoder.state_encoder(trajectory)

    # 2. Cluster the trajectories using XMeans.
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
    
    # Create an instance of the OfflineRLTrainer class
    discreteSAC = DiscreteModel()
    # Train the model:
    discreteSAC.train(chosen_cluster, n_epochs=256)

    # Save the trained model
    discreteSAC.save_model('trained_model.pt')
