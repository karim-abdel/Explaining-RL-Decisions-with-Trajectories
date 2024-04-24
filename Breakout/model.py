from d3rlpy.dataset import Episode 
import d3rlpy
from encoder import CustomCNNFactory
import torch
import numpy as np

def make_episodes(final_obs, final_act, final_rew, max_action_values):
    list_episodes = []
    for i in range(len(final_obs)):
        list_episodes.append(Episode(final_obs[i].numpy().shape[1:], max_action_values, final_obs[i].numpy(), final_act[i].numpy() ,final_rew[i].numpy()))
    return list_episodes

def fit_discrete_sac(list_episodes, n_steps=1000, n_steps_per_epoch=100, device='cpu'):
    cuda_check = True if device == 'cuda' else False
    feature_size = 128
    custom_encoder_actor = CustomCNNFactory(feature_size)
    custom_encoder_critic = CustomCNNFactory(feature_size)

    # Train an agent on the new data
    sac_sq = d3rlpy.algos.DiscreteSAC(
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            temp_learning_rate=3e-4,
            batch_size=256,
            actor_encoder_factory= custom_encoder_actor,
            critic_encoder_factory= custom_encoder_critic,
            scaler='pixel',
            target_update_interval=2500,
            use_gpu=cuda_check)
    
    sac_sq.fit(list_episodes, n_steps=n_steps, n_steps_per_epoch=n_steps_per_epoch)

    return sac_sq


def fit_sac(list_episodes, n_steps=200, n_steps_per_epoch=100, device='cpu'):
    # Train an agent on the new data
    cuda_check = True if device == 'cuda' else False
    sac = d3rlpy.algos.SAC(
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            temp_learning_rate=3e-4,
            batch_size=256,
            scaler='pixel',
            use_gpu=cuda_check)

    sac.fit(list_episodes, n_steps=n_steps, n_steps_per_epoch=n_steps_per_epoch)
    return sac

def predict(model, list_episodes):
    actions = []
    action_values = []
    for x in range(len(list_episodes)):
        action = model.predict(list_episodes[x].observations)
        actions.append(action)
        #print("Predicted action: ", action)
        action_value = model.predict_value(list_episodes[x].observations, action=action)
        # Assert all action values are positive (action value is an array of size (30, ))
        # try:
        #     assert np.all(action_value >= 0)
        # except AssertionError:
        #     print("Action value is not positive: ", action_value)
        #     print("Action: ", action)
        #     print("Observation: ", list_episodes[x].observations)
        #     print("Reward: ", list_episodes[x].rewards)
        #     print("Done: ", list_episodes[x].terminals)
        #     print("Info: ", list_episodes[x].infos)
        #     print("Observation shape: ", list_episodes[x].observations.shape)
        #     print("Reward shape: ", list_episodes[x].rewards.shape)
        #     print("Done shape: ", list_episodes[x].terminals.shape)
        #     print("Info shape: ", list_episodes[x].infos.shape)
        #     raise AssertionError
        
        action_values.append(action_value)
        
        #print("Predicted action value: ", action_values[-1])

    # Convert the list of numpy arrays to a single numpy array and then to a tensor
    actions_tensor = torch.as_tensor(np.array(actions))
    action_values_tensor = torch.as_tensor(np.array(action_values))

    return actions_tensor, action_values_tensor

def get_data_embedding(traj_embeddings):
    # Convert to numpy arrays if the embeddings are PyTorch tensors
    if isinstance(traj_embeddings[0], torch.Tensor):
        #traj_embeddings = traj_embeddings
        traj_embeddings = [te.cpu().numpy() for te in traj_embeddings]

    # Compute the sum of the embeddings, dividing by 10
    summed_embeddings = np.array(traj_embeddings).sum(axis=0) / 10.0

    # Apply softmax
    #exp_embeddings = np.exp(summed_embeddings)
    #softmax_embeddings = exp_embeddings / np.sum(exp_embeddings)
    max_embeddings = np.max(summed_embeddings)
    exp_embeddings = np.exp(summed_embeddings - max_embeddings)
    softmax_embeddings = exp_embeddings / np.sum(exp_embeddings, axis=0)

    return softmax_embeddings

def fit_per_cluster(model, 
                    model_class, 
                    model_params, 
                    data_embedding, 
                    list_episodes, 
                    clusters, 
                    trajectory_embedding, 
                    test_observations):
    
    original_actions = []
    original_action_values = []
    for t in test_observations:
        action = model.predict(t)
        print("Original model action: ", action)
        action_value = model.predict_value(t, action = action)
        print("Original model action value: ", action_value)
        print("Passing: ", action[0], action_value[0])
        original_actions.append(action[0])
        original_action_values.append(action_value[0])

    result_data_combinations = {0:(original_actions, original_action_values, get_data_embedding(data_embedding))} # Original dataset policy
    models = {0:(model, get_data_embedding(data_embedding))}
    
    for cluster_id, cluster in enumerate(clusters):
        model = model_class(**model_params)
        count_in_clusters = 0
        count_not_in_clusters = 0
        temp_data = []
        temp_traj_embeds = []
        temp_cluster_traj_embeds = []
        for traj_id, traj in enumerate(list_episodes):
            if traj_id not in cluster:
                temp_data.append(list_episodes[traj_id])
                temp_traj_embeds.append(trajectory_embedding[traj_id])
                count_not_in_clusters += 1
            else:
                temp_cluster_traj_embeds.append(trajectory_embedding[traj_id])
                count_in_clusters += 1
        print('-'*100)
        print(f'Cluster {cluster_id + 1} has {count_in_clusters} trajectories out of {count_not_in_clusters + count_in_clusters}')
        print('-'*100)
        
        # Generate the data embedding
        data_embedding_new = get_data_embedding(temp_traj_embeds)
        # Train an agent on the new data
        model.fit(temp_data, n_steps=2000, n_steps_per_epoch=200)

        models[cluster_id + 1] = (model, data_embedding_new)

        new_actions = []
        new_action_values = []
        for t in test_observations:
            action = model.predict(t)
            print("New model action: ", action)
            action_value = model.predict_value(t, action = action)
            print("New model action value: ", action_value)
            print("Passing: ", action[0], action_value[0])

            new_actions.append(action[0])
            new_action_values.append(action_value[0])
        
        result_data_combinations[cluster_id + 1] = (new_actions, new_action_values, data_embedding_new) # Clustered dataset policy (cluster_id + 1 because 0 is the original dataset)
                                                                                    # so each value is the model trained on original data - cluster 1, original data - cluster 2, etc.
        
    return models, result_data_combinations