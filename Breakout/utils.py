import numpy as np
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from model import get_data_embedding
from torch.cuda.amp import autocast, GradScaler  # For mixed precision
import logging
import random

torch.backends.cudnn.benchmark = True

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available()  else 'cpu'
    
    try:
        if torch.backends.mps.is_available():
            device = 'mps'
    except: # With some versions of torch on windows this can crash.
        pass

    return device


def generate_model(env, trajs, device='cpu'):
    len_act_space = len(env.action_space)
    dims = np.prod(env.dim)

    # Initialize the transitions on the specified device
    transition_model = torch.zeros((dims, len_act_space, dims), device=device)
    reward_model = torch.zeros((dims, len_act_space, dims), device=device)

    for traj in trajs:
        for (s, a, r, s_) in traj:
            transition_model[s, a, s_] += 1
            reward_model[s, a, s_] += r

    # Avoid dividing by zero and unnecessary device transfers
    transition_sums = transition_model.sum(2)[:, :, None] + 1e-9
    transition_model = transition_model / transition_sums
    reward_model = reward_model / transition_sums

    # Convert to CPU only if necessary, else keep on GPU/MPS
    return transition_model.cpu(), reward_model.cpu()

def create_trajectories(observations, actions, rewards, terminals, trajectory_length):
    """
    Create trajectories from the given observations, actions, rewards, and terminals.
    Parameters:
        observations (torch.Tensor): Tensor of shape [num_frames, height, width].
        stack_size (int): Number of frames to stack for each observation.
    Returns:
        torch.Tensor: Stacked observations of shape [(num_frames-stack_size+1), stack_size, height, width].
    """
    # Efficiently compute the number of trajectories
    terminal_indices = [i for i, x in enumerate(terminals) if x]
    #print("DEBUG: terminal_indices", terminal_indices)
    padded_length = sum(trajectory_length - (i % trajectory_length) for i in terminal_indices)
    total_length = len(observations) + padded_length
    num_trajectories = total_length // trajectory_length

    # Tensor shapes
    obs_shape = observations[0].shape
    act_shape = actions[0].shape
    rew_shape = rewards[0].shape
    term_shape = terminals[0].shape

    # Pre-allocate tensors
    final_obs = torch.zeros((num_trajectories, trajectory_length, *obs_shape), dtype=torch.uint8)
    final_act = torch.zeros((num_trajectories, trajectory_length, *act_shape))
    final_rew = torch.zeros((num_trajectories, trajectory_length, *rew_shape))
    final_ter = torch.zeros((num_trajectories, trajectory_length, *term_shape))

    # Populate tensors
    trajectory_idx = 0
    step_idx = 0
    for i in tqdm(range(len(observations)), desc="Processing sub'trajectories"):
        final_obs[trajectory_idx, step_idx] = torch.from_numpy(observations[i])
        final_act[trajectory_idx, step_idx] = torch.as_tensor(actions[i])
        final_rew[trajectory_idx, step_idx] = torch.as_tensor(rewards[i])
        final_ter[trajectory_idx, step_idx] = torch.as_tensor(terminals[i])
        step_idx += 1

        if terminals[i] or step_idx == trajectory_length:
            trajectory_idx += 1
            step_idx = 0
            if trajectory_idx >= num_trajectories:
                break


    return final_obs, final_act, final_rew, final_ter

def stack_frames(dataset, stack_size=4):
    """
    Stack consecutive frames in the dataset with padding for initial frames.
    
    :param dataset: A numpy array of shape [num_frames, width, height]
    :param stack_size: Number of frames to stack
    :return: A numpy array of shape [num_frames, stack_size, width, height]
    """
    
    # Padding the dataset with zeros at the beginning
    padded_dataset = np.pad(dataset, ((stack_size-1, 0), (0, 0), (0, 0)), mode='constant')

    # Stacking frames
    num_frames = len(dataset)
    indices = np.arange(stack_size) + np.arange(num_frames)[:, None]
    stacked_dataset = padded_dataset[indices]

    return stacked_dataset

def get_trajectory_embedding(model, observations, actions, rewards, is_seaquest=False, device='cpu'):
    """
    Calculates the trajectory embedding for a given model, observations, actions, and rewards.

    Args:
        model (torch.nn.Module): The model used to calculate the embedding.
        observations (list): List of observations.
        actions (list): List of actions.
        rewards (list): List of rewards.
        stack_frames_fn (function, optional): Function to stack frames for Seaquest environment. Defaults to None.
        is_seaquest (bool, optional): Flag indicating if the environment is Seaquest. Defaults to False.

    Returns:
        list: List of trajectory embeddings.
    """
    def reshape_input(observations, actions, rewards, device='cpu'):
        input_obs = observations.view(1, -1, 4*84*84).to(device)
        input_act = actions.view(1, -1, 1).to(device)
        input_rew = rewards.view(1, -1, 1).to(device)
        timesteps = torch.as_tensor([[[1]]], dtype=torch.long).to(device)  # Adjust as needed
        return input_obs, input_act, input_rew, timesteps

    model = model.to(device)
    scaler = GradScaler()

     # Adjust this based on your model's output
    if is_seaquest:
        input_obs, input_act, input_rew, timesteps = reshape_input(observations[0], actions[0], rewards[0], device=device)
        embedding = model(input_obs, input_act, rtgs=input_rew, timesteps=timesteps).detach()
        MAX_SIZE = torch.mean(embedding, dim=1).flatten().shape[0]
    else:
        combined_input = torch.cat([observations[0], actions[0], rewards[0].unsqueeze(1)], dim=1).type(torch.long)
        combined_input = combined_input.to(device)
        MAX_SIZE = model(combined_input).detach().flatten().shape[0]

    # If you encounter problem with memory, you can use memmap to store the embeddings on disk
    filename = 'seaquest_trajectory_embedding.dat' if is_seaquest else 'halfcheetah_trajectory_embedding.dat'
    #trajectory_embedding = np.memmap(filename, dtype='float16', mode='w+', shape=(len(observations), MAX_SIZE))

    trajectory_embedding = torch.zeros((len(observations), MAX_SIZE), dtype=torch.float16, device=device)

    for ind, (obs, act, rew) in enumerate(tqdm(zip(observations, actions, rewards), total=len(observations), desc="Processing trajectories")):
        with autocast():
            if is_seaquest:
                input_obs, input_act, input_rew, timesteps = reshape_input(obs, act, rew, device=device)
                embedding = model(input_obs, input_act, rtgs=input_rew, timesteps=timesteps).detach()
                embedding = torch.mean(embedding, dim=1).flatten()
            else:
                combined_input = torch.cat([obs, act, rew.unsqueeze(1)], dim=1).type(torch.long)
                combined_input = combined_input.to(device)
                embedding = model(combined_input).detach()
                embedding = torch.mean(embedding, dim=1).flatten()
            

        # Ensure the embedding size does not exceed MAX_SIZE
        embedding_size = min(len(embedding), MAX_SIZE)
        trajectory_embedding[ind, :embedding_size] = embedding[:embedding_size].cpu()

    # Synchronize changes to disk and close the memmap file
    #del trajectory_embedding

    # Reopen the memmap file in read-only mode for further use
    #trajectory_embedding = np.memmap(filename, dtype='float16', mode='r', shape=(len(observations), MAX_SIZE))

    return trajectory_embedding


def perform_clustering_and_plot(traj_embeddings, amount_initial_centers, max_clusters, ccore=False,plot=True):
    """
    Performs clustering on prepared trajectory embeddings using X-Means and plots the results.

    :param traj_embeddings: Prepared trajectory embeddings.
    :param amount_initial_centers: Initial number of centers for clustering.
    :param max_clusters: Maximum number of clusters.
    :return: None
    """
    # Create a basic logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logger.info('Starting clustering process.')

    # Initialize centers using kmeans_plusplus_initializer
    initial_centers = kmeans_plusplus_initializer(traj_embeddings, amount_initial_centers).initialize()

    logger.info('Initial centers initialized.')

    # Create and process X-Means instance
    xmeans_instance = xmeans(traj_embeddings, initial_centers, max_clusters, ccore=ccore)
    xmeans_instance.process()

    logger.info('X-Means instance processed.')
    
    # Extract clustering results: clusters and their centers
    clusters = xmeans_instance.get_clusters()

    logger.info('Clustering results extracted.')
    
    # Assign cluster labels to each trajectory
    traj_cluster_labels = np.zeros(len(traj_embeddings), dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        for traj_id in cluster:
            traj_cluster_labels[traj_id] = cluster_id

    logger.info('Cluster labels assigned to each trajectory.')

    # Perform PCA for visualization
    pca_traj = PCA(n_components=2)
    pca_traj_embeds = pca_traj.fit_transform(traj_embeddings)
    plotting_data = {
        'feature 1': pca_traj_embeds[:, 0],
        'feature 2': pca_traj_embeds[:, 1],
        'cluster id': traj_cluster_labels
    }
    df = pd.DataFrame(plotting_data)

    logger.info('PCA performed for visualization.')

    if plot:
        # Plotting
        plt.figure(figsize=(4,3))
        palette = sns.color_palette('husl', len(clusters) + 1)
        sns.scatterplot(
            x='feature 1',
            y='feature 2',
            hue='cluster id',
            palette=palette[:len(clusters)],
            data=df,
            legend=True
        )
        plt.title('Trajectory Embeddings for ' + str(amount_initial_centers) + ' initial centers')
        plt.legend(title = '$c_{j}$', loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=5)
        plt.tight_layout()
        plt.show()

        logger.info('Plot created.')

    return clusters, traj_cluster_labels

def trajectory_attributions_sq(test_observations, models, traj_embeddings, clusters):
    attributions = []
    data_embedding = get_data_embedding(traj_embeddings)
    # First model is trained on full dataset
    # Second model is trained on cluster 1
    # Third model is trained on cluster 2 ...
    original_model = models[0][0]
    original_data_embedding = models[0][1]
    for i in range (len(test_observations)):
        print('#' * 100, f'Observation {i + 1}', '#' * 100)
        print
        t = test_observations[i]
        print(t.shape)
        # plt.imshow(t[0][0], cmap='gray')
        # plt.savefig(f'test_observation_{i + 1}.png')
        original_action = original_model.predict(t)
        sub_dist = []
        for k in models:
            #v[0] # model
            #v[1] # data_embedding
            model = models[k][0]
            data_embedding = models[k][1]

            action = model.predict(t)
            print(f"{k}-th model predicted action:", action)
            print("Original model's action", original_action)

            if action != original_action:
                w_d = wasserstein_distance(original_data_embedding, data_embedding)
                print("Wasserstein distance: ", w_d)
            else:
                w_d = 1e9
            
            sub_dist.append(w_d)
        
        print("DEBUG: len(sub_dist)", len(sub_dist))
        print("DEBUG: sub_dist", sub_dist)
        print("DEBUG: np.argsort(sub_dist)", np.argsort(sub_dist))
        responsible_data_combination = np.argsort(sub_dist)[0] - 1
        print("Responsible data combination: ", responsible_data_combination)

        if sub_dist[responsible_data_combination] == 1e9 or sub_dist[responsible_data_combination] == 0:
            print("No attribution found")
            print(sub_dist[responsible_data_combination])
            continue
    
        attributions.append({
                    'models': [v[0] for v in models.values()],
                    'orig_act': original_action,
                    'new_act': action,
                    'attributed_trajs': clusters[responsible_data_combination],
                    'responsible_cluster': responsible_data_combination
                })
    
    return attributions



def trajectory_attributions_hc(test_observations, models, traj_embeddings, clusters):
    distances = []
    attributions = []
    new_actions = []
    data_embedding = get_data_embedding(traj_embeddings)
    # First model is trained on full dataset
    # Second model is trained on cluster 1
    # Third model is trained on cluster 2 ...
    original_model = models[0][0]
    original_data_embedding = models[0][1]
    
    distances = np.zeros((len(test_observations), len(models) -1), dtype=np.float32)

    t = np.expand_dims(test_observations[0], axis=0)
    original_action = original_model.predict(t)
    sub_dist = []
    for k in models:
        if k == 0:
            continue
        #v[0] # modello
        #v[1] # data_embedding
        model = models[k][0]
        data_embedding = models[k][1]

        action = model.predict(t)
        for index_act,sub_act in enumerate(action): 
            if not np.all(np.isclose(np.array(sub_act), np.array(original_action[index_act]))):
                w_d = wasserstein_distance(original_data_embedding, data_embedding)
            else:
                w_d = 1e9
            sub_dist.append(w_d)
            distances[0,k-1] = w_d
        responsible_data_combination = np.argsort(sub_dist)[0]


        if sub_dist[responsible_data_combination] == 1e9 or sub_dist[responsible_data_combination] == 0:
            print("No attribution found")
        else:
            attributions.append({
                    'models': [v[0] for v in models.values()],
                    'orig_act': original_action,
                    'new_act': action,
                    'attributed_trajs': clusters[responsible_data_combination],
                    'responsible_cluster': responsible_data_combination
                })
        
    return attributions


def print_results_hc(result_data_combinations, test_observation, models, attributions):
    
    for data_combination_id in result_data_combinations:
        print(np.array(result_data_combinations[data_combination_id][1]).mean())

    
    print("Comparing the actions")
    # result_data_comb[0][0] --> original action
    # result_data_comb[0][1] --> original action value
    # result_data_comb[0][2] --> original data embedding

    
    # result_data_comb[i][0] --> cluster i action
    # to compare with original actions, use result_data_comb[0][0]
    # compare for each cluster i, the actions with the original actions and count the number of actions that are the same
    
    action_comparison = {}
    for data_combination_id, (action_new, action_value_new, data_embedding_new) in result_data_combinations.items():
        action_comparison[data_combination_id] = 0
        for t in range(len(test_observation)):
                action_comparison[data_combination_id] +=  np.mean(np.square((action_new[t] - result_data_combinations[0][0][t])))
    
    print(np.array(list(action_comparison.values()))/(len(test_observation)))
                
    
    print('Avg Delta Q')
    for data_combination_id, (_, action_values_new, data_embedding_new) in result_data_combinations.items():
        print(np.sum(np.abs(np.array(action_values_new) - np.array(result_data_combinations[0][1])))/(len(test_observation)))
        
    data_embedding_original = models[0][1]
    print("Data distances")
    data_distances = np.zeros(len(result_data_combinations))
    (_, action_values_new, data_embedding_new)
    for data_combination_id, (_, action_values_new, data_embedding_new) in result_data_combinations.items(): 
        data_distances[data_combination_id] = wasserstein_distance(data_embedding_original, data_embedding_new)

    with np.printoptions(precision=8, suppress=True):
        print((data_distances - data_distances.min()) / (data_distances.max() - data_distances.min()))

    cluster_attr_freq = np.zeros(len(models))

    for attribution in attributions:
        cluster_attr_freq[attribution['responsible_cluster']] += 1 
    
    print(cluster_attr_freq/cluster_attr_freq.sum())



def print_results_sq(result_data_combinations_sq, test_observations, models_sq, attributions_sq):
 
    # result_data_comb[i][0] --> cluster i action
    # to compare with original actions, use result_data_comb[0][0]
    # compare for each cluster i, the actions with the original actions and count the number of actions that are the same
    action_comparison = {}
    for data_combination_id, (action_new, action_value_new, data_embedding_new) in result_data_combinations_sq.items():
        action_comparison[data_combination_id] = 0
        for t in range(len(test_observations)):
            assert len(action_new) == len(test_observations)
            if (action_new[t] == result_data_combinations_sq[0][0][t]):
                action_comparison[data_combination_id] += 1
    
    print(np.array(list(action_comparison.values()))/(len(test_observations)))

    print('Avg Delta Q')
    for data_combination_id, (_, action_values_new, data_embedding_new) in result_data_combinations_sq.items():
        print(np.sum(np.abs(np.array(action_values_new) - np.array(result_data_combinations_sq[0][1])))/(len(test_observations) - 1))
    
    data_embedding_original = models_sq[0][1]
    print("Data distances")
    data_distances = np.zeros(len(result_data_combinations_sq))
    for data_combination_id, (_, action_values_new, data_embedding_new) in result_data_combinations_sq.items(): 
        data_distances[data_combination_id] = wasserstein_distance(data_embedding_original, data_embedding_new)

    with np.printoptions(precision=8, suppress=True):
        print((data_distances - data_distances.min()) / (data_distances.max() - data_distances.min()))

    cluster_attr_freq = np.zeros(len(models_sq))
    for attribution in attributions_sq:
        cluster_attr_freq[attribution['responsible_cluster']] += 1 
    
    print(cluster_attr_freq/cluster_attr_freq.sum())
