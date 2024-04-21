import gym
from utils import stack_frames
#import gzip # This is not needed, we had a dataset in .gz format which we couldnt load in GitHub
import copy
from eorl import OfflineDataset
import numpy as np

def load_seaquest_dataset(env_name, size = 717):
    # Initialize the OfflineDataset with specified parameters
    ds = OfflineDataset(
        env = env_name,            # pass name in supported environments below
        dataset_size = 1e6,   # [0, 1e6) frames of atari
        train_split = 0.9,       # 90% training, 10% held out for testing
        obs_only = False,        # only get observations (no actions, rewards, dones)
        framestack = 1,          # number of frames per sample
        shuffle = False,         # chronological samples if False, randomly sampled if true
        stride = 1,               # return every stride`th chunk (where chunk size == `framestack)
        verbose = 1,              # 0 = silent, >0 for reporting
    )

    # Extract the dataset
    datasets = {
        'observations': ds.dataset['observation'],
        'actions': ds.dataset['action'],
        'rewards': ds.dataset['reward'],
        'terminals': ds.dataset['terminal']
    }

    # datasets_names = ["observations", "actions", "rewards", "terminals"]
    # datasets = {}
    # for dataset_name in datasets_names:
    #     with gzip.open("Seaquest_HalfCheetah/data/sq_data/"+dataset_name+".gz", 'rb') as f:
    #         datasets[dataset_name] = np.load(f, allow_pickle=False)
    #print("Number of terminal states is...", datasets['terminals'].sum())
    
    print("Dataset loaded")
    seaquest_length = np.where(np.cumsum(datasets["terminals"]) == size)[0][0]  # Note if terminals are not binary this will not work.
    print(sum(datasets["terminals"]))
    datasets['observations'] = stack_frames(datasets['observations'][:seaquest_length])
    datasets['actions'] = datasets['actions'][:seaquest_length]
    datasets['rewards'] = datasets['rewards'][:seaquest_length]
    datasets['terminals'] = datasets['terminals'][:seaquest_length]

    # Initialize the Gym environment
    env_name = 'ALE/' + env_name + '-ram-v5'
    env = gym.make(env_name)

    env = gym.wrappers.ResizeObservation(env, (84, 84))

    # Print logging information
    print(f"Dataset extracted with {len(datasets['observations'])} samples.")
    print("Information about the dataset:")
    print("Observation shape: ",datasets['observations'].shape)
    print("Action shape: ",datasets['actions'].shape)
    print("Reward shape: ",datasets['rewards'].shape)
    print("Terminal shape: ",datasets['terminals'].shape)
    # Action space is discrete, so we can just print the number of actions
    print("Action space: ",env.action_space.n)
    print("Observation space: ",env.observation_space.shape)
    print("Number of terminal states is...", datasets['terminals'].sum())
    print(f"Environment '{env_name}' initialized.")

    return datasets, env


def make_seaquest_testset(seaquestdata, data_size, test_size):
    
    terminal_indexes = np.where(seaquestdata['terminals'] == True)[0]
    trunc = terminal_indexes[-test_size]
    terminal_indexes = terminal_indexes[data_size:] # Get index of first trajectory that we want to draw

    test_observations = []
    test_actions = []
    for i in range(len(terminal_indexes)):
        ind = [i + np.random.randint(200, 300)]
        test_observations.append(seaquestdata['observations'][ind])
        test_actions.append(seaquestdata['actions'][ind][0])

    observation = copy.deepcopy(test_observations[0])

    # Observation has shape (1, 4, 84, 84)
    # We need to remove the first dimension
    # and plot each channel separately
    observation = observation.squeeze()

    # assert test_observations[0].shape == (1, 4, 84, 84)
    # if not, drop the first dimension
    if test_observations[0].shape != (1, 4, 84, 84):
        test_observations[0] = test_observations[0].squeeze() 

    # Remove all the previous observations, actions, rewards and terminals
    seaquestdata["observations"] = seaquestdata["observations"][:trunc]
    seaquestdata["actions"] = seaquestdata["actions"][:trunc]
    seaquestdata["rewards"] = seaquestdata["rewards"][:trunc]
    seaquestdata["terminals"] = seaquestdata["terminals"][:trunc]

    return seaquestdata, test_observations 
