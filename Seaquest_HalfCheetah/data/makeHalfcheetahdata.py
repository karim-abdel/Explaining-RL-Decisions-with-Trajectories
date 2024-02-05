import gym
import d4rl # Import required to register environments, you may need to also import the submodule
import numpy as np

def load_halfcheetah_dataset(env_name='halfcheetah-medium-v0', size=1000*1000,seed=42):
    #Size Given the definition of end of trajectory: https://www.gymlibrary.dev/environments/mujoco/half_cheetah/
    # Create the environment
    env = gym.make(env_name)
    env.seed(seed)

    # d4rl abides by the OpenAI gym interface
    env.reset()
    env.step(env.action_space.sample())

    # Alternatively, use d4rl.qlearning_dataset which
    # also adds next_observations.
    #dataset = d4rl.qlearning_dataset(env)
    dataset = env.get_dataset()

    dataset['observations'] = dataset['observations'][:size]
    dataset['actions'] = dataset['actions'][:size]
    dataset['rewards'] = dataset['rewards'][:size]

    # Print logging information
    print(f"Dataset extracted with {len(dataset['observations'])} samples.")
    print("Information about the dataset:")
    print("Observation shape: ",dataset['observations'].shape)
    print("Action shape: ",dataset['actions'].shape)
    print("Reward shape: ",dataset['rewards'].shape)
    #print("Terminal shape: ",dataset['terminals'].shape)
    # Action space is discrete, so we can just print the number of actions
    print("Action space: ",env.action_space.shape)
    print("Observation space: ",env.observation_space.shape)
    print(f"Environment '{env_name}' initialized.")

    
    return dataset, env

def make_halfcheetah_testset(halfcheetahdata, test_size=10):
    # select N test observations from the dataset
    hc_test_indices = np.random.randint(0, 100, size=(test_size,))
    test_observation = halfcheetahdata['observations'][hc_test_indices]
    # drop the test observations from the dataset
    halfcheetahdata['observations'] = np.delete(halfcheetahdata['observations'], hc_test_indices, axis=0)

    return halfcheetahdata, test_observation