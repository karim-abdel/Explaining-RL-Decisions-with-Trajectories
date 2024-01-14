import numpy as np

# Define your sequence encoder function E
def sequence_encoder(observation, action, reward):
    # Dumb data
    return np.random.rand(128), np.random.rand(64), np.random.rand(32)

# Step (i) - Trajectory Encoding
def encodeTrajectories(offline_data):
    trajectory_embeddings = []
    for trajectory in offline_data:
        observation_tokens = []
        action_tokens = []
        reward_tokens = []
        for (observation, action, reward) in trajectory:
            (eo, ea, er) = sequence_encoder(observation, action, reward)
            observation_tokens.extend(eo)
            action_tokens.extend(ea)
            reward_tokens.extend(er)
        total_tokens = len(observation_tokens) + \
            len(action_tokens) + len(reward_tokens)
        trajectory_embedding = np.mean(
            observation_tokens + action_tokens + reward_tokens) / (3 * total_tokens)
        trajectory_embeddings.append(trajectory_embedding)
    return trajectory_embeddings