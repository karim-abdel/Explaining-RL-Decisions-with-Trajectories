import numpy as np

def calculateActionDistance(action1, action2):
    # Replace with the action distance calculation function
    return np.linalg.norm(action1 - action2)


def calculateWassersteinDistance(data_embedding1, data_embedding2):
    # Replace with the Wasserstein distance calculation function
    return np.linalg.norm(data_embedding1 - data_embedding2)

# Step (v) - Cluster Attribution
def generateClusterAttribution(state, original_policy, explanation_policies, original_data_embedding, data_embeddings):
    original_action = original_policy(state)
    action_distances = []
    for policy in explanation_policies:
        action = policy(state)
        action_distance = calculateActionDistance(
            original_action, action)  # Implement this function
        action_distances.append(action_distance)
    candidate_clusters = np.argmax(action_distances)
    w_distances = []
    for data_embedding in data_embeddings:
        w_distance = calculateWassersteinDistance(
            original_data_embedding, data_embedding)  # Implement this function
        w_distances.append(w_distance)
    chosen_cluster = np.argmin(w_distances)
    return chosen_cluster
