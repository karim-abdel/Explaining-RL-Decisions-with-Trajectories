import numpy as np

def generateDataEmbedding(trajectory_embeddings, M, Tsoft):
    s = np.sum(trajectory_embeddings, axis=0) / M
    d = np.exp(s / Tsoft) / np.sum(np.exp(s / Tsoft))
    return d

# Step (iv) - Training Explanation Policies
def trainExpPolicies(offline_data, trajectory_embeddings, clusters, offlineRLAlgo):
    explanation_policies = []
    data_embeddings = []
    for cj in clusters:
        complementary_data = [
            trajectory for trajectory in offline_data if trajectory not in cj]
        Tj = [embedding for i, embedding in enumerate(
            trajectory_embeddings) if i not in cj]
        policy = offlineRLAlgo(complementary_data)
        data_embedding = generateDataEmbedding(Tj, M=100, Tsoft=0.1)
        explanation_policies.append(policy)
        data_embeddings.append(data_embedding)
    return explanation_policies, data_embeddings
