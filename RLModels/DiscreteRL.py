import d3rlpy
from d3rlpy.algos import DiscreteSAC as DSC
#from d3rlpy.metrics.scorer import evaluate_on_environment
#from d3rlpy.algos import train_iter
import gym

class DiscreteModel:
    def __init__(self, env_name, dataset_path, actor_lr=0.0003, critic_lr=0.0003, seed=42, use_gpu=True):
        self.env_name = env_name
        self.dataset_path = dataset_path
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.seed = seed
        self.use_gpu = use_gpu
        self.model = None
        self.env = gym.make(self.env_name)
        self.env.seed(self.seed)

    def load_dataset(self):
        dataset = d3rlpy.datasets.get_dataset(self.dataset_path)
        return dataset

    def initialize_model(self):
        self.model = DSC(actor_learning_rate=self.actor_lr, critic_learning_rate=self.critic_lr, use_gpu=True) 
    
    def train(self, n_epochs=100, n_steps_per_epoch=1000, evaluate_every=10):
        dataset = self.load_dataset()
        self.initialize_model()
        return 
        #iterator = train_iter(dataset, self.model, n_steps=n_epochs * n_steps_per_epoch, n_steps_per_epoch=n_steps_per_epoch)

        #for epoch, metrics in enumerate(iterator):
            #print(f"Epoch: {epoch}, Metrics: {metrics}")

            #if epoch % evaluate_every == 0:
                #print()
                #scores = evaluate_on_environment(self.env)()
                #print(f"Mean Reward: {scores}")

    def save_model(self, save_path):
        # Save the trained policy
        self.model.save_policy(save_path)
        print(f"Trained policy saved at: {save_path}")