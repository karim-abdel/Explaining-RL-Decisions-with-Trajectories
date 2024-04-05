import torch
import torch.nn as nn
from d3rlpy.models.encoders import EncoderFactory

class CustomCNNEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.device = 'cuda' if torch.cuda.is_available()  else 'cpu'
        
        try:
            if torch.backends.mps.is_available():
                self.device = 'mps'
        except: # With some versions of torch on windows this can crash.
            pass

        # Define the CNN architecture with efficient layer arrangements
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),  # In-place operation for memory efficiency
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        ).to(self.device)

        # Efficiently compute the size of the flattened features after conv layers
        with torch.no_grad():
            self.feature_dim = self._get_conv_output(observation_shape)

        # Define the final linear layer
        self.fc = nn.Linear(self.feature_dim, feature_size).to(self.device)

    def _get_conv_output(self, shape):
        dummy_input = torch.zeros(1, *shape, device=self.device)
        return self.conv_layers(dummy_input).view(1, -1).size(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Efficient flattening of the output
        return torch.relu(self.fc(x))

    def get_feature_size(self):
        return self.feature_size

class CustomCNNFactory(EncoderFactory):
    TYPE = 'custom'

    def __init__(self, feature_size):
        self.feature_size = feature_size

    def create(self, observation_shape):
        return CustomCNNEncoder(observation_shape, self.feature_size)

    def get_params(self, deep=False):
        return {'feature_size': self.feature_size}