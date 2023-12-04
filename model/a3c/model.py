import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(torch.nn.Module):
    def __init__(self, input_size, output_size, *args, **kwargs):
        super(ActorCritic, self).__init__(*args, **kwargs)

        self.conv = nn.Conv2d(input_size, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 6 * 6, 256)
        
        self.critic = nn.Linear(256, 1)        
        self.actor = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x.torch.relu(self.fc(x))
        return self.actor(x), self.critic(x)