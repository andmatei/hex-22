import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels))
        
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
    
class Actor(nn.Module):
    def __init__(self, in_channels, out_size):
        super(Actor, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(2, out_size),
            nn.Softmax())

    def forward(self, x):
        x = self.conv1(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)

        return x
    

class Critic(nn.Module):
    def __init__(self, in_channels):
        super(Critic, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(2, 1),
            nn.Tanh())

    def forward(self, x):
        x = self.conv1(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)

        return x

class ActorCritic(nn.Module):
    def __init__(self, input_channels, output_size):
        super(ActorCritic, self).__init__()

        self.block = ResidualBlock(input_channels, 256)
        self.actor = Actor(256, output_size)
        self.critic = Critic(256)

    def forward(self, x):
        x = self.block(x)
        return self.actor(x), self.critic(x)