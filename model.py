import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class DQN_GRAC(nn.Module):
	def __init__(self,in_channels, action_dim):
		super(DQN_GRAC, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.fc4 = nn.Linear(7 * 7 * 64, 512)
		self.fc5 = nn.Linear(512, action_dim)

		self.conv6 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
		self.conv7 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.fc9 = nn.Linear(7 * 7 * 64, 512)
		self.fc10 = nn.Linear(512, action_dim)

	def forward_all(self, x):
		x = x.view((-1,4,84,84))
		q1 = F.relu(self.conv1(x))
		q1 = F.relu(self.conv2(q1))
		q1 = F.relu(self.conv3(q1))
		q1 = F.relu(self.fc4(q1.view(q1.size(0), -1)))
		q1 = self.fc5(q1)

		q2 = F.relu(self.conv6(x))
		q2 = F.relu(self.conv7(q2))
		q2 = F.relu(self.conv8(q2))
		q2 = F.relu(self.fc9(q2.view(q2.size(0), -1)))
		q2 = self.fc10(q2)
		return q1, q2

	def forward(self, state, action):
		q1, q2 = self.forward_all(state)
		action = action.view((-1,1))
		q1 = q1.gather(1,action)
		q2 = q2.gather(1,action)
		return q1, q2
        
class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Dueling_DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc1_val = nn.Linear(in_features=7*7*64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)
        
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x






