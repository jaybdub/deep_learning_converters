import torch.nn.functional as F
import torch.nn as nn


class TestModelFC2(nn.Module):

    def __init__(self):
        super(TestModelFC2, self).__init__()
        self.fc1 = F.Linear(4, 2)
        self.fc2 = F.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
