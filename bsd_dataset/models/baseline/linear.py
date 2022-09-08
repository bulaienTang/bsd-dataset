import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import List

__all__ = [
    "linearRegression"
]

class linearRegression(nn.Module):
    def __init__(self, input_shape: List[int], target_shape: List[int]):
        super(linearRegression, self).__init__()
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.linear = nn.Linear(self.input_shape[1], self.target_shape[2])

    def forward(self, x):
        x = self.linear(x)
        return x.squeeze(1)