import torch
from torch.autograd import Variable
from typing import List

__all__ = [
    "LinReg"
]

class linearRegression(torch.nn.Module):
    def __init__(self, input_shape: List[int], target_shape: List[int]):
        super(linearRegression, self).__init__()
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.linear = torch.nn.Linear(self.input_shape[1], self.target_shape[2])

    def forward(self, x):
        x = self.linear(x)
        return x.squeeze(1)