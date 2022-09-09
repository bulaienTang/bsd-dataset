import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import List

__all__ = [
    "linearRegression"
]

class linearRegression(nn.Module):
    def __init__(self, input_shape: List[int], target_shape: List[int]): 
        #[1, 3, 18] [16, 100]
        super(linearRegression, self).__init__()
        self.w = torch.randn(size = (input_shape[1]*input_shape[2], target_shape[0]*target_shape[1]))
        self.b = torch.randn(size = (target_shape[0]*target_shape[1],1))
        self.input_shape = input_shape
        self.target_shape = target_shape
        
        # self.linear = nn.Linear(input_shape[1], target_shape[1])

    def forward(self, x):
        x = x.squeeze(1)
        print(x.shape)
        x = x.view(1, x.shape[0] * x.shape[1])
        pred = x @ self.w + self.b
        pred = pred.view(self.target_shape[0], self.target_shape[1])
        return pred