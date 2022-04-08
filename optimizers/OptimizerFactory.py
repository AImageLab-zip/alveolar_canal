import torch
from torch.optim import Adam, SGD

class OptimizerFactory():
    def __init__(self, name, params, lr):
        super(OptimizerFactory, self).__init__()
        self.name = name
        self.params = params
        self.lr = lr

    def get(self):
        if self.name == 'Adam':
            self.optimizer = Adam(params=self.params, lr=self.lr)
        elif self.name == 'SGD':
            self.optimizer = SGD(params=self.params, lr=self.lr)
        else:
            raise ValueError(f'Unknown optimizer: {self.name}')
        
        self.optimizer.name = self.name
        return self.optimizer

