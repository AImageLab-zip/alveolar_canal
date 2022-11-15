import torch
from torch.optim import Adam, SGD

class OptimizerFactory():
    def __init__(self, name, params, lr, weight_decay=0, momentum=0):
        super(OptimizerFactory, self).__init__()
        self.name           = name
        self.params         = params
        self.lr             = lr
        self.weight_decay   = weight_decay
        self.momentum       = momentum 

    def get(self):
        if self.name == 'Adam':
            self.optimizer = Adam(params=self.params, lr=self.lr)
        elif self.name == 'SGD':
            self.optimizer = SGD(params=self.params, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        else:
            raise ValueError(f'Unknown optimizer: {self.name}')
        
        self.optimizer.name = self.name
        return self.optimizer

