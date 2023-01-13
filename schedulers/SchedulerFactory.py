import torch
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR, ReduceLROnPlateau


class SchedulerFactory():
    def __init__(self, name, optimizer, LR=1e-4, steps_per_epoch=1, epochs=100, **kwargs):
        super(SchedulerFactory, self).__init__()
        self.name = name
        self.optimizer = optimizer
        self.kwargs = kwargs
        self.LR = LR
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs

    def get(self):
        if self.name == 'MultiStepLR':
            self.kwargs = {
                    'milestones': self.kwargs.get('milestones'),
                    'gamma': self.kwargs.get('gamma', 0.1),
                    }
            scheduler = MultiStepLR(self.optimizer, **self.kwargs)

        elif self.name == 'Plateau':
            self.kwargs = {
                    'mode': self.kwargs.get('mode', None),
                    'patience': self.kwargs.get('patience', None),
                    'verbose': True,
                    }
            scheduler = ReduceLROnPlateau(self.optimizer, **self.kwargs)

        elif self.name == 'OneCycleLR':
            self.kwargs = {
                    'max_lr':           self.LR,
                    'steps_per_epoch':  self.steps_per_epoch,
                    'epochs':           self.epochs,
                    }
            scheduler = OneCycleLR(self.optimizer, **self.kwargs)   
            
        else:
            raise ValueError(f'Unknown scheduler: {self.name}')
        scheduler.name = self.name
        return scheduler
