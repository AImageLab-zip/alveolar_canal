import torch
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

class SchedulerFactory():
    def __init__(self, name, optimizer, **kwargs):
        super(SchedulerFactory, self).__init__()
        self.name = name
        self.optimizer = optimizer
        self.kwargs = kwargs

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
        else:
            raise ValueError(f'Unknown scheduler: {self.name}')
        scheduler.name = self.name
        return scheduler


