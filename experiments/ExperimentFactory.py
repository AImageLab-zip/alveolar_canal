from .segmentation import Segmentation
from .generation import Generation

class ExperimentFactory:
    def __init__(self, config):
        self.name = config.experiment.name
        self.config = config

    def get(self):
        if self.name == 'Segmentation':
            experiment = Segmentation(self.config)
        elif self.name == 'Generation':
            experiment = Generation(self.config)
        else:
            raise ValueError(f'Experiment \'{self.name}\' not found')
        return experiment
