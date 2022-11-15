from .segmentation import Segmentation
from .generation import Generation

class ExperimentFactory:
    def __init__(self, config, debug=False):
        self.name = config.experiment.name
        self.config = config

    def get(self):
        if self.name == 'Segmentation':
            experiment = Segmentation(self.config, debug)
        elif self.name == 'Generation':
            experiment = Generation(self.config, debug)
        else:
            raise ValueError(f'Experiment \'{self.name}\' not found')
        return experiment
