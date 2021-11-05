# Alveolar canal 3D training

3D network for the alveolar canal segmentation.

## Usage

```python
main.py --experiment_path /path/for/your/experiment --base_config config.yaml
```
## Optional flags
--verbose redirect stream to std out instead of using a log file in the experiment directory
--competitor load training data as circle expansion instead of dense annotations
NOTE 1: this does not affect the type of model
NOTE 2: this does not affect the type of additional dataset
--test skip the training and load best weights for the experiment
--skip_dump if this flag is set the network does not dump prediction volumes on the final test
  
## available features
- multi GPU training
- tensorboard
- Unet3D (padded), ...
- BCE, CE loss, DICE loss
- 4 label / 2 labels classification

edit the yaml file to suit your needs.