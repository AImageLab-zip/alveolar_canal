# Alveolar canal 3D training

3D network for the alveolar canal segmentation directly on volumes.

## Usage

```python
main.py --experiment_path /path/for/your/experiment --base_config config.yaml --verbose
```

## available features
- multi GPU training
- tensorboard
- Unet3D (padded), ...
- BCE, CE loss, DICE loss
- 4 label / 2 labels classification

edit the yaml file to suit your needs.