# Configuration Files Guide

This directory contains JSON configuration files for all DecVAE experiments and use cases. Each configuration file specifies parameters for model architecture, training objectives, data processing, and task-specific settings.

## Directory Structure

```
config_files/
├── DecVAEs/              # DecVAE model configurations
│   ├── sim_vowels/       # Simulated vowels experiments
│       ├── latent_evaluations
│       ├── latent_traversals
│       ├── latent_visualizations
│       └── pre-training
│   ├── timit/            # TIMIT speech corpus experiments
│       ├── ...
│   ├── voc_als/          # VOC ALS dysarthria severity evaluation dataset experiments
│       ├── ...
│   └── iemocap/          # IEMOCAP emotion recognition experiments
│       ├── fine-tuning
│       ├── ...
├── VAEs/                 # Standard VAE configurations
│   ├── sim_vowels/       # Simulated vowels experiments
│   ├── timit/            # TIMIT speech corpus experiments
│   ├── voc_als/          # VOC ALS dysarthria severity evaluation dataset experiments
│   └── iemocap/          # IEMOCAP emotion recognition experiments
├── input_visualizations/ # Input visualization configs for all datasets
└── configuration_decVAE.py  # Base DecVAE configuration class
```
