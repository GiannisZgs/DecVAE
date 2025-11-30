# Variational Decomposition Autoencoders (DecVAE)

**Codebase repository for paper: "Variational decomposition autoencoding improves disentanglement of latent representations"**

Understanding the structure of complex, nonstationary, high-dimensional time-evolving signals is a central
challenge in scientific data analysis. In many domains, such as speech and biomedical signal processing, the abil
ity to learn disentangled and interpretable representations is critical for uncovering latent generative mechanisms.
Traditional approaches to unsupervised representation learning, including variational autoencoders (VAEs), of
ten struggle to capture the temporal and spectral diversity inherent in such data. Here we introduce variational
decomposition autoencoding (VDA), a framework that extends VAEs by incorporating a strong structural bias
toward signal decomposition. VDA is instantiated through variational decomposition autoencoders (DecVAEs),
i.e., encoder-only neural networks that combine a signal decomposition model, a contrastive self-supervised
task, and variational prior approximation to learn multiple latent subspaces aligned with time-frequency char
acteristics.

This library contains ```DecVAEs``` and numerous utilities/scripts for pre-training, fine-tuning transfer learning and zero-shot transfer learning of ```DecVAEs```, evaluation of ```DecVAE``` representations through disentanglement and task-specific metrics, and latent analysis tools for interpretability.``` DecVAEs``` are built by adapting the 🤗 Wav2Vec2-encoder architecture to include VAE functionality for disentangled representation learning. We accommodate the analysis of 4 datasets: TIMIT, IEMOCAP, VOC-ALS and SimVowels, a custom simulated speech dataset. Paper figures can be generated using R (```visualize_R```). Implementation is based on PyTorch, 🤗 HuggingFace Transformers and Google Research ```disentanglement_lib```.

## Overview

![Model Architecture](overview.jpg)

Our method employs a novel variational decomposition autoencoder (DecVAE) architecture with several key innovations:

- **Decomposition of the latent space**: DecVAE assumes a generative process that is expressed through multiple frequency-resonant latent subspaces.
- **Signal decomposition model**: DecVAE utilizes a signal decomposition model that decomposes inputs in C components (C masked views of the input signal), that share information with the initial input, are time-frequency orthogonal to each other and their superposition reconstructs the input signal. This creates positive (initial signal-component) and negative (component-component) pairs. Initial signal and components are propagated through the shared encoder; the latent representation space is forced to retain the orthogonality and reconstruction properties of the input space.  
- **Prior (Gaussian) approximation**: DecVAE retains the prior approximation functionality of beta-VAE-based models.
- **Latent reconstruction and orthogonality**: variational decomposition autoencoding contains two novel disentanglement mechanisms: latent signal reconstruction and orthogonality. In total, DecVAE disentangles through 3 distinct mechanisms (latent reconstruction, latent orthogonality, latent prior approximation). 
- **Novel adversarial divergence objective**: DecVAE is pre-trained with a self-supervised contrastive loss inspired by the signal decomposition dynamics of reconstruction and orthogonality. The loss forces minimal divergences for the positive pairs (initial signal-components) and maximal divergences for the negative pairs (component-component) at the same time.
- **Encoder-only**: The current version of DecVAE is a variational encoder-only disentangling autoencoder; adding generative capabilities is currently a work in progress. The role of the decoder is replaced by the latent reconstruction.
- **Dual-branch Architecture**: Supports simultaneous processing of multiple time scales for slow and fast varying generative factors (e.g. frame and sequence level).

## Installation instructions (Ubuntu 22.04, Windows 11)
```bash
# Clone repository
git clone https://github.com/GiannisZgs/DecVAE.git
cd DecVAE
```

### 1. Python Environment

**Using Conda**
```bash
conda env create -f env_setup/decVAE_conda.yml
conda activate DecVAE
```

**Using pip only**
```bash
pip install -r env_setup/decVAE_pip_requirements.txt
```

*Note: The `.yml` file creates a Conda environment with Python 3.11.9 and installs all packages from `.txt` via pip. If using Conda, you only need the `.yml` file.*

### 2. R Dependencies

Install R packages:
```bash
Rscript env_setup/setup.R
```

## Reproducibility

All experiments are fully reproducible with the provided code and configurations. We provide:

- **Complete source code** for all model components and training procedures to reproduce the whole training process and evaluation of DecVAEs (```scripts```) .
- **Configuration files** for reproducing experiments in the paper (```config_files```).
- **Evaluation protocols** for assessing disentanglement and downstream performance.
- **Analysis products** for reproducing figures (```figures```, ```supplementary_figures```) in the paper without the need of pre-training the models from scratch (```data```).
- **Visualization utilities** for reproducing figures in the paper (```visualize_R``` and ```scripts/visualize```).


### SimVowels data generation
To generate the SimVowels dataset run the below after setting up the environment:

```bash
python scripts/simulations/simulated_vowels.py
```

or download SimVowels directly from https://drive.google.com/drive/folders/1VE4mkC3P1GEDrorThmRgL07NdEoLtyf9?usp=sharing.

### Training Pipeline

#### Pretraining (Self-Supervised Learning)
```bash
accelerate launch scripts/pre-training/base_models_ssl_pretraining_new.py --config_file config_pretraining_timit_NoC3.json
```

#### Fine-tuning
```bash
accelerate launch scripts/fine_tuning/ssl_fine_tune_pretrained_models.py --config_file config_finetune_iemocap_NoC4.json
```

#### Latent Space Analysis
```bash
python scripts/post-training/latents_post_analysis.py --config_file config_latent_anal_timit.json
```

#### Component Visualization
```bash
python scripts/latent_response_analysis/latent_traversal_analysis.py --config_file config_latent_traversals_timit_renderex.json
```


## Citation
If you use this codebase in your research, please cite our paper:

Ziogas I.N., Al Shehhi A., Khandoker A.H., and Hadjileontiadis L.J. Variational decomposition autoencoding improves disentanglement of latent representations. 

and the codebase:

Ziogas I.N., Al Shehhi A., Khandoker A.H., and Hadjileontiadis L.J. Variational decomposition autoencoding improves disentanglement of latent representations. *Zenodo*

## References
Wolf T. et al. Transformers: State-of-the-art natural language processing. *In Proceedings of the 2020
 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 38–45,
 (EMNLP,2020).

Paszke A. et al. Pytorch: An imperative style, high-performance deep learning library. *Advances in neural
 information processing systems*, 32, (2019).

Shuangbin X., Chen M., Feng T., Zhan L., Zhou L., and Yu G. Use ggbreak to effectively utilize plotting
space to deal with large datasets and outliers. *Frontiers in Genetics*, 12, (2021).

Locatello F., Bauer S., Lucic M., Raetsch G., Gelly S., Sch¨olkopf B., and Bachem O. Challenging
Common Assumptions in the Unsupervised Learning of Disentangled Representations, *Proceedings of the 36th
International Conference on Machine Learning*, 97:4114–4124, (PMLR,2019).