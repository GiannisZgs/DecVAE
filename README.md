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

![Model Architecture](img/overview.jpg)

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
- **Examples** for various use cases in ```examples/notebooks```.

## Example: Disentanglement of simulated speech data (SimVowels dataset) 

### SimVowels data generation
To generate the SimVowels dataset run the below after setting up the environment:

```bash
python scripts/simulations/simulated_vowels.py
```

or download SimVowels directly from https://drive.google.com/drive/folders/1VE4mkC3P1GEDrorThmRgL07NdEoLtyf9?usp=sharing.

### Input visualization
To visualize the inputs (frame-level) to the models:

```bash
accelerate launch scripts/visualize/low_dim_vis_input.py --config_file config_files/input_visualizations/config_visualizing_input_frames_vowels.json
```

To visualize the inputs (sequence-level) to the models:

```bash
accelerate launch scripts/visualize/low_dim_vis_input.py --config_file config_files/input_visualizations/config_visualizing_input_sequences_vowels.json
```

For more details on setting crucial input visualization parameters, see ```config_files/README```.

### Pre-training of a DecVAE
Use the ```accelerate``` library: 

For single-GPU training:
```bash
accelerate launch scripts/pre-training/base_models_ssl_pretraining.py --config_file config_files/DecVAEs/sim_vowels/pre-training/config_pretraining_sim_vowels_NoC3.json
```

For multi-GPU training or to specify a GPU id:
```bash
accelerate launch --gpu_ids 0,1 scripts/pre-training/base_models_ssl_pretraining.py --config_file config_files/DecVAEs/sim_vowels/pre-training/config_pretraining_sim_vowels_NoC3.json
```
For more details on setting crucial DecVAE parameters for the pre-training, see ```config_files/README```.

### Latent evaluation - Disentanglement and task-specific
```bash
accelerate launch scripts/post-training/latents_post_analysis.py --config_file config_files/DecVAEs/sim_vowels/latent_evaluations/config_latent_anal_sim_vowels.json
```
For more details on setting crucial DecVAE parameters for the evaluation, see ```config_files/README```.

### Latent visualization
To visualize the latent representations (frame-level):

```bash
accelerate launch scripts/visualize/low_dim_vis_latents.py --config_file config_files/DecVAEs/sim_vowels/latent_visualizations/config_latent_frames_visualization_vowels.json
```
To visualize the latent representations (sequence-level):

```bash
accelerate launch scripts/visualize/low_dim_vis_latents.py --config_file config_files/DecVAEs/sim_vowels/latent_visualizations/config_latent_sequences_visualization_vowels.json
```
For more details on setting crucial latent visualization parameters, see ```config_files/README```.

### Latent traversals/response analysis
To perform latent traversal analysis for the SimVowels dataset, first generate a small-scale 
set where generative factors are controlled:

```bash
python scripts/simulations/simulated_vowels_for_latent_traversal.py
```

Then use pre-trained DecVAE models as a representation function to obtain the latent responses:

```bash
accelerate launch scripts/latent_response_analysis/latent_traversal_analysis.py --config_file config_files/DecVAEs/sim_vowels/latent_traversals/config_latent_traversals_sim_vowels.json
```
For more details on setting crucial latent traversals parameters, see ```config_files/README```.

### Benchmarks and more datasets
The exact same process as above can be followed to pre-train and evaluate VAE-based models and ICA/PCA/kPCA, as well as DecVAE models for the other 3 supported datasets (TIMIT, VOC-ALS, IEMOCAP). The exact same scripts and config_files exist for VAEs. For details and parameters check ```config_files/README```.

## Citation
If you use this codebase in your research, please cite our paper and this codebase:

Ziogas I.N., Al Shehhi A., Khandoker A.H., and Hadjileontiadis L.J. Variational decomposition autoencoding improves disentanglement of latent representations. 

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