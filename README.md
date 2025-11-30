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

This library contains ```DecVAEs``` and numerous utilities/scripts for pre-training, fine-tuning transfer learning and zero-shot transfer learning of ```DecVAEs```, evaluation of ```DecVAE``` representations through disentanglement and task-specific metrics, and latent analysis tools for interpretability.``` DecVAEs``` are built by adapting the 🤗 Wav2Vec2-encoder architecture to include VAE functionality for disentangled representation learning. We accommodate the analysis of 4 datasets: TIMIT, IEMOCAP, VOC-ALS and SimVowels, a custom simulated speech dataset. Paper figures can be generated using R (```visualize_R```). Implementation is based on ```PyTorch```, 🤗 HuggingFace Transformers and Google Research ```disentanglement_lib```.

## Methodology

![Model Architecture](overview.jpg)

Our method employs a novel variational decomposition autoencoder (DecVAE) architecture with several key innovations:

- **Component-wise Processing**: Parallel encoding pathways for distinct signal components
- **Dual-branch Architecture**: Separate content (Z) and style (S) processing streams
- **Orthogonality Constraints**: Mathematical guarantees of component independence
- **Contrastive Learning Objective**: Self-supervised training through masked prediction
- **Variational Inference**: Probabilistic modeling of component distributions

## Reproducibility

All experiments are fully reproducible with the provided code and configurations. We provide:

- **Complete source code** for all model components and training procedures to reproduce the whole training process and evaluation of DecVAEs (```scripts```) .
- **Configuration files** for reproducing experiments in the paper (```config_files```).
- **Evaluation protocols** for assessing disentanglement and downstream performance.
- **Analysis products** for reproducing figures (```figures```, ```supplementary_figures```) in the paper without the need of pre-training models (```data```).
- **Visualization utilities** for reproducing figures in the paper (```visualize_R``` and ```scripts/visualize```).

### SimVowels data generation
```bash

```

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



## Installation instructions
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

## Citation
If you use this codebase in your research, please cite our paper:

Ziogas I.N., Al Shehhi A., Khandoker A.H., and Hadjileontiadis L.J. Variational decomposition autoencoding improves disentanglement of latent representations.

## References
Wolf T. et al. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020
 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45,
 (EMNLP,2020).

Paszke A. et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural
 information processing systems, 32, (2019).

Shuangbin X., Chen M., Feng T., Zhan L., Zhou L., and Yu G. Use ggbreak to effectively utilize plotting
space to deal with large datasets and outliers. Frontiers in Genetics, 12, (2021).

Locatello F., Bauer S., Lucic M., Raetsch G., Gelly S., Sch¨olkopf B., and Bachem O. Challenging
Common Assumptions in the Unsupervised Learning of Disentangled Representations, (2019).