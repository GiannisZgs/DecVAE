# Variational Decomposition Autoencoders (DecVAE)

**Codebase repository for the publication: "Variational decomposition autoencoding improves disentanglement of latent representations"**

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

This library contains ```DecVAEs and numerous utilities/scripts for pre-training, fine-tuning transfer learning, zero-shot evaluation transfer learning of DecVAEs, and evaluation of ```DecVAE representations through disentanglement and task-specific metrics.``` DecVAEs are built by adapting the 🤗 Wav2Vec2-encoder architecture to include VAE functionality for disentangled representation learning. Paper figures can be generated using R (```visualize_R)

## Methodology

![Model Architecture](placeholder_for_architecture_diagram.png)

Our method employs a novel variational decomposition autoencoder (DecVAE) architecture with several key innovations:

- **Component-wise Processing**: Parallel encoding pathways for distinct signal components
- **Dual-branch Architecture**: Separate content (Z) and style (S) processing streams
- **Orthogonality Constraints**: Mathematical guarantees of component independence
- **Contrastive Learning Objective**: Self-supervised training through masked prediction
- **Variational Inference**: Probabilistic modeling of component distributions

The architecture incorporates transformer-based processing with specialized modules for component extraction and integration, enabling robust decomposition across various signal types.

## Reproducibility

All experiments are fully reproducible with the provided code and configurations. We provide:

- **Complete source code** for all model components and training procedures
- **Configuration files** for reproducing all experiments in the paper
- **Pre-processing pipelines** for all datasets used in our evaluations
- **Evaluation protocols** for assessing disentanglement and downstream performance


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

## Citation
If you use this code in your research, please cite our paper:

I.N., Ziogas and A., Al Shehhi and A.H., Khandoker and L.J., Hadjileontiadis. Variational decomposition autoencoding improves disentanglement of latent representations.

```bibtex
@article{author2025variational,
  title={Variational Decomposition Autoencoding: A Novel Approach to Learning Disentangled Representations from Complex Signals},
  author={Author, A. and Author, B. and Author, C.},
  journal={Nature},
  volume={X},
  number={X},
  pages={XXX--XXX},
  year={2025},
  publisher={Nature Publishing Group}
}
```

## References