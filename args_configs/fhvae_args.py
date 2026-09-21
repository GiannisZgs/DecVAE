# coding=utf-8
# Copyright 2025 Ioannis Ziogas <ziogioan@ieee.org>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Arguments for the FHVAE baseline, the Factorized Hierarchical VAE of Hsu, Zhang and Glass
(NeurIPS 2017), ported from https://github.com/wnhsu/FactorizedHierarchicalVAE.

FHVAE splits an utterance into segments and infers two latents per segment: a sequence latent whose
prior mean is a trainable per-utterance vector, and a segment latent conditioned on it. What ties an
utterance together is that prior mean, not an aggregator, which is where FHVAE differs from DecVAE's
sequence branch.

The reference swaps the paper's indices, so every citation has to be translated:

    paper z1, the segment latent   -> reference z2, qz2_x   -> here z_seg
    paper z2, the sequence latent  -> reference z1, qz1_x   -> here z_seq
    paper mu2, the per-utterance   -> reference mu1_table   -> here mu_seq
        prior mean

Where this port departs from the reference, and why:

    segments  The reference cuts 20 frames on Kaldi's 10 ms hop, so 200 ms. DecVAE's hop is 20 ms,
              so fhvae_seg_len is 10 frames for the same 200 ms. This matches the reference in
              time rather than in frame count, as CPC's prediction horizon does.
    batching  The reference draws 256 random segments from across the corpus. The shared collator
              batches whole utterances, so the segments are cut inside the forward pass and a batch
              holds every segment of a few utterances. The objective is unchanged; only the
              gradient variance differs.
    latents   d_seg and d_seq are DecVAE's latent width rather than the reference's 32 each, so the
              embeddings being compared have the same size, as CoST's trend and season do.
    readout   The reference dumps the posterior mean and logvar for its ASR probes. Only the means
              are used here, which is what DecVAE's evaluation reads.

The reference training recipe, which the shipped configs carry in the data arguments:
Adam with betas (0.95, 0.999), learning rate 1e-3 and no schedule - the config defines
lr_decay_factor but the runner never applies it - and no gradient clipping. Its L2 of 1e-4 reaches
only fully connected hidden layers, of which the released configuration has none, so it is inert.
"""

from dataclasses import dataclass, field


@dataclass
class FHVAEArguments:
    comment_fhvae_args: str = field(
        default="FHVAE Baseline Arguments",
        metadata={"help": "A comment to add to the FHVAE arguments."},
    )
    fhvae_input_type: str = field(
        default="mel",
        metadata={"help": "What each frame holds: 'mel' takes the log-mel features extracted per "
                          "frame at preprocessing, 'waveform' takes the frame samples themselves. "
                          "'mel' is the default and is the same cached tensor DecVAE's encoder "
                          "reads; the reference's own front-end is an 80-band filterbank over a "
                          "25 ms window, which is the precedent."},
    )
    fhvae_n_mels: int = field(
        default=80,
        metadata={"help": "Number of mel bands, for the 'mel' input type."},
    )
    fhvae_mel_norm: str = field(
        default="global",
        metadata={"help": "Mel normalization applied in the collator, for the 'mel' input type. The "
                          "reference applies global mean-variance normalization, which this is; do "
                          "not stack a second one on top."},
    )
    fhvae_pool_mel_bins: bool = field(
        default=False,
        metadata={"help": "Average the time bins preprocessing extracted inside each frame, leaving "
                          "one value per mel channel."},
    )
    fhvae_seg_len: int = field(
        default=10,
        metadata={"help": "Frames per segment, the unit FHVAE infers a latent pair for. The "
                          "reference cuts 20 frames on a 10 ms hop, so 200 ms; 10 frames on "
                          "DecVAE's 20 ms hop is the same 200 ms. Training draws segment starts at "
                          "random, as the reference's seg_rand does, and evaluation cuts them "
                          "non-overlapping."},
    )
    fhvae_hidden: int = field(
        default=256,
        metadata={"help": "Hidden width of the two encoders and the decoder, each a single "
                          "unidirectional LSTM layer, as the reference's lstm_1L_256 config sets."},
    )
    fhvae_d_seg: int = field(
        default=32,
        metadata={"help": "Width of the segment latent, the paper's z1 and the reference's z2. The "
                          "reference uses 32; the configs raise it to DecVAE's latent width so the "
                          "embeddings being compared have the same size."},
    )
    fhvae_d_seq: int = field(
        default=32,
        metadata={"help": "Width of the sequence latent, the paper's z2 and the reference's z1. "
                          "Kept equal to fhvae_d_seg, as the reference does."},
    )
    fhvae_seq_std: float = field(
        default=0.5,
        metadata={"help": "Standard deviation of p(z_seq | mu_seq), the reference's latent1_std. "
                          "Its square is also what the closed-form estimate of mu_seq divides by."},
    )
    fhvae_alpha: float = field(
        default=10.0,
        metadata={"help": "Weight of the discriminative term, the reference's alpha_dis, which the "
                          "TIMIT run script sets to 10. The term is a cross-entropy over the "
                          "training utterances, scored by the distance from each utterance's prior "
                          "mean. 0 trains the plain lower bound."},
    )
    fhvae_optimizer: str = field(
        default="adam",
        metadata={"help": "Optimizer family, taken from the reference implementation: 'adam' with "
                          "the betas of the data arguments, which the config sets to (0.95, 0.999). "
                          "'adamw' uses the optimizer the rest of the project trains with, and is "
                          "an ablation rather than the baseline."},
    )
    fhvae_representation: str = field(
        default="both",
        metadata={"help": "Which vector the evaluation reads per frame: 'both' concatenates the "
                          "segment and the sequence latent, which is FHVAE's full latent state and "
                          "parallels DecVAE's Z and CoST's trend with season; 'seg' is the segment "
                          "latent alone, the subspace FHVAE deliberately empties of speaker; 'seq' "
                          "is the sequence latent alone. Every variant is returned whichever is "
                          "chosen, so the choice only names the headline."},
    )
    fhvae_seq_pooling: str = field(
        default="mu",
        metadata={"help": "How the frames of an utterance are reduced to one sequence-level "
                          "embedding. 'mu' is the reference's s-vector, the closed-form estimate "
                          "sum E[z_seq] / (N + fhvae_seq_std^2) over the non-overlapping segments, "
                          "which is mean pooling of z_seq up to that factor. 'mean' averages the "
                          "frame embeddings as the other baselines do. None skips them."},
    )
    fhvae_transfer_reinit_mu_table: bool = field(
        default=True,
        metadata={"help": "Fine-tuning only. The table of per-utterance prior means has one row per "
                          "training utterance of the source corpus, so a target corpus cannot share "
                          "it. On, everything else transfers and the table starts over, which costs "
                          "nothing because the evaluation reads the encoders and estimates the means "
                          "in closed form. Off, the transfer stops instead."},
    )
