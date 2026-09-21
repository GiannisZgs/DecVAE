# models package initialization
from .dec_vae import DecVAEForPreTraining, DecVAEForSupervisedFineTuning, Dec2VecModel
from .autoencoders import VAE_1D, VAE_1D_FC, VAE_1D_ForSupervisedFineTuning, VAE_1D_FC_ForSupervisedFineTuning
from .decomposition_masking import DecompositionModule, CustomBatchNorm, CustomLayerNorm

from .baselines import FrozenSSLEncoder, build_frozen_ssl, FROZEN_SSL_CHECKPOINTS, FrameGeometry
from .baselines import CoSTEncoder, CoSTForPreTraining, build_cost
from .baselines import TFC, TFCForPreTraining, build_tfc
from .baselines import TCLNetwork, TCLForPreTraining, build_tcl
from .baselines import CPCEncoder, CPCForPreTraining, build_cpc
