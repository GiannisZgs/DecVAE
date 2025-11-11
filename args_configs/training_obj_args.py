from dataclasses import dataclass, field

@dataclass
class TrainingObjectiveArguments:
    """
    Arguments pertaining to the optimization objective(s) of the model.
    """
    comment_tr_obj_args: str = field(
        default = "No comment",
        metadata={"help": "A comment to add to the training objective arguments."},
    )
    max_frames_per_batch: int = field(
        default=100,
        metadata={"help": "The maximum number of frames per batch for calculating the decomposition loss."},
    )
    decomp_loss_reduction: str = field(
        default="sum",
        metadata={"help": "The reduction method for the decomposition loss."},
    )
    div_pos_weight: float = field(
        default=0.5,
        metadata={"help": "The weight of the positive forces in the decomposition divergence loss."},
    )
    div_neg_weight: float = field(
        default=1.0,
        metadata={"help": "The weight of the negative forces in the decomposition divergence loss."},
    )
    divergence_type: str = field(
        default="js",
        metadata={"help": "The type of divergence to use in the decomposition loss: kl or js."},
    )
    use_prior_regularization: bool = field(
        default=False,
        metadata={"help": "Whether to use prior regularization in the decomposition loss (as in VAE's ELBO)."},
    )
    beta_kl_prior_z: float = field(
        default=1.0,
        metadata={"help": "The weight of the KL divergence term in the z prior regularization - beta = 1 is the vanilla VAE."},
    )
    beta_kl_prior_s: float = field(
        default=1.0,
        metadata={"help": "The weight of the KL divergence term in the s prior regularization - beta = 1 is the vanilla VAE."},
    )
    prior_reg_weighting_z: float = field(
        default=1.0,
        metadata={"help": "The weighting of the z prior regularization term in the decomposition loss."},
    )
    prior_reg_weighting_s: float = field(
        default=1.0,
        metadata={"help": "The weighting of the s prior regularization term in the decomposition loss."},
    )
    clip_grad_value: float = field(
        default=None,
        metadata={"help": "The max allowed norm to clip the gradients to."},
    )
    weight_0_1: float = field(
        default=0.3,    
        metadata={"help": "The weight of the 0(original)-1(1st component) loss in the decomposition loss."},
    )  
    weight_0_2: float = field(
        default=0.3,
        metadata={"help": "The weight of the 0(original)-2(2nd component) loss in the decomposition loss."},
    )
    weight_0_3_and_above: float = field(    
        default=0.3,
        metadata={"help": "The weight of the 0(original)-3(3rd and above components) loss in the decomposition loss."},
    )
    decomp_loss_s_weight: float = field(
        default=1,
        metadata={"help": "The weight of the s latent decomposition loss in the total optimization objective."},
    )
    prior_reg_loss_total_weight: float = field(
        default=0.1,
        metadata={"help": "The total weight of the prior regularization loss in the optimization objective - used in supervised fine-tuning to balance the supervised loss."},
    )
    div_loss_total_weight: float = field(
        default=0.1,
        metadata={"help": "The total weight of the diversity loss in the optimization objective - used in supervised fine-tuning to balance the supervised loss."},
    )
    vae_loss_weight: float = field(
        default=0.2,
        metadata={"help": "In the supervised fine-tuning scenario, the weight of the VAE reconstruction + KL loss in the total optimization objective."},
    )
    supervised_loss_weight: float = field(
        default=0.8,
        metadata={"help": "The total weight of the supervised loss in the optimization objective - used in supervised fine-tuning to balance the unsupervised losses."},
    )
    supervised_loss_type: str = field(
        default="cross_entropy",
        metadata={"help": "The type of supervised loss to use - options are 'cross_entropy', 'focal', 'label_smoothed_ce'."},
    )
    supervised_loss_reduction: str = field(
        default="sum",
        metadata={"help": "The reduction method to use for the supervised loss - options are 'none', 'mean', 'sum'."},
    )
    supervised_loss_rel_weight: float = field(
        default=0.001,
        metadata={"help": "A relative weight to bring the supervised loss to the same scale as the other losses - used in supervised fine-tuning."},
    )