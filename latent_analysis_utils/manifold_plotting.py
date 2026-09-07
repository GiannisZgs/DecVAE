"""
Helpers for the low-dimensional input and latent visualization scripts.

Those scripts build the same manifold-projection plot many times over - once per dataset, per
domain (time or mel), per level (frame or sequence) and per colouring variable. Each occurrence
sets the same visualization flags, constructs a TSNE or UMAP estimator, assembles the same save
path and calls visualize(). These helpers hold that shared shape so a call site only states what
actually differs.
"""
import os

import torch
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .visualization_utils import visualize


def set_vis_flags(data_training_args, vis_method, frequency_vis=False,
                  generative_factors_vis=True, vis_sphere=False, plot_2d_3d='2d'):
    """
    Set the visualization flags that visualize() reads off data_training_args.

    Args:
        data_training_args: Data and training related arguments, modified in place
        vis_method (str): 'tsne' or 'umap'; also selects the sub-directory the plots are saved in
        frequency_vis (bool): Whether the plot is coloured by frequency content
        generative_factors_vis (bool): Whether the plot is coloured by a generative factor
        vis_sphere (bool): Whether to project a Gaussian sample alongside the data
        plot_2d_3d (str): '2d' or '3d'
    """
    data_training_args.frequency_vis = frequency_vis
    data_training_args.generative_factors_vis = generative_factors_vis
    data_training_args.vis_sphere = vis_sphere
    data_training_args.tsne_plot_2d_3d = plot_2d_3d
    data_training_args.vis_method = vis_method


def manifold(vis_method, vis_args, metric='cosine', perplexity=30, n_neighbors=30,
             min_dist=0.2, densmap=False, early_exaggeration=10, n_components=2):
    """
    Build the manifold_dict that visualize() expects, seeded from vis_args.random_seed_vis.

    Args:
        vis_method (str): 'tsne' or 'umap'
        vis_args (:class:`~args_configs.visualization_args.VisualizationsArguments`): Holds the seed
        metric (str): Distance metric for the estimator
        perplexity (int): TSNE perplexity - sequence-level plots use a small value, since there are
            far fewer sequences than frames
        n_neighbors (int): UMAP neighbourhood size
        min_dist (float): UMAP minimum distance
        densmap (bool): Whether UMAP runs in densMAP mode
        early_exaggeration (int): TSNE early exaggeration
        n_components (int): Dimensionality of the projection, 2 or 3
    Returns:
        manifold_dict (dict): Single-entry mapping of the method name to its estimator
    """
    if vis_method == 'tsne':
        return {'tsne': TSNE(n_components=n_components, random_state=vis_args.random_seed_vis,
                             learning_rate='auto', max_iter=1000, perplexity=perplexity,
                             metric=metric, early_exaggeration=early_exaggeration, init='pca')}
    return {'umap': umap.UMAP(n_components=n_components, random_state=vis_args.random_seed_vis,
                              metric=metric, n_neighbors=n_neighbors, min_dist=min_dist,
                              densmap=densmap)}


def plot_manifold(data_training_args, vis_args, decomp_args, config, X, OCs, y_vec, target,
                  data_set, manifold_dict, domain, group, z_or_h='z'):
    """
    Project and plot one set of representations, saving under the layout the visualization scripts
    share: <save_vis_dir>/<decomposition>/low_input_dim/<set>/<method>/<domain>/<dataset>/<group>.

    Args:
        data_training_args: Data and training related arguments; the flags must already be set
        vis_args (:class:`~args_configs.visualization_args.VisualizationsArguments`): Visualization arguments
        decomp_args (:class:`~args_configs.decomp_args.DecompositionArguments`): Decomposition arguments
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): The DecVAE configuration object
        X: Representations of the anchor subspace, usually the original signal
        OCs: Representations of the decomposed subspaces, or None
        y_vec: Target labels used to colour the plot
        target (str): Name of the coloured variable
        data_set (str): Identifier appended to the saved file names
        manifold_dict (dict): Estimator mapping, as returned by manifold()
        domain (str): e.g. 'time_domain_frame' or 'mel_domain_seq'
        group (str or None): Sub-directory for the coloured variable, e.g. 'vowels' or 'speakers'.
            Some sequence-level plots save directly under the dataset directory and pass None.
        z_or_h (str): Which representation is being plotted
    """
    save_dir = os.path.join(vis_args.save_vis_dir, decomp_args.decomp_to_perform, 'low_input_dim',
                            vis_args.set_to_use_for_vis, data_training_args.vis_method, domain,
                            data_training_args.dataset_name)
    if group is not None:
        save_dir = os.path.join(save_dir, group)

    visualize(
        data_training_args,
        config,
        X=X,
        OCs=OCs,
        z_or_h=z_or_h,
        y_vec=y_vec,
        target=target,
        data_set=data_set,
        manifold_dict=manifold_dict,
        save_dir=save_dir,
    )


def pca_reduce_frames(anchor, components, n_components, num_components, domain_label,
                      oc_label='', random_state=0):
    """
    Reduce frame-level representations with PCA: the anchor signal, each decomposition component,
    and the components concatenated together. Reports the explained variance of every fit with the
    same messages the visualization scripts printed inline.

    Frame-level only. The components are expected as (NoC, samples, features) and the concatenation
    reshapes on ``shape[1]``. The sequence-level blocks in those scripts index their components on a
    different axis, and not consistently between blocks, so they are deliberately left as they are
    rather than forced through here.

    Args:
        anchor: Representations of the original signal, shape (samples, features)
        components: Representations of the decomposition components, shape (NoC, samples, features)
        n_components (int): Number of principal components to keep
        num_components (int): How many decomposition components to reduce, i.e. config.NoC
        domain_label (str): Domain name in the messages, 'time domain' or 'mel domain'
        oc_label (str): Prefix on the per-component message; the mel blocks print 'mel OC 1 ...'
            where the time-domain blocks print 'OC 1 ...'
        random_state (int): Seed for every PCA fit
    Returns:
        anchor_reduced, components_reduced, concat_reduced (torch.Tensor): The reduced tensors,
        with the components stacked along the first axis
    """
    pca_anchor = PCA(n_components=n_components, random_state=random_state)
    anchor_reduced = torch.tensor(pca_anchor.fit_transform(anchor))
    explained_var_orig = sum(pca_anchor.explained_variance_ratio_) * 100
    print(f"Explained variance for {domain_label} original frame PCA: {explained_var_orig:.2f}%")

    components_reduced = []
    for oc in range(num_components):
        pca_OC = PCA(n_components=n_components, random_state=random_state)
        components_reduced.append(torch.tensor(pca_OC.fit_transform(components[oc])))
        explained_var = sum(pca_OC.explained_variance_ratio_) * 100
        print(f"Explained variance for {oc_label}OC {oc+1} frame PCA: {explained_var:.2f}%")
    components_reduced = torch.stack(components_reduced, dim=0)

    concat = components.transpose(0, 1).reshape(components.shape[1], -1)
    pca_concat = PCA(n_components=n_components, random_state=random_state)
    concat_reduced = torch.tensor(pca_concat.fit_transform(concat))
    explained_var_OCs = sum(pca_concat.explained_variance_ratio_) * 100
    print(f"Explained variance for {domain_label} OCs_concat frame PCA: {explained_var_OCs:.2f}%")

    return anchor_reduced, components_reduced, concat_reduced


def plot_manifold_latents(data_training_args, vis_args, variant, betas, data_subset, config,
                          X, OCs, y_vec, target, data_set, manifold_dict, embedding, group,
                          z_or_h='z'):
    """
    Project and plot one set of latent representations, saving under the layout the latent
    visualization scripts share:
    <save_vis_dir>/<variant>/<dataset>/<betas>/<subset>/<embedding>/<group>/<method>.

    Note the path differs from the input-visualization one: it carries the beta setting and the
    data subset, and the method comes last rather than in the middle.

    Args:
        data_training_args: Data and training related arguments; the flags must already be set
        vis_args (:class:`~args_configs.visualization_args.VisualizationsArguments`): Visualization arguments
        variant (str): The run variant the plots are filed under. The DecVAE scripts pass the
            decomposition method; the VAE scripts pass the model type and its input type joined.
        betas (str): The beta setting the checkpoint was trained with, as used in the save path
        data_subset (str): Which split the representations came from
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): The DecVAE configuration object
        X: Representations of the anchor subspace
        OCs: Representations of the decomposed subspaces, or None
        y_vec: Target labels used to colour the plot
        target (str): Name of the coloured variable
        data_set (str): Identifier appended to the saved file names
        manifold_dict (dict): Estimator mapping, as returned by manifold()
        embedding (str): Which embedding is plotted, e.g. 'X_OCs' or 'all_joint_emb'
        group (str): Sub-directory for the coloured variable, e.g. 'speakers' or 'speakers_seq'
        z_or_h (str): Which representation is being plotted
    """
    visualize(
        data_training_args,
        config,
        X=X,
        OCs=OCs,
        z_or_h=z_or_h,
        y_vec=y_vec,
        target=target,
        data_set=data_set,
        manifold_dict=manifold_dict,
        save_dir=os.path.join(vis_args.save_vis_dir, variant,
                              data_training_args.dataset_name, betas, data_subset,
                              embedding, group, data_training_args.vis_method),
    )


"""Aggregation strategies, in the order the latent visualization scripts emit them.

Each entry is (strategy name, save sub-directory, whether the strategy resets frequency_vis).
X_OCs_freq deliberately does not reset it: it inherits the flag the caller set, which is how the
frequency-coloured plots are produced. The other three set it to False before plotting.
"""
AGGREGATIONS = (
    ('X_OCs_freq', 'X_OCs', False),
    ('OCs_joint', 'OCs_joint_emb', True),
    ('OCs_proj', 'OCs_projection', True),
    ('all', 'all_joint_emb', True),
)


def plot_aggregations(data_training_args, vis_args, variant, betas, data_subset, config,
                      strategies, sources, y_vec, target, data_set, manifold_dict, group,
                      project_guarded=True):
    """
    Plot one colouring variable across every requested aggregation strategy.

    Replaces the four consecutive ``if "<strategy>" in vis_args.aggregation_strategies_to_plot_*``
    blocks that the latent visualization scripts repeat for each variable, method and level.

    Two details of the original are preserved deliberately. ``frequency_vis`` is reset before the
    ``config.project_OCs`` test, so it is reset even when the projection plot itself is skipped; and
    a strategy absent from ``sources`` produces nothing at all, matching a script that simply had no
    block for it.

    Args:
        data_training_args: Data and training related arguments; the flags must already be set
        vis_args (:class:`~args_configs.visualization_args.VisualizationsArguments`): Visualization arguments
        variant (str): The run variant the plots are filed under
        betas (str): The beta setting the checkpoint was trained with
        data_subset (str): Which split the representations came from
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): The DecVAE configuration object
        strategies: The configured strategies, i.e. vis_args.aggregation_strategies_to_plot_frame or _seq
        sources (dict): Strategy name -> (X, OCs) for this context. A strategy the caller has no
            representations for is simply left out.
        y_vec: Target labels used to colour the plots
        target (str): Name of the coloured variable
        data_set (str): Identifier appended to the saved file names
        manifold_dict (dict): Estimator mapping, as returned by manifold()
        group (str): Sub-directory for the coloured variable
        project_guarded (bool): Whether the OCs_proj plot is conditional on config.project_OCs.
            Most call sites in the scripts guard it, but a number plot it unconditionally, so the
            caller states which it is rather than the guard being assumed.
    """
    for name, embedding, resets_frequency in AGGREGATIONS:
        if name not in strategies or name not in sources:
            continue
        if resets_frequency:
            data_training_args.frequency_vis = False
        if name == 'OCs_proj' and project_guarded and not config.project_OCs:
            continue
        X, OCs = sources[name]
        plot_manifold_latents(data_training_args, vis_args, variant, betas, data_subset, config,
                              X=X, OCs=OCs, y_vec=y_vec, target=target, data_set=data_set,
                              manifold_dict=manifold_dict, embedding=embedding, group=group)
