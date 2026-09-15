# Response to Reviewers

**Manuscript:** "Variational decomposition autoencoding improves disentanglement of latent representations"
**Journal:** npj Artificial Intelligence
**Submission ID:** c86c5415-7be3-43e1-8a57-9b3e782755ae
**Authors:** I. N. Ziogas, A. Al Shehhi, A. H. Khandoker, L. J. Hadjileontiadis

> **WORKING DRAFT.** Scaffold only. `[TODO]` marks text not yet written. `[PENDING RESULT]` marks
> text that cannot be finalised until an experiment completes. Status tags are for our use and are
> removed before submission. Convert to .docx at the end.

---

## How this letter is organised

Twelve of the thirty-five comments raise one of five shared issues. To avoid repetition and to make
the substance easier to check, those are answered once, in full, as **Master Responses (MR-A to
MR-E)** below. Individual comments then give the reviewer-specific answer and point to the relevant
Master Response. Reviewers reading only their own section will find a complete answer there; the
Master Responses are self-contained.

| Master Response | Subject | Comments covered |
|---|---|---|
| **MR-A** | What DecVAE is: no decoder, no posterior sampling, terminology | R1 #1, R5 #1, R6 #1 (part) |
| **MR-B** | The objective: what DELBO is and is not; signs and conventions | R1 #3, R4 #4, R4 #5, R5 #14, R6 #1 (part), R6 #3 |
| **MR-C** | The probabilistic formulation: Eqs. (4b), (5), (7) | R5 #3, R5 #13, R6 #1 (part) |
| **MR-D** | Extended baselines | R1 #5, R3 #1, R5 #2 |
| **MR-E** | Scope of the decomposition prior; number of components; subspace semantics | R3 #2, R4 #1, R4 #2, R4 #3, R5 #6, R5 #8 |

---

## Status tracker

| # | Comment | Type | Depends on | Status |
|---|---|---|---|---|
| R1 #1 | Autoencoder naming | Text | MR-A | Drafting |
| R1 #2 | Compactness trade-off in Discussion | Text | | Not started |
| R1 #3 | Intuition for adversarial stabilisation | Text (+ figure) | MR-B | Not started |
| R1 #4 | Notation mapping main vs SI | Text | | Not started |
| R1 #5 | Sequence/contrastive baselines | Experiment | MR-D | Planned |
| R1 #6 | β sensitivity vs orthogonality | Analysis | | Not started |
| R2 | Editorial polish, figure captions | Text | | Not started |
| R3 #1 | TCL and CoST baselines | Experiment | MR-D | Planned |
| R3 #2 | FD fixed frequency boundaries | Text (+ analysis) | MR-E | Not started |
| R3 #3 | Unresolved citation, line 499 | Text | | Resolved in current draft |
| R3 #4 | Missing space, line 508 | Text | | Resolved, sweep pending |
| R4 #1 | Frequency decomposition is a bias, not a guarantee | Text | MR-E | Not started |
| R4 #2 | Number of components under-justified | Experiment | MR-E | Not started |
| R4 #3 | Subspace-wise evidence | Experiment | MR-E | Not started |
| R4 #4 | Meaning of orthogonality | Text | MR-B | Not started |
| R4 #5 | Is DELBO a strict bound | Text | MR-B | Drafting |
| R4 #6 | Malformed algorithm reference | Text | | Resolved in current draft |
| R5 #1 | Encoder-only naming | Text | MR-A | Drafting |
| R5 #2 | Stronger contemporary baselines | Experiment | MR-D | Planned |
| R5 #3 | "Directly into the generative process" | Text | MR-C | Not started |
| R5 #4 | All experiments are speech | Decision needed | | Not started |
| R5 #5 | "Chromagram-guided dynamics" | Text | | Ready (delete) |
| R5 #6 | SimVowels favours the prior | Experiment | MR-E | Not started |
| R5 #7 | Limitations of disentanglement metrics | Analysis | | Needs clarification |
| R5 #8 | Selection or misspecification of C | Experiment | MR-E | Not started |
| R5 #9 | VOC-ALS confounds | Experiment | | Not started |
| R5 #10 | IEMOCAP LOSO and class-wise results | Analysis | | Not started |
| R5 #11 | Interpretability quantification, seed stability | Analysis | | Not started |
| R5 #12 | Non-variational ablation | Point to results | | Results exist (DecAE) |
| R5 #13 | Eqs. (4b) and (5) | Text | MR-C | Not started |
| R5 #14 | DELBO sign changes and bound status | Text | MR-B | Drafting |
| R5 #15 | Additional citations | Text | | Assessment pending |
| R6 #1 | Equations and terminology | Text | MR-A, MR-B, MR-C | Drafting |
| R6 #2 | Padding and random concatenation in Alg. 1 | Text | | Not started |
| R6 #3 | Bound status and sign conventions | Text | MR-B | Drafting |

---

# Letter to the Editor

Dear Dr. Wang, dear Dr. Xiong,

We thank you and the six reviewers for an unusually detailed and constructive assessment of our
manuscript. The reviews have improved the work substantially, in particular by pushing us to state
precisely what our objective does and does not guarantee.

Your summary identified four areas. We address each below and describe what changed.

**1. Conceptual and mathematical framing.** We agree with the reviewers, and we have made a
substantive correction rather than a defensive clarification. On re-examination we found that the
decomposition evidence lower bound cannot be both a valid lower bound and an orthogonality-promoting
objective, for a structural reason set out in MR-B. We therefore no longer claim that the trained
objective is an evidence lower bound. The variational derivation is retained, correctly scoped, as a
motivation for the form of the objective, and the trained loss is now presented for what it is. We
have also corrected the sign conventions so that the manuscript, the supplement and the released code
agree. We have kept the name DecVAE but now define the sense in which the model autoencodes and the
sense in which it is variational (MR-A).

**2. Empirical support.** We have added [N] baselines spanning time-contrastive learning,
decomposition-based representation learning, self-supervised speech representation learning and
disentangled sequential variational models, evaluated under a protocol matched to our own (MR-D). We
have added sensitivity analyses for the decomposition method, the number of components and the
balancing hyperparameter. `[PENDING RESULT]`

**3. Disentanglement and interpretability claims.** We have added subspace-wise evidence in the form
of `[TODO: factor prediction / ablation / swapping / seed and fold stability]`, and we have narrowed
the claims where the evidence does not support their previous breadth. `[PENDING RESULT]`

**4. Presentation.** We have unified notation between the main text and the supplement, added a
notation mapping table, resolved all cross-references, added the requested citations and carried out
a full proofreading pass.

A point-by-point response follows. Changes in the manuscript are marked in `[TODO: colour / tracked
changes]`.

Yours sincerely,
Ioannis N. Ziogas, on behalf of all authors

---

# Summary of major changes

1. The trained objective is no longer described as an evidence lower bound. It is presented as the
   decomposition objective, and Supplementary Section 5 is retitled and rescoped as a variational
   motivation with an explicit statement of what is and is not bounded. (MR-B)
2. Sign conventions unified across main Eq. (1), Eq. (9), Supplementary Eq. (12), Supplementary
   Alg. 2 and Alg. 1, and made consistent with the released code. Separate symbols now distinguish
   the divergence *measures* from the *losses* built on them. (MR-B)
3. A new definitional paragraph states that DecVAE performs autoencoding at the level of latent
   distributions rather than input points, and that its variational component is an approximating
   Gaussian family with a rate term, without posterior sampling. (MR-A)
4. The probabilistic formulation in Eqs. (4b), (5) and (7) has been restated. (MR-C)
5. `[N]` additional baselines added on SimVowels and TIMIT, with `[N]` carried forward to IEMOCAP.
   (MR-D) `[PENDING RESULT]`
6. New sensitivity analyses: number of components including misspecification, decomposition method,
   and β with respect to both orthogonality and downstream accuracy. `[PENDING RESULT]`
7. New subspace-wise analyses supporting the interpretability claims. `[PENDING RESULT]`
8. Claims about generalisation across modalities narrowed to speech-like signals, or supported by
   `[TODO: decision on R5 #4]`.
9. Notation mapping table between main text and supplement; all unresolved references fixed;
   "chromagram-guided dynamics" removed; additional citations added.

---

# Master Responses

## MR-A. What DecVAE is, and the terminology

*Addresses R1 #1, R5 #1, R6 #1 (terminology).*

We accept the criticism. Two properties of our model were asserted rather than defined, and we now
define both.

**No decoder.** DecVAE has no decoder and performs no point-level reconstruction in the input space.
What it does perform is a reconstruction in the space of latent distributions: the term L_recon
aligns the latent distribution of each decomposed component to the latent distribution of the
original signal, which is the role a decoder plays in a conventional autoencoder, relocated from X to
H. Following Reviewer 1's first suggestion, we now label this explicitly as *latent, distribution-level
autoencoding* at first use in the Introduction and again at the start of Methods, rather than leaving
the reader to infer it.

We note for completeness that we have deliberately not defended the term by appeal to the
autoencoder framing of PCA and ICA. That framing rests on a linear encoder paired with a decoder and
an input-space reconstruction loss, as indeed our own Supplementary Eq. (16) states, so it does not
support decoder-free autoencoding.

**No posterior sampling.** We also make explicit a property that the previous version left implicit
and that bears on Reviewer 4's and Reviewer 6's questions about the objective. DecVAE does not sample
from the approximate posterior. The encoder produces per-subspace Gaussian parameters, the
Kullback-Leibler term to the prior is computed analytically from those parameters, and the means are
used as the representation. The variational component of the model is therefore the approximating
Gaussian family and the rate term acting on it, in the sense of the rate-distortion view of
variational autoencoders, with the usual input-space distortion replaced by the latent-space
decomposition terms. It is not variational inference over the posterior of an instantiated generative
model, and we no longer describe it as such.

**On the name.** We have retained DecVAE and VDA. `[TODO: one or two sentences of justification.
Points available: R1 explicitly offers relabelling as an alternative to renaming; the name is
established through the preprint and the public codebase; the model does autoencode and is
variational in the senses defined above, now stated rather than assumed.]`

**Changes to the manuscript:** `[TODO: locations]`

---

## MR-B. The objective: what DELBO is and is not

*Addresses R1 #3, R4 #4, R4 #5, R5 #14, R6 #1 (Eqs. 7 and 9), R6 #3.*

Reviewers 4, 5 and 6 all question whether the trained objective is a lower bound on any evidence. We
examined this carefully and we agree that it is not. We set out the reason, because it is structural
and we think it is worth stating rather than conceding vaguely.

**Why no repair is possible.** In Supplementary Eq. (10) the orthogonality term enters the bound by
subtraction, and the inequality is preserved precisely because the subtracted quantity is
non-negative. But subtraction means that maximising the bound *minimises* the pairwise divergence
between components, which drives the components together. That is the opposite of the orthogonality
we intend. Rewarding divergence would require adding the term, which destroys the inequality. The
lower-bound property and the orthogonality objective are therefore in direct conflict, and the sign
change between Supplementary Eq. (11) and Eq. (12) is exactly the point at which the objective ceases
to be a bound. We had previously described that change as a consequence of the cross-entropy
reformulation; it is more than that, and we now say so.

We also withdraw the analogy to FactorVAE, which we had used to motivate the insertion of the
orthogonality term. FactorVAE subtracts a total correlation term that it seeks to *minimise*, so
subtraction is consistent with both the bound and the goal. Our case is the mirror image, and the
analogy does not carry.

**Two further points of honesty.** First, even the retained ELBO structure was not operative in
training: as stated in MR-A, no sample is ever drawn from the approximate posterior, so the
expectation in Supplementary Eq. (9) is never estimated. Second, the object bounded in Supplementary
Section 5 is the likelihood of the encoder's own outputs h, not of the data x. Because h is produced
by the encoder being optimised, this is not an evidence in the sense the term normally carries.

**What we now claim instead.** `[TODO: the positive content. Planned: (i) a proposition stating what
Supplementary Eq. (11) does validly bound, namely the collective log-likelihood of the subspace
representations for a fixed encoder and an instantiated latent generative model; (ii) an explicit
statement that the implemented objective is a non-variational surrogate; (iii) a characterisation of
the equilibrium of the trained objective, since the alignment and separation terms cannot be
simultaneously satisfied, connecting to the SVD and PCA argument in Supplementary Section 7.]`

**Naming.** The objective is renamed `[TODO: decide. Options: keep DELBO but define it as
"decomposition objective"; or L_dec. "DELBO-inspired" is dropped either way, since it implies the
bound survives.]`

**Signs and conventions (R6 #3).** We thank Reviewer 6 for catching this. Three separate
inconsistencies existed and all are corrected.

- The symbol L_ortho denoted two different quantities with opposite monotonicity: in Eqs. (8), (10)
  and (11) the pairwise divergence, which increases with orthogonality, and in Eqs. (9) and (12) the
  cross-entropy penalty, which decreases with it. We now use distinct symbols: D_recon and D_ortho
  for the divergence measures, L_recon and L_ortho for the losses built on them.
- The prior term carried a negative sign while the algorithm minimised the total, which as Reviewer 6
  correctly observes would reward the divergence growing. The released code minimises
  L_recon + L_ortho + β·L_prior, and the manuscript now states that single canonical form in
  Eq. (1), Eq. (9), Supplementary Eq. (12) and Alg. 1.
- Supplementary Alg. 2, line 31, wrote the negative-pair cross-entropy as −log(1 − JSD). The
  implementation uses a target of one for negative pairs, giving −log(JSD). Corrected.

**On what orthogonality means (R4 #4).** Reviewer 4 is right that distributional divergence is not
geometric orthogonality, statistical independence, or semantic disentanglement, and that we moved
between these. What the objective enforces is distributional separation between component embeddings.
`[TODO: state the claim at the level the evidence supports, and cite the MI and GCN results as
evidence that separation approaches decorrelation in practice rather than by construction.]`

**On the stabilisation intuition (R1 #3).** `[TODO: two or three sentences for the main text on why
the cross-entropy formulation converts the adversarial objective into a stable minimisation and how
this relates to representation collapse. Consider adding the KL collapse curves.]`

**Changes to the manuscript:** `[TODO: locations]`

---

## MR-C. The probabilistic formulation

*Addresses R5 #3, R5 #13, R6 #1 (Eqs. 4b and 5).*

`[TODO. Issues to resolve, in order:`
- `Eq. (4b) writes p(z) as the product of the component priors. Reviewer 6 is right that this confuses`
  `the density of an additive variable with the joint density of independent variables. Decide whether`
  `to restate as a convolution, to restate the generative assumption, or to drop the factorisation.`
- `Eq. (5) is not a valid posterior factorisation: every factor conditions on all the others. State`
  `which conditional dependencies are actually modelled in the implementation, which is a single shared`
  `encoder with per-subspace heads and no explicit conditioning between subspaces.`
- `Eq. (7) normalisation, raised by R6 #1.`
- `Supplementary Eq. (8): p(z) is defined as the pushforward of p(h) under w, then used as the`
  `isotropic prior in the KL term. Reconcile or separate the two objects.`
- `Assumption 1 (invertibility of w): state what it does and does not buy, given d > k.`
- `R5 #3 asks for the connection between the assumed generative model and the training objective. Our`
  `position is that decomposition enters as a front-end and as a latent-space regulariser, and that`
  `the generative claim should be stated at that level.]`

---

## MR-D. Extended baselines

*Addresses R1 #5, R3 #1, R5 #2.*

We agree that the original comparison, being confined to independent-factor variational models and to
linear dimensionality reduction, did not test the claim against methods that use temporal or
spectral structure. We have added the following, all trained and evaluated under a protocol matched to
DecVAE.

| Method | Family | Datasets | Requested by |
|---|---|---|---|
| CoST | Decomposition-based representation learning | SimVowels, TIMIT | R3 #1, R5 #2 |
| TCL | Time-contrastive learning | SimVowels, TIMIT | R1 #5, R3 #1 |
| FHVAE | Disentangled sequential VAE | SimVowels, TIMIT | R1 #5, R5 #2 |
| TF-C | Self-supervised time-frequency representation learning | SimVowels, TIMIT | R5 #2 |
| CPC | Contrastive predictive coding | TIMIT | R5 #2 |
| wav2vec2 (frozen) | Self-supervised speech representation | SimVowels, TIMIT, IEMOCAP | R5 #2 |
| SFA | Slow feature analysis | SimVowels, TIMIT, IEMOCAP | R1 #5 |
| SlowVAE | Temporal sparse coding, slowness prior | SimVowels | R1 #5 |

`[PENDING RESULT: final set, and which are carried forward to IEMOCAP.]`

**Matched protocol.** All baselines use the same Mel filterbank front-end, matched latent
dimensionality, comparable parameter budgets, the same optimiser family and early-stopping criterion,
and identical downstream evaluation, namely five-fold outer cross-validation with five random
initialisations and leave-one-speaker-out on IEMOCAP. `[TODO: confirm final numbers and add to
Supplementary Table 5.]`

**Scope of the comparison.** Following Reviewer 3's observation that inclusion in at least one task
would strengthen the argument, we ran the full comparison on SimVowels and TIMIT and carried the
strongest baselines forward to IEMOCAP. SimVowels is the only dataset with ground-truth generative
factors, so the supervised disentanglement metrics are principled there; TIMIT is our real-speech
reference and the source of the transferred models; IEMOCAP is the hardest case for our
decomposition prior, since emotion has no direct time-frequency correspondence, and is therefore the
most informative test of the claim. We did not add baselines to VOC-ALS because that experiment
evaluates zero-shot transfer capability rather than representation quality, and transferring these
baselines under a protocol their authors did not intend would not be a fair comparison. This is
stated in the revised text and marked in the tables.

**Methods considered and not benchmarked.** Reviewer 5 asks for recent decomposition-based
representation learners. We benchmark CoST and TF-C as the members of that family that learn
unsupervised embeddings. Autoformer and FEDformer are forecasting architectures whose decomposition
blocks operate inside a supervised predictor and which do not define an unsupervised embedding;
LaST likewise requires forecasting targets. Benchmarking any of these would require inventing a
non-standard feature-extraction protocol. We therefore cite and discuss them, and we have added a
comparison to LaST in particular, which is the closest prior work to ours in that it decomposes a
latent space variationally with explicit independence pressure, differing in that its two components
are fixed semantic classes learned under forecasting supervision whereas ours are C signal-derived
components learned without supervision. `[TODO: also decide on SpeechSplit.]`

---

## MR-E. Scope of the decomposition prior, number of components, subspace semantics

*Addresses R3 #2, R4 #1, R4 #2, R4 #3, R5 #6, R5 #8.*

`[TODO. This is the second substantive theme and needs its own design session. Positions to develop:`
- `R4 #1 and R5 #6: frequency decomposition is an inductive bias, not a guarantee. Agree, and state`
  `when it is expected to help. Planned experiment: a synthetic dataset with non-frequency, mixed and`
  `correlated generative factors, to test the method where the prior is wrong. Decide whether`
  `baselines run there too.`
- `R4 #3: distinguish frequency-aligned subspaces from semantically aligned factors. Our position is`
  `that individual subspaces align with frequency-related factors and that semantic disentanglement`
  `is a property of the aggregate. Needs subspace-wise evidence: selective prediction per subspace,`
  `leave-one-subspace-out ablation, subspace swapping.`
- `R4 #2 and R5 #8: selection and misspecification of C. A principled selection rule is unlikely;`
  `planned instead is a systematic sensitivity analysis including deliberate misspecification, across`
  `datasets.`
- `R3 #2: fixed frequency boundaries in FD. Position: boundaries are a hyperparameter set once per`
  `signal class; good-enough boundaries suffice; speech shares spectral structure across our datasets,`
  `so generalisation holds within that class and would need re-tuning outside it. Say this plainly as`
  `a limitation.]`

---

# Reviewer 1

> The manuscript proposes Variational Decomposition Autoencoding (VDA) and its concrete architecture,
> DecVAE [...] The paper addresses a significant challenge in probabilistic AI.

We thank Reviewer 1 for the careful reading and for the constructive framing of the terminology issue
in particular, which prompted the reassessment described in MR-A and MR-B.

**R1 #1 (Minor). Autoencoder terminology; relabel or rename.**
See **MR-A**. We have adopted the first of the two options offered, labelling the mechanism as
latent, distribution-level autoencoding at first use, and we have retained the architecture name.
`[TODO: final wording, and the new definitional paragraph.]`

**R1 #2 (Minor). Compactness trade-off in deployment.**
`[TODO. Agree and add to the Discussion. Content: non-compact latents carry redundancy and increase
downstream cost at the edge and in clinical settings; quantify the dimensionality difference against
the VAE baselines; note the trade-off against the disentanglement and informativeness gains; note
that aggregation choice and subspace pruning are the natural levers.]`

**R1 #3 (Minor). Intuition for why the stabilised formulation avoids collapse.**
See **MR-B**. `[TODO: two or three sentences for the main text, plus a decision on adding the KL
collapse curves.]`

**R1 #4 (Minor). Notation mapping between main text and supplement.**
`[TODO. Add a notation mapping table. Must cover at least the main-text Z-tilde and z_c-tilde against
the supplementary h_i and Z^k, and the relation h = w^{-1}(z). Place at first use and repeat in the
supplement.]`

**R1 #5 (Major). Justify the absence of sequence-based or contrastive baselines.**
See **MR-D**. We have added them rather than justifying their absence. The slow-feature family is
represented by SFA and SlowVAE and the time-contrastive family by TCL. `[PENDING RESULT]`

**R1 #6. Sensitivity of β with respect to orthogonality and accuracy.**
`[TODO. We have a β sensitivity analysis for downstream accuracy. What is new here is presenting
orthogonality and accuracy against β on common axes. Clarify what the reviewer means by the
orthogonality constraint and report D_ortho at convergence against β alongside accuracy.]`

---

# Reviewer 2

> [Recommends acceptance after minor editorial polishing; asks for a proofreading pass and expanded
> figure captions.]

We thank Reviewer 2 for the generous assessment.

`[TODO. Two actions: (i) full proofreading pass for grammar and flow; (ii) expand figure captions
with explanatory detail for readers unfamiliar with decomposition-based representation learning.
Identify which captions. Candidates are Figs. 1, 2 and 6, where the decomposition mechanics and the
latent response analysis need most explanation.]`

---

# Reviewer 3

**R3 #1 (Major). Include time-series-specific baselines such as TCL and CoST in at least one task.**
See **MR-D**. Both are included, on two datasets rather than one. `[PENDING RESULT]`

**R3 #2 (Major). Do the fixed frequency boundaries of FD import excessive domain prior knowledge?**
See **MR-E**. `[TODO]`

**R3 #3 (Minor). Unresolved citation at line 499.**
Thank you. The reference now resolves correctly to Supplementary Alg. 3. `[TODO: confirm the
reviewer was reading an earlier compile, and state that a full sweep for unresolved references has
been carried out.]`

**R3 #4 (Minor). Missing space at line 508, and similar issues throughout.**
Corrected at line 508 and at line 736, where "Supplementary Alg. 1and" had the same fault. A full
formatting and typography sweep has been carried out. `[TODO: run the sweep.]`

---

# Reviewer 4

**R4 #1 (Major/Minor). Frequency decomposition is an inductive bias, not a guarantee.**
See **MR-E**. We agree with this and it is among the most useful comments we received. `[TODO]`

**R4 #2 (Major). The number of components is under-justified.**
See **MR-E**. `[TODO]`

**R4 #3 (Major). Subspace-wise evidence is needed for the interpretability claim.**
See **MR-E**. `[TODO. Note the distinction we wish to draw: subspaces align with frequency-related
factors; semantic alignment is a property of the aggregated representation. The requested analyses
are the right way to demonstrate this and we have added them.]`

**R4 #4 (Minor). Be precise about what orthogonality means.**
See **MR-B**. `[TODO]`

**R4 #5 (Major). Is DELBO a strict bound or a DELBO-inspired objective?**
See **MR-B**. It is the latter, and we have gone further than softening the terminology: we show why
no bound of the required form can exist, and we restate the claim accordingly.

**R4 #6 (Minor). Malformed algorithm reference on page 20.**
Thank you. This reference resolves correctly in the current version, and we have swept the manuscript
for unresolved references and formatting artefacts. `[TODO: confirm and run the sweep.]`

---

# Reviewer 5

**R5 #1 (Minor). Encoder-only model described as a VAE.**
See **MR-A**.

**R5 #2 (Major). Add stronger contemporary baselines.**
See **MR-D**. `[PENDING RESULT]`

**R5 #3 (Minor). Decomposition is claimed to enter the generative process but acts as a front-end.**
See **MR-C**. `[TODO. Our position: the reviewer is describing the implementation accurately, and the
generative claim should be stated at the level of the assumed model while the implementation is
described as a front-end plus latent regulariser. Say both.]`

**R5 #4 (Minor/Major). All real-world experiments are speech.**
`[DECISION NEEDED. Option A: narrow the claim to speech-like nonstationary time series throughout,
including title framing, abstract and Discussion. Option B: add one experiment on a different
modality, for example EEG or ECG. Option A is cheap and honest; Option B is expensive but answers a
reviewer who marked this Major/Minor. Note that TF-C, if adopted as a baseline, was validated on
physiological signals and gives a partial bridge.]`

**R5 #5 (Minor). Define or remove "chromagram-guided dynamics".**
Removed. The term appeared once, at line 92, and does not correspond to any component of the model.

**R5 #6 (Major). SimVowels is too favourable to the decomposition prior.**
See **MR-E**. `[TODO]`

**R5 #7 (Major). Limitations of the disentanglement metrics under correlated factors and label noise.**
`[TODO. Clarify what is meant by rankings changing. Reading: whether the ordering of methods by each
metric is stable under factor imbalance, label noise and partial observability. If so, this is a
robustness analysis over the existing latents and costs no pre-training: perturb the factor labels
and re-run the metric suite, reporting whether the method ordering is preserved.]`

**R5 #8 (Major). Automatic or misspecified selection of C.**
See **MR-E**. `[TODO]`

**R5 #9 (Major). VOC-ALS confounds; subject-independent controls and permutation tests.**
`[TODO. Agree. Planned: subject-independent splits, stratified analysis by phoneme and by speaker,
and permutation tests against the clinical stage label. This is the right place to spend VOC-ALS
effort, rather than on baselines.]`

**R5 #10 (Major). IEMOCAP: leave-session-out and leave-speaker-out separately; class-wise results.**
`[TODO. Note that in IEMOCAP session and speaker are confounded by design, since each session
contains a distinct speaker pair, so leave-session-out and leave-speaker-out are not independent
protocols. State this explicitly rather than reporting them as though they were. Class-wise
performance for angry, happy, neutral and sad can be added directly from the existing runs.]`

**R5 #11 (Major). Quantify latent-factor alignment and stability across seeds and folds.**
`[TODO. Partly free: the existing five-seed, five-fold protocol supports a stability analysis. What
is needed is a measure of whether the same dimensions encode the same factors across runs, for
example the correlation of DCI importance matrices across seeds.]`

**R5 #12 (Minor). Ablate against a non-variational encoder of the same architecture and capacity.**
This ablation is already in the manuscript. `[TODO: point the reviewer to it explicitly. The DecAE
variant, reported throughout the tables and figures, is DecVAE with β = 0, which retains the
architecture, the latent dimensionality and the decomposition-contrastive objective while removing
the prior approximation. Give the specific table and figure numbers, which were evidently not
signposted clearly enough.]`

**R5 #13 (Major). Eqs. (4b) and (5) are mathematically underdeveloped.**
See **MR-C**. `[TODO]`

**R5 #14 (Minor). The objective is not clearly a lower bound; prove it or rename it.**
See **MR-B**. We have renamed it, and we explain why the proof cannot be given.

**R5 #15 (Minor). Additional citations in affective computing and biomedical signal modelling.**
`[TODO. Assess the three suggested references for genuine relevance to our Discussion. If they are
relevant, cite them where they inform the argument rather than in a block.]`

---

# Reviewer 6

**R6 #1 (Minor). Eqs. (4b), (5), (7), (9) and the VAE terminology.**
See **MR-A** for the terminology and **MR-C** for the equations. On Eq. (9) specifically, see
**MR-B**: we agree that it is not shown to be a lower bound of any evidence, and we no longer claim
that it is.

**R6 #2 (Minor). Padding and random concatenation in Algorithm 1 are ambiguous.**
`[TODO. Give the precise procedure. Must answer the reviewer's specific worry, namely whether
randomly concatenating different utterances corrupts speaker or emotion labels. State what is
concatenated with what, whether concatenation is within-speaker, and how labels are assigned to the
resulting segments.]`

**R6 #3 (Major/Minor). Eq. (9) is not shown to be a bound; signs and directions are inconsistent.**
See **MR-B**. We are grateful for this comment. It is correct in every particular, including the
observation about the negative sign on the prior term, and it prompted the reassessment that we
regard as the most important change in this revision.

---

# Open decisions

1. **R5 #4.** Narrow the claim to speech, or add a non-speech modality.
2. **MR-B.** Final name for the objective, and whether to include the equilibrium proposition.
3. **MR-E.** Whether baselines are run on the new synthetic dataset requested by R5 #6.
4. **MR-D.** Whether SpeechSplit is benchmarked or only cited.
5. **R5 #15.** Whether the three suggested references are genuinely relevant.
6. Presentation of changes: tracked changes, colour, or a separate marked-up file.
