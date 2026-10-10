# Evaluation protocol

Scores are based on mutually represented and experimentally observed atoms.
Global/local/side-chain RMSDs use Å; lDDT is unitless on [0,1]. Atom correspondence,
coverage, and alignment anchors must be reported alongside the scores.

| Column | Procedure |
|---|---|
| `ca_rmsd` | Fit predicted to experimental mutant coordinates with one proper-rotation Kabsch fit over common Cα atoms; score those Cα positions. |
| `local_ca_rmsd_8a` | Use the same global transform; score common Cα positions within 8 Å of the mutation-site Cα in the reference. No separate local fit. |
| `mutation_sidechain_rmsd` | Use the same global Cα fit and supported symmetry-equivalent atom-name permutations; score matching mutation-site side-chain heavy atoms. Exclude N, CA, C, O, OXT. |
| `lddt` | Compare distances for common heavy-atom pairs from different residues with experimental target distance within 15 Å; average indicators for 0.5/1/2/4 Å tolerances. The distance score does not require superposition. |
| `mutation_response_rmsd_8a` | Independently fit predicted mutant and experimental mutant to the reference over the same common Cα positions; compare their reference-relative displacement vectors inside the same 8 Å neighborhood. |

The shared row-vector convention is `mobile @ R + t`. Reflections are excluded.
The local mask is defined from the reference, not from the prediction. The model
crop (12 Å by default) is separate from the metric's 8 Å Cα neighborhood.

For a common WT reference W, MR-RMSD compares
`(aligned_prediction - W) - (aligned_target - W)`; W cancels algebraically in this
difference but determines both fits and the local selection. It is a
reference-frame structural error, not a causal test proving that the substitution
alone produced the experimental difference. For sequence-only models that only
predict the mutant, ordinary de novo prediction error is also present.

The atom-name symmetry handling includes D/E carboxyl oxygens, R terminal
nitrogens, L/V terminal branches and coupled F/Y aromatic-ring swaps. The existing
evaluators minimize coordinate disagreement for label assignment, then calculate
their distance score. This is the implemented common-atom lDDT protocol, not a
claim that every behavior equals a separate full-featured lDDT software package.
Missing predicted atoms are excluded from common-atom scores, so coverage is
essential to interpreting a comparison.

## Aggregation

For each model and metric:
1. Calculate one score per coordinate pair on its valid atom subset.
2. Average valid pair scores within each `event_id`.
3. Average those event scores with equal event weight.
4. Report metric-specific pair and event denominators.

Do not confuse the event-median *subset membership rule* with the event-mean
*performance aggregation*. A glycine mutant may have no side-chain heavy atoms;
its side-chain metric is unavailable, not zero.

Copy-WT predicts unchanged reference Cα coordinates and provides a no-change
baseline. Its primary mutant all-atom lDDT and side-chain scores are deliberately
not reported, because the required mutant atom topology has not been instantiated.
Computing a common-atom diagnostic for Copy-WT is possible but is a different
comparison. Mutant-Init supplies the actual initialized mutant topology and can be
scored on all five metrics where target atoms are observed.

## Common implementation and separate adapters

`geometric_transformer/evaluate.py` implements the model's five metrics and is
also imported by the control evaluator. External-prediction scripts retain the
previously used parsing/matching implementations, redirected to the same
model-ready graph loader. They are not replacements for raw prediction mappings.
Synthetic tests exercise these adapters on controlled structures, not all biological
corner cases. Model-result CSVs with different coverage or altered mapping rules
must not be presented as directly comparable without auditing the differences.

Event-bootstrap intervals from the subset aggregation describe variability across
the evaluated events conditional on fitted models. They do not include model
selection, training-seed or repeated-sampling uncertainty, and event independence
is not guaranteed within larger sequence split clusters.
