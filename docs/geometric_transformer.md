# Geometric Transformer

Geometric Transformer uses **Qin-style** geometric attention for
mutation-conditioned coordinate refinement. Its distance/angle-aware attention
is inspired by [Qin et al. (CVPR 2022)](https://arxiv.org/abs/2202.06688),
while its equivariant coordinate updates relate to
[Satorras et al. (ICML 2021)](https://arxiv.org/abs/2102.09844).

## Forward path

1. Construct mutant-topology inputs from reference coordinates and the requested
   substitution. Newly required atoms are initialized from generic ideal-residue
   conformers without consulting mutant coordinates.
2. Select the 12 Å mutation-centered atom crop, including the mutation site.
3. Embed atom, element, reference/mutant residue, ordered substitution, mutation-site,
   new-atom, Cα, sequence-position, and distance-to-mutation features.
4. Construct invariant sinusoidal pairwise-distance and triplet-angle embeddings,
   with separate projections and angular max/mean reduction.
5. Apply sparse geometric relative-position multi-head self-attention and FFNs.
6. Decode invariant scalar coefficients on relative coordinate vectors through a
   movement-gated equivariant decoder.
7. For benchmark evaluation, restore the rest of the initialized chain unchanged.

The registration correspondence/cross-attention components of the original
GeoTransformer are not used. This is a mutation-conditioned refinement adaptation,
not the authors' original registration model.

## Default configuration

The default hidden size is 128, with 8 heads, 4 attention blocks, 3 coordinate
refinement blocks, 4 angular anchors, distance scale 0.5 Å, angle scale 15 degrees,
and dropout 0.1. Spatial graph radius is 6 Å with at most 32 spatial neighbors per
node **plus required covalent edges**. Therefore total degree need not be ≤32.
The original coordinate/edge/clash loss coefficients are 1.0/0.20/0.02.

The shipped configuration specifies example defaults. Checkpoint-specific
settings, when distributed, are recorded separately in `configs/released_run.json`.
Generic frame/torsion data/loss helpers remain for model-cache compatibility;
no other GT model architectures are included or instantiated.

## Assets and limitations

`common/ideal_residue_templates.json` contains generic standard amino-acid
conformer templates needed to initialize mutant side chains. It is a model asset,
not an experimental protein-pair dataset or a release of MutaStruct records.

Refinement is restricted to the mutation-centered atom crop. The model
therefore does not represent arbitrary distant or whole-protein rearrangements.
Coordinates outside the crop remain at their initialized positions during
benchmark evaluation.

The common graph loader constructs paired inputs/supervision from an existing
manifest. This repository does not provide the withheld dataset-curation pipeline
or a universally validated arbitrary-protein inference API.