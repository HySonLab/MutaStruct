# Method and source notices

This repository contains the MutaStruct mutation-refinement adaptation,
not the original point-cloud registration project.

Geometric structure embeddings and relative-position attention are inspired by:
Zheng Qin, Hao Yu, Changjian Wang, Yulan Guo, Yuxing Peng, and Kai Xu.
*Geometric Transformer for Fast and Robust Point Cloud Registration*. CVPR, 2022.

The scalar-relative-vector coordinate-refinement principle is related to:
Víctor Garcia Satorras, Emiel Hoogeboom, and Max Welling.
*E(n) Equivariant Graph Neural Networks*. ICML, 2021.

The optional PreMut adapter is based on the supplied MutaStruct integration code;
PreMut's software and checkpoint must be installed from their original release.
No authorship of the original methods or rights to their separate distributions
is claimed. See `LICENSING.md` for the licensing status.

Generic standard-residue templates are retained as a necessary model initialization
asset, not as a table of curated experimental mutation pairs.
