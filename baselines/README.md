# Third-party baselines

The repository evaluates **existing** AlphaFold 3, ESMFold, FoldX, Rosetta and PreMut
predictions using the paired dataset once it is available. Their upstream source,
weights, software licenses and databases are not redistributed here.

The prediction runner `premut/run.py` is a MutaStruct
mmCIF-to-PreMut adapter. Install the upstream PreMut release and its dependencies
separately. Then, in that environment:

```bash
python -m pip install -r baselines/premut/requirements.txt
python baselines/premut/run.py \
  --premut-repo /path/to/installed/PreMut \
  --checkpoint /path/to/installed/PreMut/Saved_Model/model.ckpt \
  --manifest /path/to/tier_a_pairs_with_splits.parquet \
  --structures-dir /path/to/structures \
  --output-dir /path/to/PreMut/mutastruct_test_predictions \
  --max-cases 20
```

Remove `--max-cases` for the full test set. The adapter reads reference coordinates
and mutation identity, not experimental target coordinates, to form its model input.
It retains the source implementation's stated initialization and graph conventions;
it does not substitute Geometric Transformer for PreMut.

Upstream PreMut: https://github.com/jianlin-cheng/PreMut

For the other external models, this repository includes evaluators only, not new
prediction launchers. Model settings, source versions, MSAs/templates, ligand
inputs, seeds and mapping manifests are separate reproduction requirements.
The Geometric Transformer scripts do not generate predictions for these
external methods.
