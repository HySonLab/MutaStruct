#!/usr/bin/env python3
"""Exercise all evaluation entry points on synthetic coordinates and mappings.

These artificial tests do not validate all real biological mapping corner cases.
No third-party predictor or actual benchmark record is run/distributed.
"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / 'geometric_transformer'
sys.path.insert(0, str(GT))
import numpy as np
import pandas as pd
import gemmi

spec=importlib.util.spec_from_file_location('synthetic_builder',GT/'tests/end_to_end_smoke_test.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def run(script, args):
    cp = subprocess.run([sys.executable, str(ROOT/script), *map(str,args)],
                        cwd=ROOT, text=True, capture_output=True)
    if cp.returncode:
        raise RuntimeError(f'{script} failed\n{cp.stdout}\n{cp.stderr}')
    print(f'PASS {script}')


def main():
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp); structs=base/'structures';structs.mkdir()
        helper.write_cif(structs/'ref1.cif','AAAAA',False)
        helper.write_cif(structs/'mut1.cif','AAWAA',True)
        # Deliberate artificial local backbone movement for a nonempty >1A test stratum.
        doc=gemmi.cif.read(str(structs/'mut1.cif'));block=doc.sole_block()
        positions=list(block.find_values('_atom_site.label_seq_id'))
        ys=block.find_values('_atom_site.Cartn_y')
        for i,pos in enumerate(positions):
            if pos=='3': ys[i]=str(float(ys[i])+4.)
        doc.write_file(str(structs/'mut1.cif'))
        row=dict(pair_id='synthetic_pair',event_id='synthetic_event',split_cluster_id='synthetic_cluster',
                 split='test',pair_class='canonical_wt',reference_pdb_id='REF1',reference_entity_id=1,
                 reference_chain_label_asym_id='A',reference_sequence='AAAAA',reference_aa='A',
                 target_pdb_id='MUT1',target_entity_id=1,target_chain_label_asym_id='A',
                 target_sequence='AAWAA',target_aa='W',mutation_index_0=2,mutation='A3W')
        manifest=base/'manifest.csv';pd.DataFrame([row]).to_csv(manifest,index=False)
        paths=['--manifest',manifest,'--structures-dir',structs,'--cache-dir',base/'cache']
        run('evaluation/evaluate_controls.py',[*paths,'--output-dir',base/'controls'])
        pm=pd.read_csv(base/'controls/pair_metrics.csv')
        assert set(pm['baseline'])=={'Copy-WT','Mutant-Init'}
        assert pm.loc[pm['baseline']=='Copy-WT','local_ca_rmsd_8a'].iloc[0]>1

        # Standard-residue synthetic mutant prediction used by all four external PDB adapters.
        st=gemmi.read_structure(str(structs/'mut1.cif'))
        for chain in st[0]:
            chain.name='A'
            for i,res in enumerate(chain,1):
                res.name={'A':'ALA','W':'TRP'}[res.name]
                res.seqid=gemmi.SeqId(i,' ')
        pred=base/'predicted.pdb';st.write_minimal_pdb(str(pred))
        esm=base/'esm';esm.mkdir()
        h=hashlib.sha256(b'AAWAA').hexdigest()
        (esm/f'SEQ_{h}.pdb').write_text(pred.read_text())
        pre=base/'premut';(pre/'pair_predictions').mkdir(parents=True)
        (pre/'pair_predictions/synthetic_pair.pdb').write_text(pred.read_text())
        fold=base/'foldx';fold.mkdir();ros=base/'rosetta';ros.mkdir()
        mapping=base/'predmap.csv';pd.DataFrame([{'pair_id':'synthetic_pair','prediction_path':str(pred)}]).to_csv(mapping,index=False)
        run('evaluation/evaluate_baselines.py',[*paths,'--output-dir',base/'external',
             '--esmfold-root',esm,'--premut-root',pre,'--foldx-root',fold,'--foldx-map-csv',mapping,
             '--rosetta-root',ros,'--rosetta-map-csv',mapping])
        ext=pd.read_csv(base/'external/pair_metrics.csv')
        assert set(ext['model'])=={'ESMFold','PreMut','FoldX','Rosetta'}, ext
        assert np.isfinite(ext['ca_rmsd']).all()

        af=base/'af3';af.mkdir();pid='af3_'+h[:16]
        st.make_mmcif_document().write_file(str(af/f'{pid}_model.cif'))
        afmap=base/'afmap.csv';pd.DataFrame([{'pair_id':'synthetic_pair','target_prediction_id':pid}]).to_csv(afmap,index=False)
        run('evaluation/evaluate_alphafold3.py',[*paths,'--af3-structures',af,'--af3-mapping',afmap,
                                               '--output-dir',base/'af_eval'])
        apm=pd.read_csv(base/'af_eval/pair_metrics.csv')
        assert len(apm)==1 and np.isfinite(apm['ca_rmsd']).all()
        hard=base/'hard'
        run('evaluation/build_hard_subset.py',['--manifest',manifest,'--copywt-pair-metrics',base/'controls/pair_metrics.csv',
             '--output-dir',hard,'--threshold','1','--expected-events','1','--expected-pairs','1'])
        run('evaluation/evaluate_hard_subset.py',['--hard-subset-dir',hard,
            '--result',base/'controls/pair_metrics.csv','--result',base/'external/pair_metrics.csv',
            '--result',base/'af_eval/pair_metrics.csv','--no-discover',
            '--bootstrap','100','--output-dir',base/'hard_eval'])
        coverage=pd.read_csv(base/'hard_eval/prediction_coverage.csv')
        assert len(coverage)==7 and (coverage['events_present']==1).all()
    print('All control/external/AF3/subset synthetic evaluation checks passed.')

if __name__=='__main__':
    main()
