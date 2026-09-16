# EquiScore Data Build Workflow

This is the cleaned one-command path for turning a protein-RNA/DNA PDB into model inputs.

## One-Command Build

Combined PDB:

```bash
conda run -n unimo python build_model_inputs.py ^
  --pdb F:\path\case.pdb ^
  --protein-chains A,B ^
  --rna-chains C ^
  --case-id case ^
  --out-dir output\case_build ^
  --embedding-pkl F:\path\seq_embeddings.pkl
```

Separate protein/RNA PDBs:

```bash
conda run -n unimo python build_model_inputs.py ^
  --protein-pdb F:\path\case_protein.pdb ^
  --rna-pdb F:\path\case_rna.pdb ^
  --case-id case ^
  --out-dir output\case_build ^
  --protein-embedding-pkl F:\path\pro_embeding.pkl ^
  --rna-embedding-pkl F:\path\rna_embeding.pkl
```

If embedding files are not provided, the builder writes 1280-d zero embeddings and records warnings in `manifest.json`. That is useful for checking the pipeline, but real scoring/training should use real sequence embeddings.

Use the `unimo` conda environment for this repo's current graph-building dependencies. The base Python environment on this machine does not include the full `numpy/torch/dgl/rdkit/mdtraj/lmdb/prolif` stack.

## Real Sequence Embeddings

For a single protein/RNA PDB pair, first extract chain sequences:

```bash
conda run -n unimo python extract_pdb_sequences.py ^
  --protein-pdb data\testPNA\1A4T_B.pdb ^
  --rna-pdb data\testPNA\1A4T_A.pdb ^
  --case-id 1A4T ^
  --out-pkl output\1A4T_seqs_dict.pkl
```

Then extract real sequence embeddings with ESM2 for protein and RiNALMo for RNA:

```bash
conda run -n unimo python extract_sequence_embeddings.py ^
  --seqs-pkl output\1A4T_seqs_dict.pkl ^
  --out-pkl output\1A4T_seq_embeddings.pkl
```

The embedding keys are compatible with the existing extraction convention:

- `1A4T*prot*B`
- `1A4T*rna*A`

Pass the resulting pickle into model input construction:

```bash
conda run -n unimo python build_model_inputs.py ^
  --protein-pdb data\testPNA\1A4T_B.pdb ^
  --rna-pdb data\testPNA\1A4T_A.pdb ^
  --case-id 1A4T ^
  --embedding-pkl output\1A4T_seq_embeddings.pkl ^
  --out-dir output\1A4T_build
```

## Methods To Package Together

Required for atom-level graphs:

- `data_build/atom_graph.py`
- `utils/dataset_utils.py`
- `utils/ifp_construct.py`

Required for base-level graphs:

- `data_build/base_graph.py`
- `data_build/structure_utils.py`

Required for one-click build:

- `build_model_inputs.py`
- `data_build/pipeline.py`
- `data_build/__init__.py`

Optional for three-view figures:

- `data_build/views.py`
- `data/PRA310/PRA310/get_frames.py`

Original reference scripts:

- `getProtein_RNA_Graph.py`: original atom graph examples, especially `_GetGraph()` and `make_caseData()`.
- `data/testPNA/reslevel_graph.py`: original base/residue-level graph experiments.
- `data/PRA310/PRA310/get_frames.py`: original PyMOL three-view rendering.

## Build Stages

1. Structure input

   `structure_utils.py` either splits a combined PDB by chain or copies separate protein/RNA PDBs into:

   - `split/<case_id>_protein.pdb`
   - `split/<case_id>_rna.pdb`

2. Atom graph

   `atom_graph.build_atom_graph()` builds the covalent plus interaction graph. `build_full_graph()` adds the geometric radius graph. Outputs:

   - `<prefix>_graph.pkl`
   - `<prefix>_allgraphs.pkl`
   - `<lmdb_name>/data`, where each key stores `(g, full_g, Y)`

3. Base-level graph

   `BaseLevelGraphBuilder.build_base_info()` extracts residue/base coordinates, SASA, dihedral placeholders/features, and a distance matrix. `build_res_level_entry()` formats the model fields:

   - `pro_feats`
   - `rna_feats`
   - `mol_indicator`
   - `chain_indicator`
   - `pro_coords`
   - `rna_coords`
   - `prot_emb`
   - `rna_emb`
   - `neighbor_matrix`
   - `prot_len`
   - `rna_len`
   - `prot_whole_emb`
   - `rna_whole_emb`

   Outputs:

   - `<prefix>_base_info.pkl`
   - `<prefix>_res_level.pkl`

4. Keys and manifest

   The builder writes:

   - `<prefix>_keys.csv`
   - `manifest.json`

## Dependency Checklist

Core Python packages:

- `numpy`
- `scipy`
- `pandas`
- `torch`
- `dgl`
- `rdkit`
- `biopython`
- `mdtraj`
- `lmdb`
- `prolif`
- `tqdm`

Optional external tools:

- PyMOL: only needed for `--make-views`.
- DSSP / `mkdssp`: optional, pass with `--dssp-bin`; otherwise protein secondary structure defaults to coil.

Data/model files to keep with a packaged build:

- Sequence embedding pickle(s), with keys like `<case_id>_prot_A` and `<case_id>_rna_C`.
- Model weights, for example files under `workdir/official_weight/`.
- The generated `manifest.json`, because it records the exact chains, key, output paths, and missing embedding warnings.

## Output Compatibility Notes

The new builder keeps the existing training/inference convention:

- Atom graph LMDB key defaults to `<case_id>.pdb`.
- LMDB value is `(g, full_g, Y)`.
- `res_level.pkl` is a dict keyed by the same key.
- `keys.csv` contains `key, affinity, fold0..fold4`.

For existing `ESDataset_m`, point `args.lmdb_cache` to the generated LMDB directory and load the generated `<prefix>_res_level.pkl`.
