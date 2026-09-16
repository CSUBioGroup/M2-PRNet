from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build EquiScore atom graphs, base-level inputs, and optional views from PDB files."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdb", dest="pdb_path", help="Combined protein-RNA/DNA PDB file.")
    source.add_argument("--protein-pdb", help="Protein-only PDB file. Requires --rna-pdb.")
    parser.add_argument("--rna-pdb", help="RNA/DNA-only PDB file when --protein-pdb is used.")

    parser.add_argument("--case-id", help="Case id used for output names and embedding keys.")
    parser.add_argument("--key", help="Model data key. Defaults to <case-id>.pdb.")
    parser.add_argument("--out-dir", default="build_output", help="Output directory.")
    parser.add_argument("--prefix", help="Output file prefix. Defaults to case id.")
    parser.add_argument("--protein-chains", help="Comma-separated protein chains, e.g. A,B.")
    parser.add_argument("--rna-chains", help="Comma-separated RNA/DNA chains, e.g. C.")
    parser.add_argument("--affinity", type=float, default=0.0, help="Label stored in LMDB/keys CSV.")

    parser.add_argument("--embedding-pkl", help="Combined embedding pickle.")
    parser.add_argument("--protein-embedding-pkl", help="Protein embedding pickle.")
    parser.add_argument("--rna-embedding-pkl", help="RNA embedding pickle.")
    parser.add_argument("--embedding-dim", type=int, default=1280)
    parser.add_argument(
        "--no-zero-embeddings",
        action="store_true",
        help="Fail if required embeddings are missing instead of using zero placeholders.",
    )

    parser.add_argument("--atom-cutoff", type=float, default=8.0)
    parser.add_argument("--base-cutoff", type=float, default=8.0)
    parser.add_argument("--lmdb-name", default="atom_graph_lmdb")
    parser.add_argument("--dssp-bin", help="Optional mkdssp executable path.")
    parser.add_argument("--skip-atom", action="store_true", help="Skip atom graph/LMDB generation.")
    parser.add_argument("--skip-base", action="store_true", help="Skip base-level res_level generation.")
    parser.add_argument("--make-views", action="store_true", help="Render front/side/top PNGs with PyMOL.")
    parser.add_argument("--view-resolution", type=int, default=800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.protein_pdb and not args.rna_pdb:
        raise SystemExit("--rna-pdb is required when --protein-pdb is used.")

    from data_build import BuildConfig, build_from_pdb

    config = BuildConfig(
        pdb_path=args.pdb_path,
        protein_pdb=args.protein_pdb,
        rna_pdb=args.rna_pdb,
        case_id=args.case_id,
        key=args.key,
        output_dir=args.out_dir,
        protein_chains=args.protein_chains,
        rna_chains=args.rna_chains,
        affinity=args.affinity,
        atom_cutoff=args.atom_cutoff,
        base_cutoff=args.base_cutoff,
        prefix=args.prefix,
        lmdb_name=args.lmdb_name,
        embedding_pkl=args.embedding_pkl,
        protein_embedding_pkl=args.protein_embedding_pkl,
        rna_embedding_pkl=args.rna_embedding_pkl,
        embedding_dim=args.embedding_dim,
        allow_zero_embeddings=not args.no_zero_embeddings,
        dssp_bin=args.dssp_bin,
        make_views=args.make_views,
        view_resolution=args.view_resolution,
        skip_atom=args.skip_atom,
        skip_base=args.skip_base,
    )
    result = build_from_pdb(config)

    print(f"Built inputs for {result.key}")
    print(f"Output: {result.output_dir}")
    for name, path in result.files.items():
        print(f"  {name}: {path}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
