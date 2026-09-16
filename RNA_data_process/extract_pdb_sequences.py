from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from Bio.PDB import PDBParser, PPBuilder

from data_build.structure_utils import AMINO_ACIDS, NUCLEOTIDES


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

NT_TO_1 = {
    "A": "A",
    "DA": "A",
    "C": "C",
    "DC": "C",
    "G": "G",
    "DG": "G",
    "U": "U",
    "DU": "U",
    "T": "U",
    "DT": "U",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract protein/RNA chain sequences from PDB files.")
    parser.add_argument("--pdb", action="append", default=[], help="Combined PDB file. Can be repeated.")
    parser.add_argument("--protein-pdb", help="Protein-only PDB file for one case.")
    parser.add_argument("--rna-pdb", help="RNA-only PDB file for one case.")
    parser.add_argument("--case-id", help="Case id for --protein-pdb/--rna-pdb mode.")
    parser.add_argument("--out-pkl", required=True)
    return parser.parse_args()


def extract_sequences_from_pdb(pdb_path: str | Path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(Path(pdb_path).stem, str(pdb_path))
    model = next(structure.get_models())
    prot = {}
    rna = {}
    for chain in model:
        residues = list(chain.get_residues())
        names = [residue.get_resname().strip() for residue in residues]
        aa_count = sum(name in AMINO_ACIDS for name in names)
        nt_count = sum(name in NUCLEOTIDES for name in names)
        if aa_count >= nt_count and aa_count > 0:
            seq = "".join(AA3_TO_1.get(name, "X") for name in names if name in AMINO_ACIDS)
            if seq:
                prot[chain.id] = seq
        elif nt_count > 0:
            seq = "".join(NT_TO_1.get(name, "N") for name in names if name in NUCLEOTIDES)
            if seq:
                rna[chain.id] = seq
    return prot, rna


def main() -> None:
    args = parse_args()
    results = {}

    for pdb_path in args.pdb:
        case_id = Path(pdb_path).stem
        prot, rna = extract_sequences_from_pdb(pdb_path)
        results[case_id] = {"prot": prot, "rna": rna}
        print(f"{case_id}: prot={list(prot)}, rna={list(rna)}")

    if args.protein_pdb or args.rna_pdb:
        if not (args.protein_pdb and args.rna_pdb and args.case_id):
            raise SystemExit("--protein-pdb, --rna-pdb, and --case-id are required together.")
        prot, _unused = extract_sequences_from_pdb(args.protein_pdb)
        _unused, rna = extract_sequences_from_pdb(args.rna_pdb)
        results[args.case_id] = {"prot": prot, "rna": rna}
        print(f"{args.case_id}: prot={list(prot)}, rna={list(rna)}")

    out_path = Path(args.out_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(results, handle)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
