from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
}

NUCLEOTIDES = {
    "A", "C", "G", "U", "T",
    "DA", "DC", "DG", "DT", "DU",
    "CA", "CU", "CG", "CC",
    "GA", "GC", "GG", "GU",
}


@dataclass(frozen=True)
class ChainGroups:
    protein: tuple[str, ...]
    rna: tuple[str, ...]


def parse_chain_list(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = value.replace(";", ",").split(",")
    else:
        parts = value
    return tuple(part.strip() for part in parts if str(part).strip())


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def infer_chain_groups(pdb_path: str | Path) -> ChainGroups:
    """Infer protein/RNA chains from residue names in a combined PDB."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(Path(pdb_path).stem, str(pdb_path))
    protein_chains: list[str] = []
    rna_chains: list[str] = []

    for chain in next(structure.get_models()):
        aa_count = 0
        nt_count = 0
        for residue in chain.get_residues():
            resname = residue.get_resname().strip()
            aa_count += int(resname in AMINO_ACIDS)
            nt_count += int(resname in NUCLEOTIDES)
        if aa_count == 0 and nt_count == 0:
            continue
        if aa_count >= nt_count:
            protein_chains.append(chain.id)
        else:
            rna_chains.append(chain.id)

    return ChainGroups(tuple(protein_chains), tuple(rna_chains))


def write_split_pdbs(
    pdb_path: str | Path,
    output_dir: str | Path,
    case_id: str,
    protein_chains: Iterable[str],
    rna_chains: Iterable[str],
) -> tuple[Path, Path]:
    """Write filtered protein and RNA PDB files for graph construction."""
    from Bio.PDB import PDBIO, PDBParser, Select

    pdb_path = Path(pdb_path)
    output_dir = ensure_dir(output_dir)
    protein_chains = set(parse_chain_list(tuple(protein_chains)))
    rna_chains = set(parse_chain_list(tuple(rna_chains)))

    if not protein_chains:
        raise ValueError("No protein chains were provided or inferred.")
    if not rna_chains:
        raise ValueError("No RNA/DNA chains were provided or inferred.")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(case_id, str(pdb_path))

    class _ResidueSelect(Select):
        def __init__(self, chains: set[str], residue_names: set[str]):
            self.chains = chains
            self.residue_names = residue_names

        def accept_chain(self, chain):  # noqa: ANN001 - Bio.PDB callback
            return chain.id in self.chains

        def accept_residue(self, residue):  # noqa: ANN001 - Bio.PDB callback
            return residue.get_resname().strip() in self.residue_names

    protein_pdb = output_dir / f"{case_id}_protein.pdb"
    rna_pdb = output_dir / f"{case_id}_rna.pdb"

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(protein_pdb), _ResidueSelect(protein_chains, AMINO_ACIDS))
    io.save(str(rna_pdb), _ResidueSelect(rna_chains, NUCLEOTIDES))

    return protein_pdb, rna_pdb


def chain_groups_from_args(
    pdb_path: str | Path,
    protein_chains: str | Sequence[str] | None,
    rna_chains: str | Sequence[str] | None,
) -> ChainGroups:
    protein = parse_chain_list(protein_chains)
    rna = parse_chain_list(rna_chains)
    if protein and rna:
        return ChainGroups(protein, rna)

    inferred = infer_chain_groups(pdb_path)
    return ChainGroups(protein or inferred.protein, rna or inferred.rna)
