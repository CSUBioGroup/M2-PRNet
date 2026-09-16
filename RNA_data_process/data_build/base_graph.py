from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .structure_utils import AMINO_ACIDS, NUCLEOTIDES, parse_chain_list


PROTEIN_SS_DEFAULT = [0, 0, 1]
PROTEIN_FEATURE_DIM = 8
RNA_FEATURE_DIM = 9


def normalize_angle(angle: float | int | None) -> float:
    if angle is None:
        angle = 0.0
    return (float(angle) + 180.0) / 360.0


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and hasattr(value.cpu(), "numpy"):
        return value.cpu().numpy()
    return np.asarray(value)


@dataclass
class EmbeddingStore:
    """Read per-chain embeddings from existing pickle files.

    Supported keys: <case_id>_prot_<chain> and <case_id>_rna_<chain>.
    Arrays can be residue-only [L, D] or tokenized [CLS, residues..., EOS].
    """

    combined: dict[str, Any] | None = None
    protein: dict[str, Any] | None = None
    rna: dict[str, Any] | None = None
    embedding_dim: int = 1280
    allow_zero: bool = True
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_pickles(
        cls,
        combined_pkl: str | Path | None = None,
        protein_pkl: str | Path | None = None,
        rna_pkl: str | Path | None = None,
        embedding_dim: int = 1280,
        allow_zero: bool = True,
    ) -> "EmbeddingStore":
        def _load(path: str | Path | None):
            if path is None:
                return None
            with Path(path).open("rb") as handle:
                return pickle.load(handle)

        return cls(
            combined=_load(combined_pkl),
            protein=_load(protein_pkl),
            rna=_load(rna_pkl),
            embedding_dim=embedding_dim,
            allow_zero=allow_zero,
        )

    def _candidate_keys(self, case_id: str, kind: str, chain_id: str) -> list[str]:
        stem = Path(case_id).stem
        short = stem[:4] if len(stem) >= 4 else stem
        aliases = [case_id, stem, short]
        separators = ["_", "*"]
        candidates = [
            f"{alias}{sep}{kind}{sep}{chain_id}"
            for alias in aliases
            for sep in separators
        ]
        candidates = [
            *candidates,
            f"{case_id}_{'protein' if kind == 'prot' else kind}_{chain_id}",
            f"{stem}_{'protein' if kind == 'prot' else kind}_{chain_id}",
        ]
        return list(dict.fromkeys(candidates))

    def _lookup(self, case_id: str, kind: str, chain_id: str):
        stores = []
        if kind == "prot" and self.protein is not None:
            stores.append(self.protein)
        if kind == "rna" and self.rna is not None:
            stores.append(self.rna)
        if self.combined is not None:
            stores.append(self.combined)

        for key in self._candidate_keys(case_id, kind, chain_id):
            for store in stores:
                if key in store:
                    return key, store[key]
        return None, None

    def get_chain_embeddings(
        self,
        case_id: str,
        kind: str,
        chain_id: str,
        residue_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        found_key, raw_value = self._lookup(case_id, kind, chain_id)
        if raw_value is None:
            message = (
                f"Missing embedding for {case_id}_{kind}_{chain_id}; "
                f"using zeros with shape ({residue_count}, {self.embedding_dim})."
            )
            self.warnings.append(message)
            if not self.allow_zero:
                raise KeyError(message)
            return (
                np.zeros((residue_count, self.embedding_dim), dtype=np.float32),
                np.zeros((self.embedding_dim,), dtype=np.float32),
            )

        arr = _as_numpy(raw_value).astype(np.float32)
        if arr.ndim != 2:
            message = f"Embedding {found_key} has shape {arr.shape}; expected 2D."
            self.warnings.append(message)
            if not self.allow_zero:
                raise ValueError(message)
            return (
                np.zeros((residue_count, self.embedding_dim), dtype=np.float32),
                np.zeros((self.embedding_dim,), dtype=np.float32),
            )

        arr = self._fix_dim(arr, found_key)
        if arr.shape[0] == residue_count:
            residue_emb = arr
            whole_emb = arr.mean(axis=0) if residue_count else np.zeros((self.embedding_dim,), dtype=np.float32)
        elif arr.shape[0] >= residue_count + 1:
            whole_emb = arr[0]
            residue_emb = arr[1 : residue_count + 1]
        else:
            self.warnings.append(
                f"Embedding {found_key} is shorter than residues: {arr.shape[0]} < {residue_count}; padded with zeros."
            )
            residue_emb = np.zeros((residue_count, self.embedding_dim), dtype=np.float32)
            usable = min(arr.shape[0], residue_count)
            residue_emb[:usable] = arr[:usable]
            whole_emb = arr[0] if arr.shape[0] else np.zeros((self.embedding_dim,), dtype=np.float32)
        return residue_emb, whole_emb

    def _fix_dim(self, arr: np.ndarray, key: str) -> np.ndarray:
        if arr.shape[1] == self.embedding_dim:
            return arr
        fixed = np.zeros((arr.shape[0], self.embedding_dim), dtype=np.float32)
        width = min(arr.shape[1], self.embedding_dim)
        fixed[:, :width] = arr[:, :width]
        self.warnings.append(
            f"Embedding {key} dim {arr.shape[1]} adjusted to {self.embedding_dim}."
        )
        return fixed


class BaseLevelGraphBuilder:
    def __init__(self, distance_threshold: float = 8.0, dssp_bin: str | None = None):
        from Bio.PDB import PDBParser

        self.distance_threshold = distance_threshold
        self.dssp_bin = dssp_bin
        self.parser = PDBParser(QUIET=True)

    def build_base_info(self, protein_pdb: str | Path, rna_pdb: str | Path) -> dict[str, Any]:
        protein_bases = self.extract_residue_coordinates(protein_pdb, "protein")
        rna_bases = self.extract_residue_coordinates(rna_pdb, "rna")
        if not protein_bases:
            raise ValueError(f"No protein residues found in {protein_pdb}")
        if not rna_bases:
            raise ValueError(f"No RNA/DNA residues found in {rna_pdb}")
        neighbor_matrix = self.build_neighbor_matrix(protein_bases, rna_bases)
        return {
            "protein_bases": protein_bases,
            "rna_bases": rna_bases,
            "neighbor_matrix": neighbor_matrix,
            "prot_path": str(protein_pdb),
            "rna_path": str(rna_pdb),
        }

    def extract_residue_coordinates(self, pdb_path: str | Path, molecule_type: str) -> dict[str, dict[str, Any]]:
        structure = self.parser.get_structure(Path(pdb_path).stem, str(pdb_path))
        model = next(structure.get_models())
        md_features = self._mdtraj_features(pdb_path, molecule_type)
        ss_features = self._dssp_features(pdb_path) if molecule_type == "protein" else {}
        allowed = AMINO_ACIDS if molecule_type == "protein" else NUCLEOTIDES

        residues: dict[str, dict[str, Any]] = {}
        for chain in model:
            for residue in chain:
                residue_name = residue.get_resname().strip()
                if residue_name not in allowed:
                    continue
                resnum = residue.get_id()[1]
                key = f"{chain.id}_{residue_name}_{resnum}"
                coord, b_factor = self._residue_coord_and_bfactor(residue, molecule_type)
                item = {
                    "residue_name": residue_name,
                    "chain_id": chain.id,
                    "residue_number": resnum,
                    "coord": coord,
                    "b_factor": b_factor,
                    "molecule_type": molecule_type,
                    "sas": md_features.get(key, {}).get("sas", 0.0),
                }
                if molecule_type == "protein":
                    item["ss"] = ss_features.get(f"{chain.id}_{resnum}", PROTEIN_SS_DEFAULT)
                    item.update({name: md_features.get(key, {}).get(name, 0.0) for name in ("phi", "psi", "omega")})
                else:
                    item.update(
                        {
                            name: md_features.get(key, {}).get(name, 0.0)
                            for name in ("gamma", "delta", "chi", "alpha", "beta", "epsilon", "zeta")
                        }
                    )
                residues[key] = item
        return residues

    def _residue_coord_and_bfactor(self, residue, molecule_type: str):  # noqa: ANN001
        if molecule_type == "protein" and "CA" in residue:
            atom = residue["CA"]
            return atom.get_coord(), atom.get_bfactor()
        if molecule_type == "rna":
            if "C1'" in residue:
                atom = residue["C1'"]
                return atom.get_coord(), atom.get_bfactor()
            if "C1*" in residue:
                atom = residue["C1*"]
                return atom.get_coord(), atom.get_bfactor()
        coords = [atom.get_coord() for atom in residue.get_atoms()]
        if not coords:
            return np.zeros((3,), dtype=np.float32), 0.0
        return np.mean(coords, axis=0), 0.0

    def _mdtraj_features(self, pdb_path: str | Path, molecule_type: str) -> dict[str, dict[str, float]]:
        try:
            import mdtraj as md
        except Exception:
            return {}

        try:
            traj = md.load(str(pdb_path))
        except Exception:
            return {}

        features: dict[str, dict[str, float]] = {}
        try:
            sasa = md.shrake_rupley(traj, mode="residue")
            for chain in traj.topology.chains:
                for residue in chain.residues:
                    key = f"{chain.chain_id}_{residue.name}_{residue.resSeq}"
                    features.setdefault(key, {})["sas"] = float(sasa[0, residue.index])
        except Exception:
            pass

        if molecule_type == "protein":
            self._add_protein_dihedrals(traj, features, md)
        else:
            self._add_rna_dihedrals(traj, features, md)
        return features

    def _add_protein_dihedrals(self, traj, features: dict[str, dict[str, float]], md):  # noqa: ANN001
        for name, func, atom_position in (
            ("phi", md.compute_phi, 1),
            ("psi", md.compute_psi, 0),
            ("omega", md.compute_omega, 0),
        ):
            try:
                indices, values = func(traj)
            except Exception:
                continue
            for atom_indices, angle in zip(indices, values[0]):
                residue = traj.topology.atom(int(atom_indices[atom_position])).residue
                key = f"{residue.chain.chain_id}_{residue.name}_{residue.resSeq}"
                features.setdefault(key, {})[name] = float(np.degrees(angle))

    def _add_rna_dihedrals(self, traj, features: dict[str, dict[str, float]], md):  # noqa: ANN001
        definitions = {
            "alpha": ["O3'", "P", "O5'", "C5'"],
            "beta": ["P", "O5'", "C5'", "C4'"],
            "gamma": ["O5'", "C5'", "C4'", "C3'"],
            "delta": ["C5'", "C4'", "C3'", "O3'"],
            "epsilon": ["C4'", "C3'", "O3'", "P"],
            "zeta": ["C3'", "O3'", "P", "O5'"],
        }
        for residue in traj.topology.residues:
            if residue.name not in NUCLEOTIDES:
                continue
            key = f"{residue.chain.chain_id}_{residue.name}_{residue.resSeq}"
            for angle_name, atom_names in definitions.items():
                atom_indices = [self._find_md_atom(residue, atom_name) for atom_name in atom_names]
                if any(idx is None for idx in atom_indices):
                    continue
                try:
                    value = md.compute_dihedrals(traj, [atom_indices])[0][0]
                    features.setdefault(key, {})[angle_name] = float(np.degrees(value))
                except Exception:
                    continue
            chi_atoms = ["O4'", "C1'", "N9", "C4"] if residue.name in {"A", "G", "DA", "DG"} else ["O4'", "C1'", "N1", "C2"]
            atom_indices = [self._find_md_atom(residue, atom_name) for atom_name in chi_atoms]
            if not any(idx is None for idx in atom_indices):
                try:
                    value = md.compute_dihedrals(traj, [atom_indices])[0][0]
                    features.setdefault(key, {})["chi"] = float(np.degrees(value))
                except Exception:
                    pass

    @staticmethod
    def _find_md_atom(residue, atom_name: str):  # noqa: ANN001
        aliases = {atom_name}
        if "'" in atom_name:
            aliases.add(atom_name.replace("'", "*"))
        if "*" in atom_name:
            aliases.add(atom_name.replace("*", "'"))
        for atom in residue.atoms:
            if atom.name in aliases:
                return atom.index
        return None

    def _dssp_features(self, pdb_path: str | Path) -> dict[str, list[int]]:
        if not self.dssp_bin:
            return {}
        try:
            from Bio.PDB import DSSP

            structure = self.parser.get_structure(Path(pdb_path).stem, str(pdb_path))
            model = next(structure.get_models())
            dssp = DSSP(model, str(pdb_path), dssp=self.dssp_bin)
        except Exception:
            return {}
        ss = {}
        for key in dssp.keys():
            chain_id = key[0]
            resnum = key[1][1]
            ss[f"{chain_id}_{resnum}"] = self.ss_code_to_onehot(dssp[key][2])
        return ss

    @staticmethod
    def ss_code_to_onehot(ss_code: str) -> list[int]:
        if ss_code in {"H", "G", "I"}:
            return [1, 0, 0]
        if ss_code in {"E", "B"}:
            return [0, 1, 0]
        return PROTEIN_SS_DEFAULT

    def build_neighbor_matrix(self, protein_bases: dict[str, Any], rna_bases: dict[str, Any]) -> np.ndarray:
        nodes = list(protein_bases.items()) + list(rna_bases.items())
        total = len(nodes)
        matrix = np.zeros((total, total), dtype=np.float32)
        for i, (_key_i, item_i) in enumerate(nodes):
            for j in range(i + 1, total):
                _key_j, item_j = nodes[j]
                distance = float(np.linalg.norm(item_i["coord"] - item_j["coord"]))
                if distance <= self.distance_threshold:
                    matrix[i, j] = distance
                    matrix[j, i] = distance
        return matrix


def build_res_level_entry(
    base_info: dict[str, Any],
    case_id: str,
    embedding_store: EmbeddingStore | None = None,
    protein_chains: str | list[str] | tuple[str, ...] | None = None,
    rna_chains: str | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    protein_chain_filter = set(parse_chain_list(protein_chains))
    rna_chain_filter = set(parse_chain_list(rna_chains))
    protein_items = [
        (key, value)
        for key, value in base_info["protein_bases"].items()
        if not protein_chain_filter or value["chain_id"] in protein_chain_filter
    ]
    rna_items = [
        (key, value)
        for key, value in base_info["rna_bases"].items()
        if not rna_chain_filter or value["chain_id"] in rna_chain_filter
    ]

    if not protein_items or not rna_items:
        raise ValueError("Cannot build res_level entry without both protein and RNA residues.")

    pro_feats = []
    pro_coords = []
    rna_feats = []
    rna_coords = []
    mol_indicator = []
    chain_ids = []

    for _key, item in protein_items:
        pro_feats.append(
            [
                item.get("sas", 0.0),
                item.get("b_factor", 0.0),
                normalize_angle(item.get("phi", 0.0)),
                normalize_angle(item.get("psi", 0.0)),
                normalize_angle(item.get("omega", 0.0)),
                *item.get("ss", PROTEIN_SS_DEFAULT),
            ]
        )
        pro_coords.append(item.get("coord", [0.0, 0.0, 0.0]))
        mol_indicator.append([1, 0])
        chain_ids.append(item["chain_id"])

    for _key, item in rna_items:
        rna_feats.append(
            [
                item.get("sas", 0.0),
                item.get("b_factor", 0.0),
                normalize_angle(item.get("gamma", 0.0)),
                normalize_angle(item.get("delta", 0.0)),
                normalize_angle(item.get("chi", 0.0)),
                normalize_angle(item.get("alpha", 0.0)),
                normalize_angle(item.get("beta", 0.0)),
                normalize_angle(item.get("epsilon", 0.0)),
                normalize_angle(item.get("zeta", 0.0)),
            ]
        )
        rna_coords.append(item.get("coord", [0.0, 0.0, 0.0]))
        mol_indicator.append([0, 1])
        chain_ids.append(item["chain_id"])

    embedding_store = embedding_store or EmbeddingStore()
    prot_emb, prot_whole = _concat_embeddings(case_id, "prot", protein_items, embedding_store)
    rna_emb, rna_whole = _concat_embeddings(case_id, "rna", rna_items, embedding_store)

    chain_indicator = []
    chain_to_idx: dict[str, int] = {}
    for chain_id in chain_ids:
        if chain_id not in chain_to_idx and len(chain_to_idx) < 6:
            chain_to_idx[chain_id] = len(chain_to_idx)
        one_hot = [0] * 6
        if chain_id in chain_to_idx:
            one_hot[chain_to_idx[chain_id]] = 1
        chain_indicator.append(one_hot)

    all_keys = list(base_info["protein_bases"].keys()) + list(base_info["rna_bases"].keys())
    selected_keys = [key for key, _item in protein_items] + [key for key, _item in rna_items]
    key_to_idx = {item_key: idx for idx, item_key in enumerate(all_keys)}
    selected_idx = [key_to_idx[item_key] for item_key in selected_keys]
    neighbor_matrix = np.asarray(base_info["neighbor_matrix"], dtype=np.float32)[
        np.ix_(selected_idx, selected_idx)
    ]

    data_type = [0] * len(pro_feats) + [1] * len(rna_feats)
    return {
        "pro_feats": pro_feats,
        "rna_feats": rna_feats,
        "mol_indicator": np.asarray(mol_indicator),
        "chain_indicator": np.asarray(chain_indicator),
        "pro_coords": np.asarray(pro_coords),
        "rna_coords": np.asarray(rna_coords),
        "data_type": np.asarray(data_type),
        "prot_emb": np.asarray(prot_emb, dtype=np.float32),
        "rna_emb": np.asarray(rna_emb, dtype=np.float32),
        "neighbor_matrix": neighbor_matrix,
        "prot_len": len(pro_feats),
        "rna_len": len(rna_feats),
        "prot_whole_emb": np.asarray(prot_whole, dtype=np.float32),
        "rna_whole_emb": np.asarray(rna_whole, dtype=np.float32),
    }


def _concat_embeddings(
    case_id: str,
    kind: str,
    items: list[tuple[str, dict[str, Any]]],
    store: EmbeddingStore,
) -> tuple[np.ndarray, np.ndarray]:
    chain_order: list[str] = []
    per_chain_counts: dict[str, int] = {}
    for _key, item in items:
        chain_id = item["chain_id"]
        if chain_id not in per_chain_counts:
            chain_order.append(chain_id)
            per_chain_counts[chain_id] = 0
        per_chain_counts[chain_id] += 1

    residue_embeddings = []
    whole_embeddings = []
    for chain_id in chain_order:
        residues, whole = store.get_chain_embeddings(case_id, kind, chain_id, per_chain_counts[chain_id])
        residue_embeddings.append(residues)
        whole_embeddings.extend(np.atleast_1d(whole).tolist())
    if residue_embeddings:
        return np.concatenate(residue_embeddings, axis=0), np.asarray(whole_embeddings, dtype=np.float32)
    return np.zeros((0, store.embedding_dim), dtype=np.float32), np.asarray([], dtype=np.float32)


def save_base_outputs(
    case_id: str,
    key: str,
    base_info: dict[str, Any],
    res_level_entry: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "case",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_info_path = output_dir / f"{prefix}_base_info.pkl"
    res_level_path = output_dir / f"{prefix}_res_level.pkl"
    with base_info_path.open("wb") as handle:
        pickle.dump({case_id: base_info}, handle)
    with res_level_path.open("wb") as handle:
        pickle.dump({key: res_level_entry}, handle)
    return base_info_path, res_level_path
