from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AtomGraphArgs:
    """Minimal args object needed by utils.dataset_utils graph helpers."""

    model: str = "EquiScore"
    pred_mode: str = "ligand"
    virtual_aromatic_atom: bool = True
    edge_bias: bool = True
    rel_3d_pos_bias: bool = True
    in_degree_bias: bool = True
    FP: bool = True
    lap_pos_enc: bool = False
    pos_enc_dim: int = 16


def _simple_atom_mapping(mol, bases, offset: int = 0):  # noqa: ANN001 - rdkit type optional
    from utils.dataset_utils import GetNum

    mapping = {}
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_type = GetNum(
            atom.GetSymbol(),
            ["C", "N", "O", "S", "F", "P", "Cl", "Br", "B", "H", "other"],
        )
        mapping[atom_idx + offset] = {
            "type": "atom",
            "type_id": atom_type[0],
            "rdkit_index": atom_idx,
            "bases": bases[atom_idx],
        }
    return mapping


def build_atom_graph(
    protein_pdb: str | Path,
    rna_pdb: str | Path,
    args: AtomGraphArgs | None = None,
):
    """Build the atom-level DGL graph used by EquiScore."""
    import dgl
    import numpy as np
    import torch
    from rdkit import Chem

    from utils.dataset_utils import (
        add_atom_to_mol,
        get_atom_graphformer_feature,
        get_mol_info,
        mol2graph,
        preprocess_item_map,
    )
    from utils.ifp_construct import get_nonBond_pair

    args = args or AtomGraphArgs()
    protein = Chem.MolFromPDBFile(str(protein_pdb), removeHs=True, sanitize=False)
    rna = Chem.MolFromPDBFile(str(rna_pdb), removeHs=True, sanitize=False)
    if protein is None:
        raise ValueError(f"RDKit failed to read protein PDB: {protein_pdb}")
    if rna is None:
        raise ValueError(f"RDKit failed to read RNA PDB: {rna_pdb}")

    Chem.AssignStereochemistry(protein, cleanIt=True, force=True)
    Chem.AssignStereochemistry(rna, cleanIt=True, force=True)

    n1, d1, adj1, bases_rna = get_mol_info(rna)
    n2, d2, adj2, bases_protein = get_mol_info(protein)

    h1 = np.concatenate(
        [get_atom_graphformer_feature(rna, FP=True), np.zeros((n1, 1), dtype=int)],
        axis=1,
    )
    h2 = np.concatenate(
        [get_atom_graphformer_feature(protein, FP=True), np.ones((n2, 1), dtype=int)],
        axis=1,
    )

    if args.virtual_aromatic_atom:
        adj1, h1, d1, n1, mapping_rna = add_atom_to_mol(
            rna, adj1, h1, d1, n1, bases_rna
        )
        adj2, h2, d2, n2, mapping_protein = add_atom_to_mol(
            protein, adj2, h2, d2, n2, bases_protein
        )
    else:
        mapping_rna = _simple_atom_mapping(rna, bases_rna)
        mapping_protein = _simple_atom_mapping(protein, bases_protein)

    features = torch.from_numpy(np.concatenate([h1, h2], axis=0))
    adj = np.zeros((n1 + n2, n1 + n2))
    adj[:n1, :n1] = adj1
    adj[n1:, n1:] = adj2

    special_edges: set[tuple[int, int]] = set()
    subatom: list[int] = []
    try:
        atom_pairs, _interaction_types, subatom = get_nonBond_pair(rna, protein)
    except Exception:
        atom_pairs = []
        subatom = []

    if atom_pairs:
        fp_pairs = np.asarray(atom_pairs)
        u = list(fp_pairs[:, 0]) + list(n1 + fp_pairs[:, 1])
        v = list(n1 + fp_pairs[:, 1]) + list(fp_pairs[:, 0])
        adj[u, v] = 1
        special_edges = set(zip(u, v))

    adj_for_graph = np.copy(adj)
    item = mol2graph((rna, protein), features, args, adj=adj_for_graph, n1=n1, n2=n2, dm=(d1, d2))
    graph = preprocess_item_map(item, args, np.copy(adj_for_graph), n1, n2, mapping_rna, mapping_protein)

    edge_types = torch.zeros(graph.number_of_edges(), dtype=torch.int32)
    src, dst = graph.edges()
    for edge_id in range(graph.number_of_edges()):
        if (src[edge_id].item(), dst[edge_id].item()) in special_edges:
            edge_types[edge_id] = 1
    graph.edata["type"] = edge_types
    graph.ndata["coors"] = torch.from_numpy(np.concatenate([d1, d2], axis=0))

    nodes_to_keep = list(range(n1))
    for idx in range(n1, n1 + n2):
        if adj[idx, :n1].sum() > 0 or adj[:n1, idx].sum() > 0:
            nodes_to_keep.append(idx)
    for idx in subatom:
        node_idx = idx + n1
        if node_idx not in nodes_to_keep and node_idx < n1 + n2:
            nodes_to_keep.append(node_idx)

    graph = dgl.node_subgraph(graph, nodes_to_keep)
    valid = torch.zeros((graph.num_nodes(),))
    if args.pred_mode == "ligand":
        valid[: min(n1, graph.num_nodes())] = 1
    elif args.pred_mode == "protein":
        valid[min(n1, graph.num_nodes()) :] = 1
    else:
        raise ValueError(f"Unsupported pred_mode: {args.pred_mode}")
    graph.ndata["V"] = valid.float().reshape(-1, 1)
    return graph


def build_full_graph(graph, cutoff: float = 8.0):  # noqa: ANN001 - DGL type optional
    import dgl
    import numpy as np
    from scipy.spatial import distance_matrix

    src_edges, dst_edges = graph.edges()
    coords = graph.ndata["coors"].detach().cpu().numpy()
    distance = distance_matrix(coords, coords)

    if graph.number_of_edges() > 0:
        graph_edges = np.concatenate(
            [
                src_edges.reshape(-1, 1).detach().cpu().numpy(),
                dst_edges.reshape(-1, 1).detach().cpu().numpy(),
            ],
            axis=1,
        )
    else:
        graph_edges = np.empty((0, 2), dtype=np.int64)

    src, dst = np.where(distance < cutoff)
    radius_edges = np.concatenate([src.reshape(-1, 1), dst.reshape(-1, 1)], axis=1)
    all_edges = np.unique(np.concatenate([radius_edges, graph_edges], axis=0), axis=0)

    full_graph = dgl.graph((all_edges[:, 0], all_edges[:, 1]), num_nodes=graph.num_nodes())
    full_graph.ndata["coors"] = graph.ndata["coors"]
    full_graph.ndata["x"] = graph.ndata["x"]
    return full_graph


def save_atom_graph_pickles(
    key: str,
    graph,
    full_graph,
    output_dir: str | Path,
    prefix: str = "case",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / f"{prefix}_graph.pkl"
    all_graph_path = output_dir / f"{prefix}_allgraphs.pkl"
    with graph_path.open("wb") as handle:
        pickle.dump({key: graph}, handle)
    with all_graph_path.open("wb") as handle:
        pickle.dump({key: full_graph}, handle)
    return graph_path, all_graph_path


def write_atom_graph_lmdb(
    key: str,
    graph,
    full_graph,
    label: float,
    lmdb_path: str | Path,
    map_size: int = int(1e10),
):
    import lmdb

    lmdb_path = Path(lmdb_path)
    lmdb_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_path), map_size=map_size, max_dbs=1)
    try:
        graph_db = env.open_db("data".encode())
        with env.begin(write=True, db=graph_db) as txn:
            txn.put(key.encode(), pickle.dumps((graph, full_graph, label)))
    finally:
        env.close()
    return lmdb_path
