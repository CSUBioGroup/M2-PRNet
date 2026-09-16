from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .structure_utils import chain_groups_from_args, ensure_dir, parse_chain_list, write_split_pdbs


@dataclass
class BuildConfig:
    pdb_path: str | None = None
    protein_pdb: str | None = None
    rna_pdb: str | None = None
    case_id: str | None = None
    key: str | None = None
    output_dir: str = "build_output"
    protein_chains: str | None = None
    rna_chains: str | None = None
    affinity: float = 0.0
    atom_cutoff: float = 8.0
    base_cutoff: float = 8.0
    prefix: str | None = None
    lmdb_name: str = "atom_graph_lmdb"
    embedding_pkl: str | None = None
    protein_embedding_pkl: str | None = None
    rna_embedding_pkl: str | None = None
    embedding_dim: int = 1280
    allow_zero_embeddings: bool = True
    dssp_bin: str | None = None
    make_views: bool = False
    view_resolution: int = 800
    skip_atom: bool = False
    skip_base: bool = False


@dataclass
class BuildResult:
    case_id: str
    key: str
    output_dir: str
    protein_pdb: str
    rna_pdb: str
    files: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def build_from_pdb(config: BuildConfig) -> BuildResult:
    output_dir = ensure_dir(config.output_dir)
    case_id = _resolve_case_id(config)
    key = config.key or f"{case_id}.pdb"
    prefix = config.prefix or case_id

    protein_pdb, rna_pdb, protein_chains, rna_chains = _prepare_structure_inputs(config, case_id, output_dir)
    result = BuildResult(
        case_id=case_id,
        key=key,
        output_dir=str(output_dir),
        protein_pdb=str(protein_pdb),
        rna_pdb=str(rna_pdb),
    )

    if not config.skip_atom:
        from .atom_graph import (
            AtomGraphArgs,
            build_atom_graph,
            build_full_graph,
            save_atom_graph_pickles,
            write_atom_graph_lmdb,
        )

        graph = build_atom_graph(protein_pdb, rna_pdb, AtomGraphArgs())
        full_graph = build_full_graph(graph, cutoff=config.atom_cutoff)
        lmdb_path = output_dir / config.lmdb_name
        write_atom_graph_lmdb(key, graph, full_graph, config.affinity, lmdb_path)
        graph_pkl, allgraphs_pkl = save_atom_graph_pickles(key, graph, full_graph, output_dir, prefix=prefix)
        result.files.update(
            {
                "lmdb": str(lmdb_path),
                "graph_pkl": str(graph_pkl),
                "allgraphs_pkl": str(allgraphs_pkl),
            }
        )

    if not config.skip_base:
        from .base_graph import (
            BaseLevelGraphBuilder,
            EmbeddingStore,
            build_res_level_entry,
            save_base_outputs,
        )

        store = EmbeddingStore.from_pickles(
            combined_pkl=config.embedding_pkl,
            protein_pkl=config.protein_embedding_pkl,
            rna_pkl=config.rna_embedding_pkl,
            embedding_dim=config.embedding_dim,
            allow_zero=config.allow_zero_embeddings,
        )
        builder = BaseLevelGraphBuilder(distance_threshold=config.base_cutoff, dssp_bin=config.dssp_bin)
        base_info = builder.build_base_info(protein_pdb, rna_pdb)
        res_entry = build_res_level_entry(
            base_info,
            case_id=case_id,
            embedding_store=store,
            protein_chains=protein_chains,
            rna_chains=rna_chains,
        )
        base_info_pkl, res_level_pkl = save_base_outputs(
            case_id, key, base_info, res_entry, output_dir, prefix=prefix
        )
        result.files.update(
            {
                "base_info_pkl": str(base_info_pkl),
                "res_level_pkl": str(res_level_pkl),
            }
        )
        result.warnings.extend(store.warnings)

    keys_csv = _write_keys_csv(output_dir, prefix, key, config.affinity)
    result.files["keys_csv"] = str(keys_csv)

    if config.make_views:
        from .views import render_three_views

        source_pdb = Path(config.pdb_path) if config.pdb_path else _merge_view_source(
            output_dir, case_id, protein_pdb, rna_pdb
        )
        view_files = render_three_views(
            source_pdb,
            output_dir / "views",
            case_id,
            protein_chains,
            rna_chains,
            resolution=config.view_resolution,
        )
        result.files["views_dir"] = str(output_dir / "views")
        result.files["view_pngs"] = json.dumps([str(path) for path in view_files], ensure_ascii=False)

    manifest_path = _write_manifest(config, result)
    result.files["manifest"] = str(manifest_path)
    return result


def _resolve_case_id(config: BuildConfig) -> str:
    if config.case_id:
        return config.case_id
    if config.pdb_path:
        return Path(config.pdb_path).stem
    if config.protein_pdb:
        name = Path(config.protein_pdb).stem
        for suffix in ("_protein", "_PROT"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name
    raise ValueError("Provide --pdb or --protein-pdb/--rna-pdb.")


def _prepare_structure_inputs(config: BuildConfig, case_id: str, output_dir: Path):
    split_dir = ensure_dir(output_dir / "split")
    if config.pdb_path:
        groups = chain_groups_from_args(config.pdb_path, config.protein_chains, config.rna_chains)
        protein_pdb, rna_pdb = write_split_pdbs(
            config.pdb_path,
            split_dir,
            case_id,
            groups.protein,
            groups.rna,
        )
        return protein_pdb, rna_pdb, groups.protein, groups.rna

    if not config.protein_pdb or not config.rna_pdb:
        raise ValueError("Use either --pdb or both --protein-pdb and --rna-pdb.")

    protein_pdb = split_dir / f"{case_id}_protein.pdb"
    rna_pdb = split_dir / f"{case_id}_rna.pdb"
    shutil.copyfile(config.protein_pdb, protein_pdb)
    shutil.copyfile(config.rna_pdb, rna_pdb)
    protein_chains = parse_chain_list(config.protein_chains)
    rna_chains = parse_chain_list(config.rna_chains)
    return protein_pdb, rna_pdb, protein_chains, rna_chains


def _write_keys_csv(output_dir: Path, prefix: str, key: str, affinity: float) -> Path:
    path = output_dir / f"{prefix}_keys.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "affinity", "fold0", "fold1", "fold2", "fold3", "fold4"])
        writer.writerow([key, affinity, "test", "test", "test", "test", "test"])
    return path


def _merge_view_source(output_dir: Path, case_id: str, protein_pdb: Path, rna_pdb: Path) -> Path:
    merged = output_dir / f"{case_id}.pdb"
    with merged.open("w") as out:
        for path in (protein_pdb, rna_pdb):
            with Path(path).open() as handle:
                for line in handle:
                    if line.startswith("END"):
                        continue
                    out.write(line)
        out.write("END\n")
    return merged


def _write_manifest(config: BuildConfig, result: BuildResult) -> Path:
    output_dir = Path(result.output_dir)
    path = output_dir / "manifest.json"
    payload = {
        "config": asdict(config),
        "result": asdict(result),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path
