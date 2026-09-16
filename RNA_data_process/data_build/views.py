from __future__ import annotations

from pathlib import Path


def render_three_views(
    pdb_path: str | Path,
    output_dir: str | Path,
    case_id: str,
    protein_chains: list[str] | tuple[str, ...],
    rna_chains: list[str] | tuple[str, ...],
    resolution: int = 800,
) -> list[Path]:
    """Render front/side/top PNGs with the existing PyMOL view script."""
    try:
        from data.PRA310.PRA310.get_frames import generate_three_views
    except Exception as exc:  # PyMOL is optional.
        raise RuntimeError(
            "PyMOL view generation is unavailable. Install pymol or run without --make-views."
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chain_info = {
        case_id.upper(): {
            "protein_chains": list(protein_chains),
            "rna_chains": list(rna_chains),
            "kd": "N/A",
            "dg": "N/A",
            "DDG": "N/A",
        }
    }
    generate_three_views(str(pdb_path), str(output_dir), chain_info, resolution=resolution)
    return [output_dir / f"{case_id}_{view}.png" for view in ("front", "side", "top")]
