from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract protein/RNA sequence embeddings for EquiScore data building."
    )
    parser.add_argument("--seqs-pkl", required=True, help="Pickle with {'name': {'prot': {}, 'rna': {}}}.")
    parser.add_argument("--out-pkl", required=True, help="Output embedding pickle.")
    parser.add_argument("--device", default=None, help="Example: cuda:0 or cpu. Defaults to cuda:0 if available.")
    parser.add_argument("--key-sep", default="*", choices=["*", "_"], help="Embedding key separator.")
    parser.add_argument("--protein-layer", type=int, default=33)
    parser.add_argument("--rna-model", default="giga-v1")
    return parser.parse_args()


def load_models(device, protein_layer: int, rna_model_name: str):
    os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"

    import esm
    from rinalmo.pretrained import get_pretrained_model

    model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model_esm = model_esm.to(device)
    model_esm.eval()
    batch_converter = alphabet.get_batch_converter()

    rna_model, rna_alphabet = get_pretrained_model(model_name=rna_model_name)
    rna_model = rna_model.to(device)
    rna_model.eval()

    return model_esm, batch_converter, rna_model, rna_alphabet, protein_layer


def get_protein_embedding(seq: str, model_esm, batch_converter, device, layer: int):
    data = [("protein", seq)]
    _labels, _strs, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model_esm(tokens, repr_layers=[layer], return_contacts=False)

    rep = results["representations"][layer]
    return rep[0, 1:-1].detach().cpu().numpy()


def get_rna_embedding(seq: str, rna_model, rna_alphabet, device, use_amp: bool):
    import torch

    tokens = torch.tensor(
        rna_alphabet.batch_tokenize([seq]),
        dtype=torch.int64,
        device=device,
    )

    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = rna_model(tokens)
        else:
            outputs = rna_model(tokens)

    hidden = outputs["representation"]
    return hidden[0, 1:-1].detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    import torch

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda"

    with Path(args.seqs_pkl).open("rb") as handle:
        seqs_dict = pickle.load(handle)

    model_esm, batch_converter, rna_model, rna_alphabet, protein_layer = load_models(
        device, args.protein_layer, args.rna_model
    )

    results = {}
    for name, item in seqs_dict.items():
        for chain, seq in item.get("prot", {}).items():
            key = f"{name}{args.key_sep}prot{args.key_sep}{chain}"
            emb = get_protein_embedding(seq, model_esm, batch_converter, device, protein_layer)
            results[key] = emb
            print(f"done: {key} {emb.shape}")

        for chain, seq in item.get("rna", {}).items():
            key = f"{name}{args.key_sep}rna{args.key_sep}{chain}"
            emb = get_rna_embedding(seq, rna_model, rna_alphabet, device, use_amp)
            results[key] = emb
            print(f"done: {key} {emb.shape}")

    out_path = Path(args.out_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(results, handle)

    print(f"All embeddings saved: {out_path}")


if __name__ == "__main__":
    main()
