from __future__ import annotations

from pathlib import Path

import torch

from ciphernet_ood.ood.signatures import fit_cpg_signatures
from ciphernet_ood.ood.threshold import calibrate_threshold
from ciphernet_ood.training.checkpointing import load_checkpoint
from ciphernet_ood.utils.device import get_device
from common import build_model, graph_files, loaders_from_splits, load_project, load_splits, parse_base_args


def main() -> None:
    parser = parse_base_args()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    config = load_project(args)
    device = get_device()
    splits = load_splits(config, args.dataset)
    loaders = loaders_from_splits(config, splits)
    model = build_model(config, graph_files(config, args.dataset), device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    signatures = fit_cpg_signatures(model, loaders["train"], config, device)
    signatures["threshold"] = calibrate_threshold(model, loaders["val"], signatures, config, device)
    output = Path(config["project"]["checkpoint_dir"]) / args.dataset / "cpg_signatures.pt"
    torch.save(signatures, output)
    print(f"Saved signatures to {output}")


if __name__ == "__main__":
    main()
