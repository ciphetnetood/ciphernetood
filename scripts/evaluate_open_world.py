from __future__ import annotations

import torch

from ciphernet_ood.evaluation.open_world import evaluate_open_world
from ciphernet_ood.training.checkpointing import load_checkpoint
from ciphernet_ood.utils.device import get_device
from common import build_model, graph_files, loaders_from_splits, load_project, load_splits, parse_base_args


def main() -> None:
    parser = parse_base_args()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--signatures", required=True)
    parser.add_argument("--ood-split", default=None)
    args = parser.parse_args()
    config = load_project(args)
    device = get_device()
    splits = load_splits(config, args.dataset)
    loaders = loaders_from_splits(config, splits)
    model = build_model(config, graph_files(config, args.dataset), device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    signatures = torch.load(args.signatures, map_location="cpu")
    ood_files = splits.get("ood", splits["test"] if args.ood_split is None else [])
    ood_loader = loaders_from_splits(config, {"ood": ood_files})["ood"]
    print(evaluate_open_world(model, loaders["test"], ood_loader, signatures, config, device))


if __name__ == "__main__":
    main()
