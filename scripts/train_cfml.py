from __future__ import annotations

from pathlib import Path

from ciphernet_ood.training.trainer import train_cfml
from ciphernet_ood.utils.device import get_device
from ciphernet_ood.utils.seed import seed_everything
from common import build_model, graph_files, loaders_from_splits, load_project, load_splits, parse_base_args


def main() -> None:
    parser = parse_base_args()
    parser.add_argument("--open-world", action="store_true")
    args = parser.parse_args()
    config = load_project(args)
    seed_everything(int(config["project"]["seed"]))
    device = get_device()
    splits = load_splits(config, args.dataset)
    loaders = loaders_from_splits(config, splits)
    model = build_model(config, graph_files(config, args.dataset), device)
    ckpt_dir = Path(config["project"]["checkpoint_dir"]) / args.dataset
    metrics = train_cfml(model, loaders["train"], loaders["val"], config, device, ckpt_dir)
    print(metrics)


if __name__ == "__main__":
    main()
