from __future__ import annotations

from pathlib import Path

from ciphernet_ood.data.sessionize import build_graph_dataset
from ciphernet_ood.data.splits import stratified_split
from ciphernet_ood.utils.io import save_json
from common import graph_files, load_project, parse_base_args, save_splits


def main() -> None:
    parser = parse_base_args()
    args = parser.parse_args()
    config = load_project(args)
    label_map = build_graph_dataset(config, args.dataset)
    files = graph_files(config, args.dataset)
    split_cfg = config["splits"]["closed_set"]
    splits = stratified_split(files, float(split_cfg["train"]), float(split_cfg["val"]), int(config["project"]["seed"]))
    save_splits(config, args.dataset, splits)
    split_root = Path(config["data"]["split_dir"]) / args.dataset
    save_json(label_map, split_root / "label_map.json")
    print(f"Built {len(files)} graph files for {args.dataset}.")


if __name__ == "__main__":
    main()
