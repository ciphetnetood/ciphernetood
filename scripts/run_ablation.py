from __future__ import annotations

from copy import deepcopy

from common import load_project, parse_base_args


def main() -> None:
    parser = parse_base_args()
    parser.add_argument("--ablation", choices=["packet_length", "margin", "gram", "component"], required=True)
    args = parser.parse_args()
    config = load_project(args)
    variants = []
    if args.ablation == "packet_length":
        for p in [10, 15, 20, 25, 30, 50, 100]:
            cfg = deepcopy(config)
            cfg["session"]["max_packets"] = p
            variants.append((f"P={p}", cfg))
    elif args.ablation == "margin":
        for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
            cfg = deepcopy(config)
            cfg["cfml"]["beta"] = beta
            variants.append((f"beta={beta}", cfg))
    elif args.ablation == "gram":
        for layers in [[5], [4, 5], [3, 4, 5], [1, 2, 3, 4, 5]]:
            cfg = deepcopy(config)
            cfg["ood"]["selected_layers"] = layers
            cfg["ood"]["layer_weights"] = [1.0 / len(layers)] * len(layers)
            variants.append((f"layers={layers}", cfg))
    else:
        variants.append(("full", config))
    print("Prepared ablation variants. Run train/evaluate with each emitted config in a sweep runner.")
    for name, _ in variants:
        print(name)


if __name__ == "__main__":
    main()
