from __future__ import annotations

import torch

from ciphernet_ood.data.graph_builder import session_to_graph
from ciphernet_ood.data.pcap_tools import read_pcap_sessions
from ciphernet_ood.ood.scorer import cpg_scores
from ciphernet_ood.training.checkpointing import load_checkpoint
from ciphernet_ood.utils.device import get_device
from common import build_model, graph_files, load_project, parse_base_args


def main() -> None:
    parser = parse_base_args()
    parser.add_argument("--pcap", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--signatures", default=None)
    args = parser.parse_args()
    config = load_project(args)
    device = get_device()
    model = build_model(config, graph_files(config, args.dataset), device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    sessions = read_pcap_sessions(args.pcap)
    for idx, packets in enumerate(sessions):
        graph = session_to_graph(packets, label=0, config=config)
        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        graph = graph.to(device)
        if args.signatures:
            signatures = torch.load(args.signatures, map_location="cpu")
            score, pred = cpg_scores(model, graph, signatures, config)
            status = "OOD" if float(score.item()) > float(signatures["threshold"]) else "ID"
            print({"session": idx, "prediction": int(pred.item()), "ood_score": float(score.item()), "status": status})
        else:
            print({"session": idx, "prediction": int(model.predict(graph).item())})


if __name__ == "__main__":
    main()
