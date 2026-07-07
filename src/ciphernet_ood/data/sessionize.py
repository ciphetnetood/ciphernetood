from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from ciphernet_ood.data.graph_builder import session_to_graph
from ciphernet_ood.data.pcap_tools import read_pcap_sessions


def build_graph_dataset(config: dict, dataset: str) -> dict[str, int]:
    raw_root = Path(config["data"]["raw_dir"]) / dataset
    out_root = Path(config["data"]["processed_dir"]) / dataset
    out_root.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([path for path in raw_root.iterdir() if path.is_dir()])
    label_map = {path.name: idx for idx, path in enumerate(class_dirs)}
    for class_dir in class_dirs:
        label = label_map[class_dir.name]
        out_class = out_root / class_dir.name
        out_class.mkdir(parents=True, exist_ok=True)
        captures = list(class_dir.glob("*.pcap")) + list(class_dir.glob("*.pcapng"))
        for capture in tqdm(captures, desc=f"{dataset}/{class_dir.name}"):
            sessions = read_pcap_sessions(capture)
            for session_idx, packets in enumerate(sessions):
                graph = session_to_graph(packets, label, config)
                graph.class_name = class_dir.name
                graph.source_file = str(capture)
                graph.session_index = session_idx
                output = out_class / f"{capture.stem}_{session_idx:06d}.pt"
                import torch

                torch.save(graph, output)
    return label_map
