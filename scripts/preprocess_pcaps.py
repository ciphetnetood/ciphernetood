from __future__ import annotations

from pathlib import Path

from ciphernet_ood.data.pcap_tools import convert_pcapng_to_pcap, split_with_splitcap
from common import load_project, parse_base_args


def main() -> None:
    parser = parse_base_args()
    parser.add_argument("--use-splitcap", action="store_true")
    args = parser.parse_args()
    config = load_project(args)
    raw_root = Path(config["data"]["raw_dir"]) / args.dataset
    session_root = Path(config["data"]["session_dir"]) / args.dataset
    for pcapng in raw_root.glob("**/*.pcapng"):
        converted = session_root / pcapng.relative_to(raw_root).with_suffix(".pcap")
        convert_pcapng_to_pcap(pcapng, converted, config["data"].get("editcap_path", "editcap"))
        if args.use_splitcap:
            split_with_splitcap(converted, converted.with_suffix(""), config["data"].get("splitcap_path", "SplitCap.exe"))


if __name__ == "__main__":
    main()
