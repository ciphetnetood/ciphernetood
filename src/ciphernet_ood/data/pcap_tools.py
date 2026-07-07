from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path

from ciphernet_ood.data.packet_features import PacketRecord


def run_external_tool(command: list[str]) -> None:
    subprocess.run(command, check=True)


def convert_pcapng_to_pcap(input_path: Path, output_path: Path, editcap_path: str = "editcap") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_external_tool([editcap_path, "-F", "pcap", str(input_path), str(output_path)])


def split_with_splitcap(input_path: Path, output_dir: Path, splitcap_path: str = "SplitCap.exe") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_external_tool([splitcap_path, "-r", str(input_path), "-s", "session", "-o", str(output_dir)])


def _packet_flow_key(packet) -> tuple | None:
    if not packet.haslayer("IP"):
        return None
    ip = packet["IP"]
    proto = int(ip.proto)
    sport = dport = 0
    if packet.haslayer("TCP"):
        sport = int(packet["TCP"].sport)
        dport = int(packet["TCP"].dport)
    elif packet.haslayer("UDP"):
        sport = int(packet["UDP"].sport)
        dport = int(packet["UDP"].dport)
    a = (ip.src, sport)
    b = (ip.dst, dport)
    return (a, b, proto) if a <= b else (b, a, proto)


def read_pcap_sessions(path: str | Path) -> list[list[PacketRecord]]:
    try:
        from scapy.all import IP, Raw, rdpcap  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Scapy is required for pure-Python PCAP parsing.") from exc

    sessions: dict[tuple, list[PacketRecord]] = defaultdict(list)
    first_endpoint: dict[tuple, tuple[str, int]] = {}
    for packet in rdpcap(str(path)):
        key = _packet_flow_key(packet)
        if key is None or not packet.haslayer(IP):
            continue
        ip = packet[IP]
        sport = int(packet.sport) if hasattr(packet, "sport") else 0
        endpoint = (ip.src, sport)
        first_endpoint.setdefault(key, endpoint)
        direction = 1 if endpoint == first_endpoint[key] else -1
        payload = bytes(packet[Raw].load) if packet.haslayer(Raw) else bytes(packet.payload)
        sessions[key].append(
            PacketRecord(
                payload=payload,
                size=len(packet),
                direction=direction,
                timestamp=float(packet.time),
                position=len(sessions[key]),
            )
        )
    return [records for records in sessions.values() if records]
