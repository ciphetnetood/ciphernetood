# CipherNet-OOD
![Alt text](assets/fig_model.png?raw=true "Model")

Implementation of **CipherNet-OOD: Open-World Encrypted Traffic Classification via CipherFlow Manifold Learning and CipherProtoGram Detection under Class Imbalance**.

The repository implements the three-stage framework described in the manuscript:

1. Session representation from encrypted traffic captures.
2. CipherFlow Manifold Learning (CFML) with traffic-mode prototypes and traffic-conditioned angular margins.
3. CipherProtoGram OOD Detection (CPG-OOD) using prototype-conditioned Gram signatures.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Install the PyTorch Geometric wheels matching your CUDA/PyTorch environment if the generic install does not resolve them automatically.

## Data Layout

Raw captures are not tracked by git.

```text
data/raw/<dataset>/<class_name>/*.pcap|*.pcapng
data/sessions/<dataset>/...
data/processed/<dataset>/*.pt
data/splits/<dataset>/*.json
```

## Quick Start

Build graphs from PCAP files:

```bash
python scripts/build_graphs.py --config configs/default.yaml --dataset iscx_vpn
```

Train CFML:

```bash
python scripts/train_cfml.py --config configs/default.yaml --dataset iscx_vpn
```

Fit CipherProtoGram signatures:

```bash
python scripts/fit_cpg_ood.py --config configs/default.yaml --dataset iscx_vpn --checkpoint checkpoints/iscx_vpn/best.pt
```

Evaluate closed-set classification:

```bash
python scripts/evaluate_closed_set.py --config configs/default.yaml --dataset iscx_vpn --checkpoint checkpoints/iscx_vpn/best.pt
```

Evaluate open-world OOD detection:

```bash
python scripts/evaluate_open_world.py --config configs/default.yaml --dataset iscx_vpn --checkpoint checkpoints/iscx_vpn/best.pt --signatures checkpoints/iscx_vpn/cpg_signatures.pt
```

## Reproducibility Defaults

The default config follows the manuscript implementation details: `P=25`, hidden dimension `64`, `5` GraphTransformer layers, `3` prototypes per class, Adam with learning rate `1e-4`, weight decay `1e-5`, batch size `32`, up to `500` epochs, and early stopping patience `10`.

## Notes

This code removes shortcut identifiers from the feature representation. It does not decrypt payloads. Dataset licenses vary; download each benchmark dataset from its official source.
