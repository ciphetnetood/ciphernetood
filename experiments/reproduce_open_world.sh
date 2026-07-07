python scripts/build_graphs.py --config configs/default.yaml --dataset iscx_vpn
python scripts/train_cfml.py --config configs/default.yaml --dataset iscx_vpn --open-world
python scripts/fit_cpg_ood.py --config configs/default.yaml --dataset iscx_vpn --checkpoint checkpoints/iscx_vpn/best.pt
python scripts/evaluate_open_world.py --config configs/default.yaml --dataset iscx_vpn --checkpoint checkpoints/iscx_vpn/best.pt --signatures checkpoints/iscx_vpn/cpg_signatures.pt
