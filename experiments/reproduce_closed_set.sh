python scripts/build_graphs.py --config configs/default.yaml --dataset iscx_vpn
python scripts/train_cfml.py --config configs/default.yaml --dataset iscx_vpn
python scripts/evaluate_closed_set.py --config configs/default.yaml --dataset iscx_vpn --checkpoint checkpoints/iscx_vpn/best.pt
