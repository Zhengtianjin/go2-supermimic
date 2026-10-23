"""Train a chase / dodge policy with SuperMimic.

Flat ground, from scratch:
  python scripts/train.py --task Go2-Chase-Flat --run-name flat

Obstacles or terrain, warm-started from a flat policy (see README.md for the
terrain recipe with the MoE-CTS and flat-policy data drivers):
  python scripts/train.py --task Go2-Chase-Obstacles --resume <flat model.pt> --run-name obstacles

Every TrainCfg field (go2_supermimic/supermimic/train.py) is a flag.
"""

import torch

from go2_supermimic.supermimic.train import main

if __name__ == "__main__":
  torch.set_float32_matmul_precision("high")
  main()
