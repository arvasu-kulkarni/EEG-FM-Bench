#!/bin/bash

cd "$(dirname "$0")/.." || exit

# Example: traditional visualization (t-SNE / Grad-CAM / IG)
PYTHONPATH=$PWD python plot_vis.py \
  t_sne \
  assets/conf/baseline/csbrain/csbrain_unified.yaml \
  plot/configs/example/tsne_config_csbrain.yaml

# Example: head pooling + MLP comparison
# PYTHONPATH=$PWD python plot_vis.py \
#   head_mlp \
#   assets/conf/baseline/cbramod/cbramod_eval.yaml \
#   plot/configs/example/head_mlp_config.yaml


