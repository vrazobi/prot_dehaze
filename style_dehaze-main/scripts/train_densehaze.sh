#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p dataset

if [ ! -e dataset/Dataset_DENSEHAZE ]; then
  ln -s ../../StyleTransferDehaze/data/Dataset_DENSEHAZE dataset/Dataset_DENSEHAZE
fi

if [ ! -e vgg_normalised.pth ]; then
  ln -s ../StyleTransferDehaze/vgg_normalised.pth vgg_normalised.pth
fi

python - <<'PY'
from model.utils.utils import get_config
cfg = get_config('./config/Dataset_DENSEHAZE_AFINE_8flow_3block_pixel_content.yaml')
assert cfg['img_w'] == 256 and cfg['img_h'] == 256, cfg
assert cfg['batch_size'] == 1, cfg
print('Using image size: 256x256, batch size: 1')
PY

python train.py \
  --cfg_path ./config/Dataset_DENSEHAZE_AFINE_8flow_3block_pixel_content.yaml \
  --root_dir ./dataset/Dataset_DENSEHAZE \
  --job_name Dataset_DENSEHAZE_AFINE_8flow_3block_pixel_content \
  --no_wandb
