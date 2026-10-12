# DehazeFlow / Style Dehaze

This repository contains the AFINE-style dehazing training code used for paired haze removal experiments. The current training setup focuses on Glow-based invertible restoration with CBAM attention and a paired supervision objective:

- VGG feature/content consistency between restored output and ground truth
- pixel-level L1 loss between restored output and ground truth
- no style, reconstruction, smoothness, or SCM penalty for the included NH-HAZE/Dense-Haze configs

The repository also includes scripts for downstream RTTS object detection evaluation with YOLOv8.

## Project Structure

```text
.
├── train.py
├── inference.py
├── config/
│   ├── Dataset_NHHAZE_AFINE_8flow_3block_pixel_content.yaml
│   └── Dataset_DENSEHAZE_AFINE_8flow_3block_pixel_content.yaml
├── scripts/
│   ├── train_nhhaze.sh
│   └── train_densehaze.sh
├── experiments/
│   ├── rtts_yolov8/
│   └── report_tools/
├── dataset/
│   ├── trainA / trainB
│   ├── valA / valB
│   └── testA / testB
├── weight/              # downloaded checkpoints (not tracked by git)
├── examples/            # sample inference input and output
└── output/              # training/evaluation outputs (not tracked by git)
```

Large datasets, checkpoints, output folders, YOLO weights, and VGG weights are intentionally ignored by git.

## Dataset Format

Each paired dataset should follow this layout:

```text
Dataset_NAME/
├── trainA/   # hazy input images
├── trainB/   # clear / GT images
├── valA/     # optional hazy validation images
├── valB/     # optional validation GT images
├── testA/    # hazy test images
└── testB/    # test GT images
```

The training scripts create symlinks to datasets from the sibling `StyleTransferDehaze` repository when available:

```text
../StyleTransferDehaze/data/Dataset_NHHAZE
../StyleTransferDehaze/data/Dataset_DENSEHAZE
../StyleTransferDehaze/vgg_normalised.pth
```

## Training

Activate the environment first:

```bash
conda activate CenterNet
cd /home/alfin/Documents/ITB/TA/IEEE/style_dehaze
```

Train NH-HAZE:

```bash
bash scripts/train_nhhaze.sh
```

Train Dense-Haze:

```bash
bash scripts/train_densehaze.sh
```

Both included configs currently use:

```yaml
img_w: 256
img_h: 256
n_flow: 8
n_block: 3
batch_size: 1
content_weight: 1
p_weight: 1
style_weight: 0
recon_weight: 0
smooth_weight: 0
scm_weight: 0
```

Dense-Haze uses a smaller learning rate because the degradation is more severe and can be less stable during optimization.

## Training Outputs

Each run writes to:

```text
output/<job_name>_100_<n_flow>_<n_block>/
```

Important outputs:

```text
training.log
training_metrics.csv
validation_metrics.csv
test_metrics.csv
validation_latest.json
test_latest.json
best_metrics.json
plots/train_losses.png
plots/learning_rate.png
plots/psnr_curves.png
plots/ssim_curves.png
img_save/
test_save/
model_save/
```

Checkpoints:

```text
model_save/last.pth.tar
model_save/epoch_XXX_val.pth.tar
model_save/epoch_XXX_test.pth.tar
model_save/best_val_psnr.pth.tar
model_save/best_val_ssim.pth.tar
model_save/best_test_psnr.pth.tar
model_save/best_test_ssim.pth.tar
```

## Manual Training Command

The launcher scripts call `train.py` like this:

```bash
python train.py \
  --cfg_path ./config/Dataset_NHHAZE_AFINE_8flow_3block_pixel_content.yaml \
  --root_dir ./dataset/Dataset_NHHAZE \
  --job_name Dataset_NHHAZE_AFINE_8flow_3block_pixel_content \
  --no_wandb
```

Use `--no_wandb` for local runs without Weights & Biases.

## Pretrained Weights

Download the checkpoint for the dataset you want to use, then place it in the
`weight/` directory. Checkpoints are not tracked by Git.

| Dataset | Download | Inference configuration |
| --- | --- | --- |
| NH-HAZE | [Google Drive folder](https://drive.google.com/drive/folders/1qcMNpyB5qTYrhEcQa6kgx5uuSnXzuqXZ?usp=drive_link) | `config/Dataset_NHHAZE_AFINE_8flow_3block_pixel_content.yaml` |
| Dense-Haze | [Google Drive file](https://drive.google.com/file/d/1Fne-KuODgU7hVx8vY1XGYdy4l0WMjAAk/view?usp=drive_link) | `config/Dataset_DENSEHAZE_AFINE_8flow_3block_pixel_content.yaml` |

The default NH-HAZE checkpoint location is:

```text
weight/best_test_psnr.pth.tar
```

## Standalone Inference

Dehaze one image with the included NH-HAZE checkpoint:

```bash
python inference.py --input /path/to/hazy.jpg --output /path/to/dehazed.png --keep-size
```

Try the included NH-HAZE test image directly from the repository root:

```bash
python inference.py \
  --input examples/nhhaze_test_hazy.png \
  --output examples/nhhaze_test_dehazed.png \
  --keep-size \
  --device cpu
```

Both `--config` and `--checkpoint` default to the NH-HAZE configuration and
`weight/best_test_psnr.pth.tar`. To process a directory, provide
directory paths for `--input` and `--output`; its subdirectory layout is retained.
The model uses a compatible CUDA build when available and otherwise falls back to CPU;
use `--device cpu` to force CPU inference.

For Dense-Haze, point to its downloaded checkpoint explicitly:

```bash
python inference.py \
  --input examples/nhhaze_test_hazy.png \
  --output examples/densehaze_dehazed.png \
  --config config/Dataset_DENSEHAZE_AFINE_8flow_3block_pixel_content.yaml \
  --checkpoint weight/<downloaded-dense-checkpoint>.pth.tar \
  --keep-size
```

## RTTS YOLOv8 Downstream Evaluation

After training, evaluate the dehazed RTTS images with YOLOv8s/m/l:

```bash
DEHAZE_BATCH_SIZE=2 DEVICE=0 IMG_SIZE=250 \
bash experiments/rtts_yolov8/run_rtts_yolov8_comparison.sh \
  ./config/Dataset_NHHAZE_AFINE_8flow_3block_pixel_content.yaml \
  ./output/Dataset_NHHAZE_AFINE_8flow_3block_pixel_content_100_8_3/model_save/last.pth.tar
```

For Dense-Haze:

```bash
DEHAZE_BATCH_SIZE=2 DEVICE=0 IMG_SIZE=250 \
bash experiments/rtts_yolov8/run_rtts_yolov8_comparison.sh \
  ./config/Dataset_DENSEHAZE_AFINE_8flow_3block_pixel_content.yaml \
  ./output/Dataset_DENSEHAZE_AFINE_8flow_3block_pixel_content_100_8_3/model_save/last.pth.tar
```

The RTTS runner:

1. dehazes RTTS test images,
2. evaluates YOLOv8 on original hazy images,
3. evaluates YOLOv8 on dehazed images,
4. generates plots and markdown summaries.

Default paths:

```text
RTTS_ROOT=../StyleTransferDehaze/data/RTTS
DEHAZED_ROOT=./output/rtts_dehazed
RESULT_ROOT=./output/rtts_yolov8_comparison
YOLO_MODELS=../StyleTransferDehaze/yolov8s.pt ../StyleTransferDehaze/yolov8m.pt ../StyleTransferDehaze/yolov8l.pt
```

The summary is written to:

```text
output/rtts_yolov8_comparison/thesis_summary/
```

## Notes

- If training runs out of memory, keep `img_h/img_w` at `256` and `batch_size` at `1`.
- The dataset loader always resizes after augmentation, so rotated images still batch correctly.
- The RTTS dehazing script matches this repository's reference-free model path: `trainer.model(source)`.

## Citation

If you use this repository, cite the corresponding thesis/paper work:

```text
Style-Aware Dehazing via Glow-based Style Transfer with Attention Modulation
Alfin Nurhalim, Institut Teknologi Bandung
```
