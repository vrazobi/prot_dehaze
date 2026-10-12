import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def read_image_set(rtts_root, split, limit=None):
    split_path = Path(rtts_root) / 'ImageSets' / 'Main' / f'{split}.txt'
    stems = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    if limit is not None:
        stems = stems[:limit]
    return stems


def resolve_image(images_dir, stem):
    for ext in IMG_EXTS:
        path = Path(images_dir) / f'{stem}{ext}'
        if path.exists():
            return path
    raise FileNotFoundError(f'Could not find image for stem {stem} under {images_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--rtts-root', default='./data/RTTS')
    parser.add_argument('--split', default='test')
    parser.add_argument('--output-root', default='./output/rtts_dehazed')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    from model.trainers.trainer import Trainer
    from model.utils.utils import get_config

    cfg = get_config(args.config)
    cfg['cfg_path'] = args.config
    cfg['wandb'] = None

    trainer = Trainer(cfg)
    trainer.load_model(args.checkpoint, flow_only=False)
    trainer.model.eval()

    model_size = (cfg['img_w'], cfg['img_h'])
    images_dir = Path(args.rtts_root) / 'JPEGImages'
    out_dir = Path(args.output_root) / 'JPEGImages'
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = read_image_set(args.rtts_root, args.split, args.limit)

    pending = []
    skipped = 0
    for stem in stems:
        path = resolve_image(images_dir, stem)
        out_path = out_dir / path.name
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        pending.append((stem, path, out_path))

    if skipped:
        print(f'Skipping {skipped} existing dehazed images. Use --force to regenerate.')

    batch_size = max(1, args.batch_size)
    with torch.no_grad():
        for start in tqdm(range(0, len(pending), batch_size), desc=f'Dehaze RTTS {args.split}'):
            batch = pending[start:start + batch_size]
            originals = []
            tensors = []
            for _, path, _ in batch:
                original = Image.open(path).convert('RGB')
                source = original.resize(model_size, Image.BICUBIC)
                originals.append(original)
                tensors.append(transforms.ToTensor()(source))

            source_tensor = torch.stack(tensors, dim=0).cuda()
            outputs = trainer.model(source_tensor)
            outputs = torch.clamp(outputs, 0, 1).cpu()

            for output, original, (_, _, out_path) in zip(outputs, originals, batch):
                output = torch.nn.functional.interpolate(
                    output.unsqueeze(0),
                    size=(original.height, original.width),
                    mode='bilinear',
                    align_corners=False,
                )
                save_image(output[0], out_path)

    print(f'Saved {len(pending)} new dehazed images to {out_dir}')
    print(f'Total requested images: {len(stems)}')


if __name__ == '__main__':
    main()
