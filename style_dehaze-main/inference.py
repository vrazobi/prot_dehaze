#!/usr/bin/env python3
"""Run a trained Style Dehaze checkpoint on one image or a directory of images."""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


class InferenceModel(nn.Module):
    """The checkpoint's restoration network, without training-only dependencies."""

    def __init__(self, cfg):
        super().__init__()
        from model.network.glow import Glow

        self.glow = Glow(
            in_channel=3,
            n_flow=cfg['n_flow'],
            n_block=cfg['n_block'],
            affine=cfg['affine'],
            conv_lu=not cfg['no_lu'],
            alpha=cfg.get('attention') is not None,
        )

    def forward(self, content_images):
        latent = self.glow(content_images, forward=True)
        return self.glow(latent, forward=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='An image file or directory of images.')
    parser.add_argument('--output', required=True, type=Path, help='Output image file or directory.')
    parser.add_argument(
        '--config',
        type=Path,
        default=REPO_ROOT / 'config/Dataset_NHHAZE_AFINE_8flow_3block_pixel_content.yaml',
        help='Model architecture configuration.',
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=REPO_ROOT / 'weight/best_test_psnr.pth.tar',
        help='Checkpoint to load.',
    )
    parser.add_argument('--device', default=None, help='cuda, cpu, or a CUDA device such as cuda:0.')
    parser.add_argument('--keep-size', action='store_true', help='Resize restored images back to their input size.')
    return parser.parse_args()


def image_paths(input_path):
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f'Unsupported input image type: {input_path.suffix}')
        return [input_path], input_path.parent
    if input_path.is_dir():
        paths = sorted(path for path in input_path.rglob('*') if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            raise ValueError(f'No supported images found under {input_path}')
        return paths, input_path
    raise FileNotFoundError(f'Input not found: {input_path}')


def output_path_for(input_path, root, output):
    if input_path.is_file() and output.suffix:
        return output
    return output / input_path.relative_to(root)


def default_device(requested_device):
    if requested_device:
        return torch.device(requested_device)
    if not torch.cuda.is_available():
        return torch.device('cpu')
    capability = torch.cuda.get_device_capability()
    required_arch = f'sm_{capability[0]}{capability[1]}'
    if required_arch not in torch.cuda.get_arch_list():
        print(f'CUDA device {required_arch} is unsupported by this PyTorch build; using CPU.')
        return torch.device('cpu')
    return torch.device('cuda')


def main():
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f'Config not found: {args.config}')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    from model.utils.utils import get_config, remove_prefix

    device = default_device(args.device)
    cfg = get_config(args.config)
    model = InferenceModel(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = remove_prefix(checkpoint['state_dict'], 'module.')
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    inputs, root = image_paths(args.input)
    if args.input.is_file() and not args.output.suffix:
        args.output.mkdir(parents=True, exist_ok=True)
    model_size = (cfg['img_w'], cfg['img_h'])
    to_tensor = transforms.ToTensor()

    with torch.inference_mode():
        for input_path in tqdm(inputs, desc='Dehazing'):
            original = Image.open(input_path).convert('RGB')
            source = original.resize(model_size, Image.BICUBIC)
            tensor = to_tensor(source).unsqueeze(0).to(device)
            restored = torch.clamp(model(tensor), 0, 1).cpu()
            if args.keep_size:
                restored = torch.nn.functional.interpolate(
                    restored, size=(original.height, original.width), mode='bilinear', align_corners=False
                )
            output_path = output_path_for(args.input, root, args.output) if args.input.is_file() else output_path_for(input_path, root, args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(restored[0], output_path)

    print(f'Saved {len(inputs)} dehazed image(s) to {args.output}')


if __name__ == '__main__':
    main()
