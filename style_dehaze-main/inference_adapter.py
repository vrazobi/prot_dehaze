
"""Local inference adapter for Style Dehaze."""

import os
import sys
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch import nn
import gdown

CHECKPOINT_PATH = Path(
    os.getenv(
        "STYLE_DEHAZE_CHECKPOINT",
        str(REPO_ROOT / "weights" / "best_test_psnr.pth"),
    )
)

DRIVE_CHECKPOINT_URL = "https://drive.google.com/file/d/1332m5gGXRFGQ8sLbjXYBbmCJagRgTsQj/view?usp=drive_link"
def ensure_checkpoint():
    if CHECKPOINT_PATH.is_file():
        return

    CHECKPOINT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    downloaded = gdown.download(
        url=DRIVE_CHECKPOINT_URL,
        output=str(CHECKPOINT_PATH),
        quiet=False,
        fuzzy=True,
    )

    if not downloaded or not CHECKPOINT_PATH.is_file():
        raise DehazingAPIError(
            "Gagal mengunduh checkpoint dari Google Drive."
        )



MODEL_NAME = "Style Dehaze (local checkpoint)"

# File ini berada di dalam folder style_dehaze-main.
REPO_ROOT = Path(__file__).resolve().parent

CONFIG_PATH = Path(
    os.getenv(
        "STYLE_DEHAZE_CONFIG",
        str(
            REPO_ROOT
            / "config"
            / "Dataset_NHHAZE_AFINE_8flow_3block_pixel_content.yaml"
        ),
    )
)

CHECKPOINT_PATH = Path(
    os.getenv(
        "STYLE_DEHAZE_CHECKPOINT",
        str(REPO_ROOT / "weights" / "best_test_psnr.pth"),
    )
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class DehazingAPIError(RuntimeError):
    pass


def default_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")

    capability = torch.cuda.get_device_capability()
    required_arch = f"sm_{capability[0]}{capability[1]}"

    if required_arch not in torch.cuda.get_arch_list():
        return torch.device("cpu")

    return torch.device("cuda")


class InferenceModel(nn.Module):
    """Restoration network matching inference.py."""

    def __init__(self, cfg):
        super().__init__()

        from model.network.glow import Glow

        self.glow = Glow(
            in_channel=3,
            n_flow=cfg["n_flow"],
            n_block=cfg["n_block"],
            affine=cfg["affine"],
            conv_lu=not cfg["no_lu"],
            alpha=cfg.get("attention") is not None,
        )

    def forward(self, content_images):
        latent = self.glow(content_images, forward=True)
        return self.glow(latent, forward=False)


def load_model():
    if not CONFIG_PATH.is_file():
        raise DehazingAPIError(
            f"Config tidak ditemukan: {CONFIG_PATH}"
        )
    ensure_checkpoint()
    if not CHECKPOINT_PATH.is_file():
        raise DehazingAPIError(
            f"Checkpoint tidak ditemukan: {CHECKPOINT_PATH}"
        )

    try:
        from model.utils.utils import get_config, remove_prefix

        device = default_device()
        cfg = get_config(CONFIG_PATH)

        model = InferenceModel(cfg).to(device)

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )

        state_dict = remove_prefix(
            checkpoint["state_dict"], "module."
        )

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        return model, cfg, device

    except Exception as exc:
        raise DehazingAPIError(
            f"Gagal memuat model: {type(exc).__name__}: {exc}"
        ) from exc


@lru_cache(maxsize=1)
def get_model():
    return load_model()


def run_dehazing(image: Image.Image):
    """Return (dehazed PIL image, model information)."""

    try:
        model, cfg, device = get_model()

        original = image.convert("RGB")
        original_size = original.size

        model_size = (cfg["img_w"], cfg["img_h"])
        source = original.resize(model_size, Image.BICUBIC)

        tensor = transforms.ToTensor()(source)
        tensor = tensor.unsqueeze(0).to(device)

        with torch.inference_mode():
            restored = torch.clamp(model(tensor), 0, 1)

            restored = F.interpolate(
                restored,
                size=(original.height, original.width),
                mode="bilinear",
                align_corners=False,
            )

        output = transforms.ToPILImage()(restored[0].cpu())

        info = {
            "name": MODEL_NAME,
            "device": str(device),
            "source": "Local Style Dehaze checkpoint",
            "config": str(CONFIG_PATH),
            "checkpoint": str(CHECKPOINT_PATH),
            "input_size": original_size,
            "model_size": model_size,
        }

        return output, info

    except DehazingAPIError:
        raise
    except Exception as exc:
        raise DehazingAPIError(
            f"Inference gagal: {type(exc).__name__}: {exc}"
        ) from exc
