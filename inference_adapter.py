"""AOD-Net inference adapter using the public pretrained checkpoint.

Model code/checkpoint source: https://github.com/Huangshifu628/aod-net
Repository declares MIT license. Checkpoint is downloaded from its public
`snapshots/dehazer.pth` file on first inference; it is cached in a temp folder.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tempfile
import time
import urllib.request

import numpy as np
from PIL import Image


CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/Huangshifu628/aod-net/main/snapshots/dehazer.pth"
)
MODEL_REPO = "https://github.com/Huangshifu628/aod-net"


class ModelUnavailableError(RuntimeError):
    """Raised when the pretrained model/checkpoint cannot be loaded."""


def _download_checkpoint() -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "aodnet_pretrained"
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cache_dir / "dehazer.pth"

    if checkpoint.exists() and checkpoint.stat().st_size > 1024:
        return checkpoint

    temp_checkpoint = cache_dir / "dehazer.pth.part"
    try:
        request = urllib.request.Request(
            CHECKPOINT_URL,
            headers={"User-Agent": "AI-Image-Dehazing-Streamlit-Demo/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response, temp_checkpoint.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if not temp_checkpoint.exists() or temp_checkpoint.stat().st_size <= 1024:
            raise RuntimeError("File checkpoint kosong atau ukurannya tidak valid.")
        temp_checkpoint.replace(checkpoint)
        return checkpoint
    except Exception as exc:
        try:
            temp_checkpoint.unlink(missing_ok=True)
        except OSError:
            pass
        raise ModelUnavailableError(
            "Tidak bisa mengunduh checkpoint pretrained AOD-Net. Pastikan server punya "
            "akses internet ke raw.githubusercontent.com. Sumber checkpoint: " + CHECKPOINT_URL
        ) from exc


class AODNet:
    """Namespace only; actual torch.nn.Module is built lazily below."""


@lru_cache(maxsize=1)
def _load_model():
    # Import torch only when the user first runs inference, avoiding a heavy import
    # on the initial page render where possible.
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise ModelUnavailableError(
            "PyTorch belum tersedia. Pastikan `torch` tercantum di requirements.txt "
            "dan proses instalasi dependency deployment selesai."
        ) from exc

    class AODNetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU(inplace=True)
            self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
            self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
            self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
            self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
            self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)

        def forward(self, x):
            x1 = self.relu(self.e_conv1(x))
            x2 = self.relu(self.e_conv2(x1))
            x3 = self.relu(self.e_conv3(torch.cat((x1, x2), dim=1)))
            x4 = self.relu(self.e_conv4(torch.cat((x2, x3), dim=1)))
            x5 = self.relu(self.e_conv5(torch.cat((x1, x2, x3, x4), dim=1)))
            clean = self.relu((x5 * x) - x5 + 1)
            return clean

    start_load = time.perf_counter()
    checkpoint_path = _download_checkpoint()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        # The public checkpoint is a state_dict, so weights_only=True is preferred.
       
        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        
        except TypeError:
            # Kompatibilitas dengan PyTorch versi lama
            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
            )
        
        except Exception as safe_load_error:
            # Fallback hanya untuk checkpoint AOD-Net dari sumber
            # yang memang sudah diperiksa dan dipercaya.
            if "Weights only load failed" not in str(safe_load_error):
                raise ValueError('tolol')
        
            try:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except Exception as legacy_load_error:
                raise RuntimeError(
                    "Checkpoint gagal dimuat dengan loader aman maupun "
                    "loader kompatibilitas. "
                    f"Safe loader: {safe_load_error}. "
                    f"Compatibility loader: {legacy_load_error}."
                ) from legacy_load_error

        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
        model = AODNetModel()
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        # Keep CPU threading modest on free hosted instances.
        if device.type == "cpu":
            torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    except Exception as exc:
        raise ModelUnavailableError(
            "Checkpoint AOD-Net berhasil ditemukan, tetapi tidak kompatibel dengan arsitektur. "
            "Periksa sumber checkpoint/model. Detail: " + str(exc)
        ) from exc

    return model, device, time.perf_counter() - start_load, torch


def run_dehazing(image: Image.Image):
    """Run actual AOD-Net inference and return (PIL output, metadata)."""
    if not isinstance(image, Image.Image):
        raise TypeError("Input harus berupa PIL.Image.Image.")
    if image.width * image.height > 2_500_000:
        raise ValueError(
            "Gambar terlalu besar untuk demo gratis CPU. Maksimum 2,5 megapiksel; "
            "resize gambar sebelum upload."
        )

    model, device, load_seconds, torch = _load_model()
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    start_inference = time.perf_counter()
    with torch.inference_mode():
        output = model(tensor)
    inference_seconds = time.perf_counter() - start_inference

    output = output.squeeze(0).detach().float().cpu().clamp(0.0, 1.0)
    output_np = output.permute(1, 2, 0).numpy()
    output_u8 = np.round(output_np * 255.0).astype(np.uint8)
    output_image = Image.fromarray(output_u8, mode="RGB")

    return output_image, {
        "name": "AOD-Net (pretrained)",
        "device": str(device),
        "source": MODEL_REPO,
        "checkpoint": CHECKPOINT_URL,
        "license": "MIT (repository declares MIT; retain attribution and verify terms for intended use)",
        "inference_seconds": inference_seconds,
        "model_load_seconds": load_seconds,
        "checkpoint_source": "public repository checkpoint; no local training performed",
    }
