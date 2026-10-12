
"""Local inference adapter for the Style Dehaze checkpoint."""

import os
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

MODEL_NAME = "Style Dehaze (local checkpoint)"

# Default: file adapter berada di root repository Style Dehaze.
# REPO_ROOT = Path(
#     os.getenv("STYLE_DEHAZE_ROOT", str(Path(__file__).resolve().parent))
# )

REPO_ROOT = Path(__file__).resolve().parent

CONFIG_PATH = Path(
    os.getenv(
        "STYLE_DEHAZE_CONFIG",
        str(REPO_ROOT / "config/Dataset_NHHAZE_AFINE_8flow_3block_pixel_content.yaml"),
    )
)

CHECKPOINT_PATH = Path(
    os.getenv(
        "STYLE_DEHAZE_CHECKPOINT",
        str(REPO_ROOT / "weight/best_test_psnr.pth.tar"),
    )
)


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

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
def load_model():
    """Load the local architecture and checkpoint once."""
    if not CONFIG_PATH.is_file():
        raise DehazingAPIError(
            f"Config tidak ditemukan: {CONFIG_PATH}"
        )

    if not CHECKPOINT_PATH.is_file():
        raise DehazingAPIError(
            f"Checkpoint tidak ditemukan: {CHECKPOINT_PATH}"
        )

    try:
        from model.utils.utils import get_config, remove_prefix
        from model.network.glow import Glow
        from torch import nn

        class InferenceModel(nn.Module):
            def __init__(self, cfg):
                super().__init__()
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

    except DehazingAPIError:
        raise
    except Exception as exc:
        raise DehazingAPIError(
            f"Gagal memuat model Style Dehaze: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


@lru_cache(maxsize=1)
def get_model():
    # Cache prevents reloading the checkpoint for every uploaded image.
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

            # Restore the original input dimensions.
            if restored.shape[-2:] != (
                original.height, original.width
            ):
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



# """Remote inference adapter using a publicly hosted Gradio dehazing demo.

# The app does not load any checkpoint locally. Uploaded image data is sent to the
# external Hugging Face Space, whose uptime/quotas/API may change.
# """
# from __future__ import annotations

# import os
# import tempfile
# from pathlib import Path
# from urllib.request import urlopen

# from PIL import Image

# MODEL_NAME = "GAN-based Image Dehazing Demo (remote hosted API)"
# MODEL_SOURCE = "https://huggingface.co/spaces/saravanaKumarT/gan_dehaze"
# SPACE_ID = "saravanaKumarT/gan_dehaze"


# class DehazingAPIError(RuntimeError):
#     pass


# def _to_image(result) -> Image.Image:
#     """Convert common Gradio image response formats into a PIL image."""
#     if isinstance(result, (list, tuple)):
#         if not result:
#             raise DehazingAPIError("API mengembalikan hasil kosong.")
#         result = result[0]

#     if isinstance(result, Image.Image):
#         return result.convert("RGB")

#     if isinstance(result, dict):
#         # Gradio versions can return {path, url, ...} for file outputs.
#         local_path = result.get("path")
#         url = result.get("url")
#         if local_path and Path(str(local_path)).exists():
#             return Image.open(str(local_path)).convert("RGB")
#         if url:
#             result = url
#         else:
#             raise DehazingAPIError(f"Format response gambar tidak dikenali: {list(result.keys())}")

#     if isinstance(result, (str, os.PathLike)):
#         value = str(result)
#         if value.startswith("http://") or value.startswith("https://"):
#             with urlopen(value, timeout=60) as response:
#                 data = response.read()
#             from io import BytesIO
#             return Image.open(BytesIO(data)).convert("RGB")
#         path = Path(value)
#         if path.exists():
#             return Image.open(path).convert("RGB")

#     raise DehazingAPIError(f"Tipe output API tidak dikenali: {type(result).__name__}")


# def run_dehazing(image: Image.Image):
#     """Send the input to the hosted public Space and return (PIL image, info)."""
#     try:
#         from gradio_client import Client
#     except ImportError as exc:
#         raise DehazingAPIError("Paket gradio_client belum terpasang. Deploy ulang agar requirements.txt dipasang.") from exc

#     tmp_path = None
#     try:
#         with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
#             tmp_path = tmp.name
#         image.convert("RGB").save(tmp_path, format="PNG")

#         client = Client(SPACE_ID, verbose=False)
#         try:
#             # result = client.predict(tmp_path, api_name="/predict")
#             result = client.predict(tmp_path, fn_index=0)
#         except Exception as first_error:
#             # Ask Gradio for endpoint details to aid debugging; do not silently use fake output.
#             try:
#                 api_details = client.view_api(print_info=False)
#                 raise DehazingAPIError(
#                     f"Gagal memanggil endpoint /predict. Cek API demo; detail endpoint: {api_details}. "
#                     f"Error awal: {first_error}"
#                 ) from first_error
#             except DehazingAPIError:
#                 raise
#             except Exception:
#                 raise DehazingAPIError(
#                     f"Panggilan API /predict gagal: {first_error}. API publik mungkin sedang tidur, antre, atau berubah."
#                 ) from first_error

#         output = _to_image(result)
#         info = {
#             "name": MODEL_NAME,
#             "device": "remote hosted service (device details not exposed)",
#             "source": MODEL_SOURCE,
#         }
#         return output, info
#     except DehazingAPIError:
#         raise
#     except Exception as exc:
#         raise DehazingAPIError(
#             f"Tidak dapat menghubungi Space {SPACE_ID}: {type(exc).__name__}: {exc}. "
#             "Pastikan Space publik masih aktif dan API-nya tersedia."
#         ) from exc
#     finally:
#         if tmp_path:
#             try:
#                 Path(tmp_path).unlink(missing_ok=True)
#             except OSError:
#                 pass
