"""Remote inference adapter using a publicly hosted Gradio dehazing demo.

The app does not load any checkpoint locally. Uploaded image data is sent to the
external Hugging Face Space, whose uptime/quotas/API may change.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

MODEL_NAME = "GAN-based Image Dehazing Demo (remote hosted API)"
MODEL_SOURCE = "https://huggingface.co/spaces/saravanaKumarT/gan_dehaze"
SPACE_ID = "saravanaKumarT/gan_dehaze"


class DehazingAPIError(RuntimeError):
    pass


def _to_image(result) -> Image.Image:
    """Convert common Gradio image response formats into a PIL image."""
    if isinstance(result, (list, tuple)):
        if not result:
            raise DehazingAPIError("API mengembalikan hasil kosong.")
        result = result[0]

    if isinstance(result, Image.Image):
        return result.convert("RGB")

    if isinstance(result, dict):
        # Gradio versions can return {path, url, ...} for file outputs.
        local_path = result.get("path")
        url = result.get("url")
        if local_path and Path(str(local_path)).exists():
            return Image.open(str(local_path)).convert("RGB")
        if url:
            result = url
        else:
            raise DehazingAPIError(f"Format response gambar tidak dikenali: {list(result.keys())}")

    if isinstance(result, (str, os.PathLike)):
        value = str(result)
        if value.startswith("http://") or value.startswith("https://"):
            with urlopen(value, timeout=60) as response:
                data = response.read()
            from io import BytesIO
            return Image.open(BytesIO(data)).convert("RGB")
        path = Path(value)
        if path.exists():
            return Image.open(path).convert("RGB")

<<<<<<< HEAD
    raise DehazingAPIError(f"Tipe output API tidak dikenali: {type(result).__name__}")
=======
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
>>>>>>> f97eb423e7c99f857790a58246f975354ed154be


def run_dehazing(image: Image.Image):
    """Send the input to the hosted public Space and return (PIL image, info)."""
    try:
        from gradio_client import Client
    except ImportError as exc:
        raise DehazingAPIError("Paket gradio_client belum terpasang. Deploy ulang agar requirements.txt dipasang.") from exc

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        image.convert("RGB").save(tmp_path, format="PNG")

        client = Client(SPACE_ID, verbose=False)
        try:
            result = client.predict(tmp_path, api_name="/predict")
        except Exception as first_error:
            # Ask Gradio for endpoint details to aid debugging; do not silently use fake output.
            try:
                api_details = client.view_api(print_info=False)
                raise DehazingAPIError(
                    f"Gagal memanggil endpoint /predict. Cek API demo; detail endpoint: {api_details}. "
                    f"Error awal: {first_error}"
                ) from first_error
            except DehazingAPIError:
                raise
            except Exception:
                raise DehazingAPIError(
                    f"Panggilan API /predict gagal: {first_error}. API publik mungkin sedang tidur, antre, atau berubah."
                ) from first_error

        output = _to_image(result)
        info = {
            "name": MODEL_NAME,
            "device": "remote hosted service (device details not exposed)",
            "source": MODEL_SOURCE,
        }
        return output, info
    except DehazingAPIError:
        raise
    except Exception as exc:
        raise DehazingAPIError(
            f"Tidak dapat menghubungi Space {SPACE_ID}: {type(exc).__name__}: {exc}. "
            "Pastikan Space publik masih aktif dan API-nya tersedia."
        ) from exc
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
