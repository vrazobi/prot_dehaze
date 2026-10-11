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

    raise DehazingAPIError(f"Tipe output API tidak dikenali: {type(result).__name__}")


def run_dehazing(image: Image.Image):
    """Send the input to the hosted public Space and return (PIL image, info)."""
    try:
        from gradio_client import Client, handle_file
    except ImportError as exc:
        raise DehazingAPIError("Paket gradio_client belum terpasang. Deploy ulang agar requirements.txt dipasang.") from exc

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        image.convert("RGB").save(tmp_path, format="PNG")

        client = Client(SPACE_ID, verbose=False)
        try:
            # result = client.predict(tmp_path, api_name="/predict")
            result = client.predict(handle_file(tmp_path), fn_index=0)
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
