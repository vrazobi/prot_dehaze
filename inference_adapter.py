"""Adapter inference: sambungkan fungsi ini ke repository model dehazing yang sebenarnya.

Tidak ada filter fallback dan tidak ada output palsu. Sampai model/checkpoint asli
diintegrasikan, fungsi akan gagal secara eksplisit.
"""
from PIL import Image


class ModelUnavailableError(RuntimeError):
    """Raised when the actual pretrained model is not configured."""


def run_dehazing(image: Image.Image):
    """Return (output_pil_image, model_info_dict).

    Replace this function with the repository's actual preprocessing -> model
    inference -> postprocessing pipeline. Keep the model's original resize,
    normalization, color order, and checkpoint-loading logic.

    Required return:
        output: PIL.Image.Image
        model_info: e.g. {"name": "model name/version", "device": "cpu"}
    """
    raise ModelUnavailableError(
        "Belum ada adapter model yang terhubung. Diperlukan repository model, "
        "kode inference/preprocessing, dependensi yang sesuai, dan checkpoint "
        "pretrained yang tervalidasi. Tidak ada hasil dehazing yang dibuat."
    )
