"""Optional image-quality metrics. Requires valid, aligned reference image."""
import numpy as np
from PIL import Image


def calculate_psnr_ssim(output: Image.Image, reference: Image.Image):
    if output.size != reference.size:
        raise ValueError("Dimensi output dan reference harus sama.")
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except ImportError as exc:
        raise RuntimeError("Paket scikit-image belum terpasang.") from exc

    out = np.asarray(output.convert("RGB"), dtype=np.uint8)
    ref = np.asarray(reference.convert("RGB"), dtype=np.uint8)
    psnr = float(peak_signal_noise_ratio(ref, out, data_range=255))
    ssim = float(structural_similarity(ref, out, channel_axis=2, data_range=255))
    return psnr, ssim
