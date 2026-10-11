# AI Image Dehazing — Streamlit + AOD-Net

Prototype UI for an ITB AI Center exhibition, using a public pretrained AOD-Net checkpoint. This app does **not** train a model and does not substitute a filter or sample image for inference.

## Model provenance

- **Model:** AOD-Net (All-in-One Network for Dehazing)
- **Paper:** Boyi Li et al., *AOD-Net: All-in-One Dehazing Network*, ICCV 2017. https://arxiv.org/abs/1707.06543
- **Public model/code repository:** https://github.com/Huangshifu628/aod-net
- **Architecture source:** `net.py` in that repository; the architecture is reproduced in `inference_adapter.py` with the matching parameter names.
- **Pretrained checkpoint:** `snapshots/dehazer.pth`, downloaded from https://raw.githubusercontent.com/Huangshifu628/aod-net/main/snapshots/dehazer.pth on first inference and cached in a temporary directory.
- **License:** the repository declares MIT. Preserve attribution, and verify model/checkpoint licensing for your particular exhibition/commercial use.
- **Version:** repository `main` branch (not pinned to a commit); checkpoint is fetched at runtime. For reproducible scientific use, pin a commit and record the checkpoint SHA256.
- **Training:** this application uses the published pretrained checkpoint; no training is performed by the app.

## Features

- Upload PNG/JPEG/WEBP (up to 10 MB; max 2.5 megapixels for free CPU hosting)
- Actual AOD-Net inference using PyTorch
- Before/after images and PNG download
- Input/output dimensions, input filesize, measured forward-pass time, model load/download time
- Optional PSNR/SSIM only when a reference image is provided and dimensions match
- Error messages for invalid upload, missing checkpoint, incompatible weights, and oversized images

## Run locally on Windows PowerShell

```powershell
cd C:\path\to\project
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The first inference downloads the pretrained checkpoint, so the machine needs internet access to `raw.githubusercontent.com`. The UI page may open before the model is downloaded; press **Jalankan dehazing** to load and run it.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `inference_adapter.py`, `metrics.py`, `requirements.txt`, `.streamlit/config.toml`, and this README to the root of a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repo/branch and set main file to `app.py`.
3. Deploy. PyTorch installation and free-host resource limits may affect build time and performance.
4. Test a real hazy image after deployment. A successful page render is not by itself proof that inference works.
5. If the hosting service cannot download the checkpoint at runtime, manually download the checkpoint from the URL above and adjust `inference_adapter.py` to use a committed or externally hosted local checkpoint, while respecting the source license and repository size limits.

## Metrics and benchmark protocol

PSNR/SSIM are only shown when the user provides a reference image with matching dimensions. Same dimensions alone do not prove a valid ground-truth pairing. This UI has not independently benchmarked SOTS, O-HAZE, NH-HAZE, or Dense-HAZE. To report benchmark results, use the official paired test split and document image pairing, preprocessing, checkpoint, CPU/GPU, inference timing, peak RAM/VRAM where measurable, and aggregation method.

## Limitations

- AOD-Net is a compact, older dehazing model; results on real-world fog, strong haze, night scenes, or uneven illumination may be limited.
- Free Streamlit hosting is CPU/RAM constrained; the UI limits input to 2.5 megapixels.
- No server-side permanent storage is intentionally implemented. Uploaded images are sent to the server for processing and may exist in process memory or hosting temporary infrastructure/logs according to the host's policies. Do not upload confidential images.
- The app records actual inference timing but does not currently measure peak RAM/VRAM.
