# AI Image Dehazing — Streamlit + Remote Model API

## What's different
This version does not load local PyTorch checkpoints. It sends the uploaded image to a public Gradio/Hugging Face Space and returns the model's actual output. This avoids local checkpoint-version and GPU configuration issues.

**Remote model:** [GAN-based Image Dehazing Demo](https://huggingface.co/spaces/saravanaKumarT/gan_dehaze)

The model's published demo description focuses on satellite optical imagery. General photography results can vary. The API is operated by a third party, so availability, queue times, limits, and endpoints can change.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Publish on Streamlit Community Cloud

1. Push these files to the root of a GitHub repo.
2. Create an app at https://share.streamlit.io/
3. Select the repository, `main` branch, and `app.py` as main file.
4. Deploy and test with a non-sensitive image.

## Data privacy
Uploaded images are transmitted to the external model Space for processing. Do not upload private/confidential pictures. This application does not deliberately persist the uploaded file; however, the external service may handle requests under its own policies and temporary storage rules.

## Metrik and claims
PSNR/SSIM are shown only when the user uploads a reference image with matching dimensions. Matching dimensions alone does not prove it is a valid ground-truth pair. No benchmark results for SOTS/O-HAZE/NH-HAZE/Dense-HAZE are claimed by this starter.

## Operational caveats
- The public Space can sleep, queue, hit limits, or change its API.
- `Waktu round-trip API` includes upload/network/queue/server processing/download; it is not pure model inference time.
- If the API fails, check the linked Space's status and use `client.view_api()` to inspect the current endpoint.
