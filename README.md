# AI Image Dehazing — Streamlit Starter

Prototype UI for an ITB AI Center exhibition. This starter deliberately **does not pretend to run AI inference** until the real pretrained model repository and compatible checkpoint are integrated.

## Current status

- [x] Streamlit UI, image upload and format/size validation
- [x] Before/after layout and PNG download flow (after a model adapter returns output)
- [x] Inference timing and basic file/dimension metadata
- [x] Optional PSNR/SSIM when a reference image is supplied and dimensions match
- [x] Explicit failure when the model is not configured
- [ ] Real pretrained model/checkpoint integration — required before public demo
- [ ] Actual benchmark on SOTS, O-HAZE, NH-HAZE, Dense-HAZE
- [ ] Measured RAM/VRAM and documented license/model provenance

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app will start, but inference will report that the model adapter is not configured. This is intentional.

## Integrate the actual model

Edit `inference_adapter.py` and call the model repository's real inference path. Before enabling the demo, verify:

1. Exact model architecture and source repository/commit.
2. Checkpoint file exists and is the correct checkpoint for that architecture.
3. Checkpoint loads without missing/unexpected-key errors (or explain expected keys).
4. Preprocessing, color order, normalization, resizing, and postprocessing match the original implementation.
5. A single test image runs end-to-end and output is finite, valid, and visually produced by the model.
6. Record the model name, version/commit, checkpoint provenance, license, device, and measured inference time.

Do not replace inference with a visual filter or a stored sample output.

## Deployment: Streamlit Community Cloud

1. Put the project in a GitHub repository.
2. Ensure `app.py`, `inference_adapter.py`, `metrics.py`, and `requirements.txt` are at the repository root.
3. Deploy the repository from Streamlit Community Cloud.
4. Add model code/checkpoint only if licensing permits and the checkpoint size/resource needs fit the host limits. Otherwise use an appropriately secured external inference service.
5. Test a real image after deployment; a successful page load is not proof that inference works.

## Privacy

On a hosted deployment, uploaded images are sent to the server running the app for processing. This code does not intentionally persist uploaded images, but temporary memory, hosting logs, crash reports, or platform-level handling may apply. Do not upload confidential images. Review the host's current data-handling terms before public deployment.

## Metrics and benchmark protocol

PSNR/SSIM are only computed when the user supplies a reference image with matching dimensions. Matching dimensions alone does not prove that the images are a valid ground-truth pair.

For dataset evaluation, use the official test split and correct paired ground truth for each dataset. Record checkpoint, preprocessing, input resolution, device, per-image inference time, peak RAM/VRAM where measurable, PSNR/SSIM aggregation method, and failures. Do not claim dataset results until actually measured.
