from __future__ import annotations

import io
import time
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st


import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "style_dehaze-main"

sys.path.insert(0, str(MODEL_DIR))

from inference_adapter import (
    run_dehazing,
    DehazingAPIError,
    MODEL_NAME,
)



from metrics import calculate_psnr_ssim

import gdown


# DRIVE_DIR = Path("drive_images")


APP_TITLE = "AI Image Dehazing"
MAX_UPLOAD_MB = 10

st.set_page_config(page_title=APP_TITLE, page_icon="🌫️", layout="wide")



DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1H8gAkak2guCgxrlx53M7c-29pnTh3o51"
)

DRIVE_DIR = BASE_DIR / "drive_images"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@st.cache_data
def download_drive_images():
    import gdown

    DRIVE_DIR.mkdir(parents=True, exist_ok=True)

    gdown.download_folder(
        url=DRIVE_FOLDER_URL,
        output=str(DRIVE_DIR),
        quiet=True,
        use_cookies=False,
    )

    return sorted(
        p for p in DRIVE_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def choose_image(label, key):
    source = st.radio(
        label,
        ["Upload perangkat", "Google Drive"],
        horizontal=True,
        key=f"{key}_source",
    )

    if source == "Upload perangkat":
        uploaded = st.file_uploader(
            f"Pilih gambar {label.lower()} (maks. 10 MB)",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"{key}_upload",
        )

        if uploaded is None:
            return None, None

        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error("Ukuran gambar maksimal 10 MB.")
            return None, None

        try:
            img = ImageOps.exif_transpose(
                Image.open(uploaded)
            ).convert("RGB")
            return img, uploaded.size
        except (UnidentifiedImageError, OSError, ValueError):
            st.error("File bukan gambar yang valid.")
            return None, None

    try:
        images = download_drive_images()

        if not images:
            st.warning(
                "Gambar tidak ditemukan. Pastikan folder Drive "
                "bisa diakses dan berisi gambar."
            )
            return None, None

        selected = st.selectbox(
            f"Pilih gambar {label.lower()} dari Drive",
            images,
            format_func=lambda p: p.name,
            key=f"{key}_drive",
        )

        img = ImageOps.exif_transpose(
            Image.open(selected)
        ).convert("RGB")

        return img, selected.stat().st_size

    except Exception as exc:
        st.error(f"Gagal mengambil gambar dari Drive: {exc}")
        return None, None


st.markdown("""
<style>
.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 1.35rem 1.5rem;
    border-radius: 20px;
    background: linear-gradient(
        120deg,
        rgba(40,75,120,.18),
        rgba(90,150,170,.10)
    );
    border: 1px solid rgba(120,140,160,.25);
    margin-bottom: 1.2rem;
}

div[data-testid="stMetric"] {
    border: 1px solid rgba(120,140,160,.25);
    padding: 8px 10px;
    border-radius: 14px;
    min-width: 0;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div {
    font-size: 12px !important;
    line-height: 1.3 !important;
    overflow-wrap: anywhere;
}

.stCaption,
[data-testid="stCaptionContainer"] {
    font-size: 12px !important;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 12px;
    min-height: 2.8rem;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
  <div style="font-size:.85rem; letter-spacing:.08em; text-transform:uppercase; opacity:.75;">
    Pusat AI ITB · Prototype
  </div>
  <h1 style="margin:.35rem 0 .25rem 0;">AI Image Dehazer</h1>
  <p style="margin:0; opacity:.82;">Reduce Haziness in An Image.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Model Dehazing")
    st.write(MODEL_NAME)
    st.caption(
        "Inference dilakukan menggunakan checkpoint "
        "lokal pada perangkat yang menjalankan aplikasi."
    )

    st.divider()
    st.subheader("Privasi")
    st.write(
        "Gambar diproses secara lokal oleh aplikasi. "
        "Pastikan deployment tidak meneruskan gambar "
        "ke layanan eksternal."
    )

left, right = st.columns([1.05, 1], gap="large")
with left:
    st.subheader("1 · Input gambar")
    uploaded = st.file_uploader("Pilih gambar atau drag-and-drop maks 10 MB", type=["png", "jpg", "jpeg", "webp"])
    reference_file = st.file_uploader(
        "Ground-truth/reference (opsional; wajib pasangan yang benar-benar cocok)",
        type=["png", "jpg", "jpeg", "webp"], key="reference",
        help="Metrik hanya dihitung untuk reference yang valid dan sejajar dimensinya."
    )
    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"Ukuran file melewati batas {MAX_UPLOAD_MB} MB.")
            input_image = None
        else:
            try:
                input_image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
                if input_image.width < 16 or input_image.height < 16:
                    st.error("Gambar terlalu kecil. Gunakan minimal 16 × 16 piksel.")
                    input_image = None
                else:
                    st.image(input_image, caption="Gambar input", use_container_width=True)
                    st.caption(f"{input_image.width} × {input_image.height} px · {uploaded.size / 1024:.1f} KB")
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("File tidak dapat dibaca sebagai gambar yang valid.")
                input_image = None
    else:
        input_image = None
        st.info("Unggah gambar untuk memulai.")
    run_clicked = st.button("Jalankan dehazing", type="primary", use_container_width=True)

with right:
    st.subheader("2 · Hasil dehazing")
    if "api_result" not in st.session_state:
        st.session_state.api_result = None
        st.session_state.api_stats = None

    if run_clicked:
        if input_image is None:
            st.warning("Unggah gambar yang valid terlebih dahulu.")
        else:
            st.session_state.api_result = None
            st.session_state.api_stats = None
            try:
                with st.spinner("Memproses gambar dengan model Style Dehaze..."):
                    started = time.perf_counter()
                    output_image, model_info = run_dehazing(input_image)
                    elapsed = time.perf_counter() - started
                if not isinstance(output_image, Image.Image):
                    raise TypeError("API tidak mengembalikan gambar yang dapat dibaca.")
                output_image = output_image.convert("RGB")
                output_buffer = io.BytesIO()
                output_image.save(output_buffer, format="PNG")
                st.session_state.api_result = output_buffer.getvalue()
                st.session_state.api_stats = {
                    "elapsed": elapsed,
                    "model_info": model_info,
                    "input_size": input_size or 0,
                    "output_dims": output_image.size,
                }
            except DehazingAPIError as exc:
                st.error("Model API belum bisa memproses gambar. Coba lagi setelah beberapa saat.")
                st.caption(str(exc))
                st.markdown(f"Buka [demo model]({MODEL_SOURCE}) untuk mengecek apakah layanannya aktif.")
            except Exception as exc:
                st.error("Pemrosesan gagal.")
                st.caption(f"Detail: {type(exc).__name__}: {exc}")

    if st.session_state.api_result:
        output_image = Image.open(io.BytesIO(st.session_state.api_result)).convert("RGB")
        a, b = st.columns(2, gap="small")
        with a:
            st.image(input_image, caption="Before", use_container_width=True)
        with b:
            st.image(output_image, caption="After · hasil API model", use_container_width=True)
        st.download_button("Download hasil (PNG)", data=st.session_state.api_result,
                           file_name="dehazed_result.png", mime="image/png", use_container_width=True)
        stats = st.session_state.api_stats or {}
        m1, m2, m3 = st.columns(3)
        m1.metric("Waktu proses", f"{stats.get('elapsed', 0):.2f} s")
        m2.metric(
            "Dimensi output",
            f"{output_image.width} × {output_image.height} px"
        )
        m3.metric(
            "Ukuran input",
            f"{stats.get('input_size', 0) / 1024:.1f} KB"
        )
        st.caption("Waktu yang ditampilkan mencakup pengiriman gambar, antrean/komputasi server, dan pengambilan hasil—bukan murni waktu inferensi GPU.")
        if reference_image is not None:
            if reference_image.size != output_image.size:
                st.warning(
                    "PSNR/SSIM tidak dihitung karena dimensi "
                    "reference berbeda dari output."
                )
            else:
                psnr, ssim = calculate_psnr_ssim(
                    output_image, reference_image
                )
        else:
            st.caption("PSNR/SSIM tidak ditampilkan karena ground-truth/reference belum diberikan.")
    else:
        st.info("Hasil akan muncul setelah API model berhasil memproses gambar.")

st.divider()
st.caption("Catatan: demo model eksternal memiliki fokus pada citra satelit. Hasil untuk foto umum bisa bervariasi. Belum ada benchmark SOTS/O-HAZE/NH-HAZE/Dense-HAZE yang dijalankan oleh aplikasi ini.")
