from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from metrics import calculate_psnr_ssim
from inference_adapter import run_dehazing, ModelUnavailableError

APP_TITLE = "AI Image Dehazing"
MAX_UPLOAD_MB = 10
MAX_PIXELS = 2_500_000

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container {max-width: 1180px; padding-top: 1.5rem; padding-bottom: 3rem;}
    .hero {padding: 1.35rem 1.5rem; border-radius: 20px;
           background: linear-gradient(120deg, rgba(40,75,120,.18), rgba(90,150,170,.10));
           border: 1px solid rgba(120,140,160,.25); margin-bottom: 1.2rem;}
    .muted {opacity: .72; font-size: .92rem;}
    div[data-testid="stMetric"] {border: 1px solid rgba(120,140,160,.25);
        padding: 12px 14px; border-radius: 14px;}
    .stButton > button, .stDownloadButton > button {border-radius: 12px; min-height: 2.8rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div style="font-size:.85rem; letter-spacing:.08em; text-transform:uppercase; opacity:.75;">
    Pusat AI ITB · Prototype
  </div>
  <h1 style="margin:.35rem 0 .25rem 0;">AI Image Dehazing</h1>
  <p style="margin:0; opacity:.82;">
    Unggah gambar berkabut, jalankan model dehazing, lalu bandingkan hasilnya.
  </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Pengaturan")
    st.caption("Validasi input")
    st.write(f"Format: PNG, JPEG, WEBP")
    st.write(f"Batas file: {MAX_UPLOAD_MB} MB")
    st.write("Batas resolusi: 2,5 megapiksel")
    st.caption("Model: AOD-Net pretrained · CPU/GPU otomatis sesuai server")
    st.divider()
    st.caption("Privasi")
    st.write("Gambar diproses oleh server yang menjalankan aplikasi. Starter ini tidak menyimpan gambar secara permanen dengan sengaja; hosting dapat memiliki log/penyimpanan sementara sesuai konfigurasi.")

left, right = st.columns([1.05, 1], gap="large")
with left:
    st.subheader("1 · Input gambar")
    uploaded = st.file_uploader(
        "Pilih gambar hazy/foggy atau drag-and-drop di sini",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        help=f"Format PNG/JPEG/WEBP, maksimum {MAX_UPLOAD_MB} MB.",
    )

    reference_file = st.file_uploader(
        "Ground-truth/reference (opsional, harus cocok dengan gambar input)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        key="reference",
        help="Gunakan hanya pasangan ground-truth yang benar-benar sesuai. Ukuran harus sama setelah preprocessing.",
    )

    run_clicked = st.button("Jalankan dehazing", type="primary", use_container_width=True)

    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"Ukuran file melebihi batas {MAX_UPLOAD_MB} MB.")
            input_image = None
        else:
            try:
                input_image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
                if input_image.width < 16 or input_image.height < 16:
                    st.error("Gambar terlalu kecil. Gunakan gambar minimal 16 × 16 piksel.")
                    input_image = None
                elif input_image.width * input_image.height > MAX_PIXELS:
                    st.error("Resolusi gambar melebihi 2,5 megapiksel. Resize gambar sebelum upload agar inference CPU gratis tidak kehabisan memori.")
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

with right:
    st.subheader("2 · Hasil dehazing")
    if "dehazed_bytes" not in st.session_state:
        st.session_state.dehazed_bytes = None
        st.session_state.run_stats = None
        st.session_state.input_fingerprint = None

    if run_clicked:
        if uploaded is None or input_image is None:
            st.warning("Unggah gambar yang valid terlebih dahulu.")
        else:
            st.session_state.dehazed_bytes = None
            st.session_state.run_stats = None
            try:
                with st.spinner("Menjalankan model dehazing…"):
                    start = time.perf_counter()
                    output_image, model_info = run_dehazing(input_image)
                    elapsed_total = time.perf_counter() - start
                    elapsed = (model_info or {}).get("inference_seconds", elapsed_total)

                if not isinstance(output_image, Image.Image):
                    raise TypeError("Adapter model harus mengembalikan PIL.Image.Image.")
                output_image = output_image.convert("RGB")
                if output_image.size != input_image.size:
                    st.warning("Dimensi output berbeda dari input. Output ditampilkan pada ukuran aslinya.")

                buffer = io.BytesIO()
                output_image.save(buffer, format="PNG")
                st.session_state.dehazed_bytes = buffer.getvalue()
                st.session_state.run_stats = {
                    "elapsed": elapsed,
                    "model_info": model_info or {},
                    "input_size": input_image.size,
                    "output_size": output_image.size,
                    "file_bytes": uploaded.size,
                }
                st.session_state.input_fingerprint = uploaded.name + str(uploaded.size)
            except ModelUnavailableError as exc:
                st.error("Model AI belum siap dijalankan.")
                st.code(str(exc))
                st.info("Inference sengaja dinonaktifkan sampai repository, dependensi, dan checkpoint asli terpasang serta lolos tes kompatibilitas.")
            except Exception as exc:
                st.error("Inference gagal. Periksa log server dan konfigurasi model.")
                st.caption(f"Detail teknis: {type(exc).__name__}: {exc}")

    if st.session_state.dehazed_bytes:
        output_image = Image.open(io.BytesIO(st.session_state.dehazed_bytes)).convert("RGB")
        input_col, output_col = st.columns(2, gap="small")
        with input_col:
            st.image(input_image, caption="Before", use_container_width=True)
        with output_col:
            st.image(output_image, caption="After · output model", use_container_width=True)

        st.download_button(
            "Download hasil (PNG)",
            data=st.session_state.dehazed_bytes,
            file_name="dehazed_result.png",
            mime="image/png",
            use_container_width=True,
        )

        stats = st.session_state.run_stats or {}
        m1, m2, m3 = st.columns(3)
        m1.metric("Waktu inferensi", f"{stats.get('elapsed', 0):.3f} s")
        m2.metric("Dimensi output", f"{output_image.width} × {output_image.height}")
        m3.metric("Ukuran input", f"{stats.get('file_bytes', 0) / 1024:.1f} KB")

        model_info = stats.get("model_info", {})
        st.caption(f"Model: {model_info.get('name', 'Tidak dilaporkan')} · Device: {model_info.get('device', 'Tidak dilaporkan')} · Source: {model_info.get('source', 'N/A')}")
        if model_info.get("model_load_seconds") is not None:
            st.caption(f"Waktu load/download model (terpisah dari inferensi): {model_info['model_load_seconds']:.2f} s. Model disimpan di cache selama proses server aktif.")

        if reference_file is not None:
            try:
                reference_image = ImageOps.exif_transpose(Image.open(reference_file)).convert("RGB")
                if reference_image.size != output_image.size:
                    st.warning(
                        f"PSNR/SSIM tidak dihitung: ukuran reference {reference_image.size} "
                        f"berbeda dari output {output_image.size}. Pastikan pasangan dan preprocessing benar."
                    )
                else:
                    psnr, ssim = calculate_psnr_ssim(output_image, reference_image)
                    c1, c2 = st.columns(2)
                    c1.metric("PSNR", f"{psnr:.3f} dB")
                    c2.metric("SSIM", f"{ssim:.4f}")
                    st.caption("Metrik dihitung terhadap reference yang diunggah. Validitas pasangan ground-truth tetap menjadi tanggung jawab pengguna.")
            except Exception as exc:
                st.warning(f"Metrik tidak dihitung karena reference tidak valid: {exc}")
        else:
            st.caption("PSNR/SSIM tidak ditampilkan karena ground-truth/reference belum diberikan.")
    else:
        st.info("Hasil akan muncul di sini setelah model pretrained berhasil dijalankan.")
        st.markdown("""
        **Model yang digunakan: AOD-Net**

        Model pretrained diunduh dari repository publik saat inferensi pertama. Server harus memiliki akses internet untuk mengambil checkpoint; setelah berhasil, checkpoint dicache selama runtime aktif.
        """)

st.divider()
st.caption("Model: AOD-Net, ICCV 2017. Hasil dapat berbeda pada haze nyata; benchmark dataset belum dilakukan oleh aplikasi ini. Jangan gunakan sebagai bukti ilmiah/forensik tanpa validasi.")
