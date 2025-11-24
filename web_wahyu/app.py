# app.py — VERSI FINAL & AMAN TOTAL

import streamlit as st
import cv2
import numpy as np
from preprocessing import getResi, getDataImg, extractData

# === CONFIG & STYLE ===
st.set_page_config(page_title="Resi OCR", layout="wide")
st.title("Ekstraksi Data Resi Otomatis")

# CSS biar rapi
st.markdown("""
<style>
    .centered-header h2 {
        text-align: center;
        color: #1e6b3d;
        font-weight: bold;
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)

# === UPLOAD GAMBAR ===
uploaded = st.file_uploader("Upload resi", type=["jpg", "png", "jpeg", "webp"])

if uploaded:
    file_bytes = uploaded.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_original is None:
        st.error("Gambar gagal dibaca!")
    else:
        col1, col2 = st.columns(2)

        # === KOLOM 1: GAMBAR ASLI ===
        with col1:
            st.subheader("Gambar Asli")
            st.image(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB),
                     use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # === TOMBOL PROSES ===
        if st.button("Proses Sekarang"):
            with st.spinner("Sedang mendeteksi & membaca teks..."):
                
                # 1. Crop resi utama (dari gambar asli)
                resi_crop = getResi(img_original)
                
                # 2. Crop pengirim & penerima (dari gambar asli!)
                crops_with_labels = getDataImg(img_original)  # ← PAKAI img_original!

                # 3. Ekstrak data
                hasil = extractData(crops_with_labels)

            # === KOLOM 2: HASIL ===
            with col2:
                
                st.markdown('<div class="centered-header"><h2>Detail Resi dari YOLOv5</h2></div>', 
                           unsafe_allow_html=True)

                # Tampilkan resi crop (jika ada)
                if resi_crop is not None and resi_crop.size > 0:
                    st.image(cv2.cvtColor(resi_crop, cv2.COLOR_BGR2RGB),
                             caption="Area Resi", use_container_width=True)
                else:
                    st.info("Resi tidak terdeteksi atau crop kosong")

                st.markdown("<hr>", unsafe_allow_html=True)

                # Tampilkan crop pengirim & penerima
                if crops_with_labels:
                    cols = st.columns(len(crops_with_labels))
                    for idx, (crop, label) in enumerate(crops_with_labels):
                        with cols[idx]:
                            # PENGECEKAN AMAN TANPA safe_rgb()
                            if crop is not None and crop.size > 0:
                                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                        caption=label.capitalize(),
                                        use_container_width=True)
                            else:
                                st.warning(f"{label.capitalize()} kosong")
                else:
                    st.warning("Pengirim atau penerima tidak terdeteksi")

                # === HASIL EKSTRAKSI ===
                st.markdown('<div class="centered-header"><h2>Hasil Extract</h2></div>', 
                           unsafe_allow_html=True)
                
                st.json(hasil, expanded=True)

        else:
            with col1:
                st.info("Upload gambar lalu klik **Proses Sekarang** untuk memulai")

else:
    st.info("Silakan upload gambar resi (JPG/PNG) untuk memulai")