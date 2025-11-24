import cv2
import matplotlib.pyplot as plt
from skimage.filters import unsharp_mask
import numpy as np
import pytesseract
from rapidfuzz import process, fuzz
import re
from regexText import *
import pandas as pd
import torch
import streamlit as st
import os

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Matikan AutoUpdate (opsional, agar tidak ganggu)
os.environ['ULTRALYTICS_AUTO_UPDATE'] = '0'

@st.cache_resource
def load_yolov5_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan: {model_path}")
        return None
    try:
        # GUNAKAN torch.hub UNTUK YOLOv5
        model = torch.hub.load(
            'ultralytics/yolov5', 
            'custom', 
            path=model_path,
            force_reload=True,
            trust_repo=True,
            _verbose=False
        )
        return model
    except Exception as e:
        st.error(f"Gagal load YOLOv5 model {model_path}: {e}")
        return None

# Load model YOLOv5
model_resi = load_yolov5_model('./model/best_3.pt')
model_info_penting = load_yolov5_model('./model/best_4.pt')
model_data = load_yolov5_model('./model/best_5.pt')


# --- Bright Image ---
def brightImage(img):
    return cv2.convertScaleAbs(img, alpha=0.65, beta=80)

# --- Preprocessing ---
def pra_proses_gambar(image):
    img = image.copy()
    h, w = img.shape[:2]
    scale = 1.5
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    img = (unsharp_mask(img, radius=1, amount=2) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def ensure_min_width(image, target_w=1400):
    h, w = image.shape[:2]
    if w >= target_w: return image
    scale = target_w / w
    return cv2.resize(image, (target_w, int(h * scale)), interpolation=cv2.INTER_LINEAR)

# --- OCR dari numpy array ---
def extractText(image_np):

    best_text, best_conf = "", None

    tess_config = [r"--oem 1 --psm 6", r"--oem 1 --psm 4"]
    if image_np is None or image_np.size == 0:
        return ""
    
    img = brightImage(pra_proses_gambar(image_np))
    for config in tess_config:
        text = pytesseract.image_to_string(img, lang='ind+eng', config=config)
        score = len(text.strip())
        if score > len(best_text.strip()):  # Jika berhasil baca sesuatu
            best_text = text
    return best_text

def process_single_block(full_text):
    if not full_text:
        return {"Nama": "Tidak ditemukan", "Alamat": "Tidak ditemukan", "Kontak": "Tidak ditemukan"}

    lines = [line.strip() for line in full_text.split('\n') if line.strip()]
    result = {"Nama": "Tidak ditemukan", "Alamat": "Tidak ditemukan", "Kontak": "Tidak ditemukan"}
    processed = set()
    alamat_map = {}

    # --- 1. Nama ---
    for i, line in enumerate(lines):
        if i in processed: continue
        clean = CONTACT_RE.sub('', line).strip()
        m = NAME_ANCHOR_RE.match(clean)
        if m:
            name = postprocess_name(m.group(2).strip())
            if name:
                result["Nama"] = name
                processed.add(i)
                break

    # --- 2. Alamat & Kontak ---
    for i, line in enumerate(lines):
        if i in processed or not line.strip():
            continue

        line_norm = normalize_contact_line(line)
        cands = extract_phone_candidates(line_norm)
        line_has_kw = CONTACT_KEYWORDS_RE.search(line_norm) is not None

        # Cari nomor telepon
        sel_phone = None
        sel_span = None
        for cand in cands:
            norm = normalize_phone_keep_plus(cand)
            if is_valid_phone(norm, line_has_kw):
                start = line_norm.find(cand)
                if start != -1:
                    sel_phone = norm
                    sel_span = (start, start + len(cand))
                    break

        # Cek keyword alamat
        has_specific = any(re.search(r'\b' + re.escape(k) + r'\b', line_norm, re.IGNORECASE)
                           for k in labels_to_find_general["Alamat_Keywords_Spesifik"])
        has_general = (any(re.search(re.escape(k), line_norm) for k in labels_to_find_general["Alamat_Keywords_Umum"])
                       and (len(line_norm.split()) > 1 or any(c.isalpha() for c in line_norm)))

        # Jika ada nomor + alamat → simpan kontak & alamat
        if sel_phone and (has_specific or has_general):
            if result["Kontak"] == "Tidak ditemukan":
                result["Kontak"] = sel_phone
            addr_after = line_norm[sel_span[1]:].lstrip(' ,;:-').strip()
            if addr_after:
                alamat_map[i] = addr_after
            processed.add(i)
            continue

        # Jika hanya ada alamat
        if has_specific or has_general:
            alamat_map[i] = line_norm
            processed.add(i)

    # Gabungkan alamat
    if alamat_map:
        sorted_lines = [alamat_map[idx] for idx in sorted(alamat_map.keys())]
        result["Alamat"] = " ".join(sorted_lines).strip()
    else:
        # Fallback: baris panjang tanpa kontak
        fallback = [
            ln for j, ln in enumerate(lines)
            if j not in processed and ln.strip() and CONTACT_RE.search(ln) is None
            and len(ln.split()) > 1 and any(c.isalpha() for c in ln)
        ]
        if fallback:
            result["Alamat"] = " ".join(fallback).strip()

    # --- 3. Kontak Fallback (jika belum ketemu) ---
    if result["Kontak"] == "Tidak ditemukan":
        for line in lines:
            line_norm = normalize_contact_line(line)
            cands = extract_phone_candidates(line_norm)
            for cand in cands:
                norm = normalize_phone_keep_plus(cand)
                if is_valid_phone(norm, True):
                    result["Kontak"] = norm
                    break
            if result["Kontak"] != "Tidak ditemukan":
                break

    return result
    


# --- Deteksi Object ---
def getObject(image, model=model_resi):
    if model is None or image is None: return None
    results = model(image)
    df = results.pandas().xyxy
    df = pd.concat(df)
    df = df.sort_values(by='confidence', ascending=False)
    filtered = df[df['name'] == 'resi']
    return filtered if not filtered.empty else None

def getInfoPenting(image, model=model_info_penting):
    if model is None or image is None: return None
    results = model(image)
    df = results.pandas().xyxy
    df = pd.concat(df)
    df = df.sort_values(by='confidence', ascending=False)
    filtered = df[df['name'] == 'info-penting-resi']
    return filtered if not filtered.empty else None

# === PERBAIKAN UTAMA: SEMUA CROP DARI GAMBAR ASLI! ===
def getResi(img_original):
    resi_df = getObject(img_original)  # Deteksi dari gambar asli
    if resi_df is None or resi_df.empty:
        return None

    r = resi_df.iloc[0]
    h, w = img_original.shape[:2]
    x1, y1, x2, y2 = map(int, [r['xmin'], r['ymin'], r['xmax'], r['ymax']])
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    resi_crop = img_original[y1:y2, x1:x2]
    
    # Info penting juga dari gambar asli!
    info_df = getInfoPenting(img_original)
    if info_df is not None and not info_df.empty:
        info = info_df.iloc[0]
        ix1, iy1, ix2, iy2 = map(int, [info['xmin'], info['ymin'], info['xmax'], info['ymax']])
        iy1 = max(0, iy1 - 15)
        ix1, ix2 = max(0, ix1), min(w, ix2)
        iy1, iy2 = max(0, iy1), min(h, iy2)
        return img_original[iy1:iy2, ix1:ix2]  # Info penting dari asli!

    return resi_crop  # Kalau tidak ada info penting, kembalikan resi yang di-brighten

# === PENGIRIM & PENERIMA: DARI GAMBAR ASLI! ===
def getDataImg(img_original, model=model_data):
    if model is None or img_original is None or img_original.size == 0:
        return []

    model.conf = 0.01
    results = model(img_original)
    df = results.pandas().xyxy[0]

    if not {'pengirim', 'penerima'}.issubset(set(df['name'])):
        return []

    # Sesuaikan tinggi pengirim = penerima
    penerima = df[df['name'] == 'penerima'].iloc[0]
    height = penerima['ymax'] - penerima['ymin']
    df.loc[df['name'] == 'pengirim', 'ymax'] = df['ymin'] + height

    # Ambil confidence tertinggi
    boxes = df[df['name'].isin(['pengirim', 'penerima'])].loc[
        df.groupby('name')['confidence'].idxmax()
    ]

    crops = []
    h, w = img_original.shape[:2]

    for _, row in boxes.iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])

        # Tambah margin
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img_original[y1:y2, x1:x2]
        crops.append((crop, row['name']))

    return crops

# === extractData (DIPERBAIKI SEDIKIT) ===
def extractData(image_list):
    if not image_list:
        empty = {"Nama": "Tidak ditemukan", "Alamat": "Tidak ditemukan", "Kontak": "Tidak ditemukan"}
        return {"pengirim": empty, "penerima": empty}

    result = {"pengirim": None, "penerima": None}

    for crop, label in image_list:
        text = extractText(crop)
        processed = process_single_block(text)
        if label == "pengirim":
            result["pengirim"] = processed
        elif label == "penerima":
            result["penerima"] = processed

    # Isi yang kosong
    empty = {"Nama": "Tidak ditemukan", "Alamat": "Tidak ditemukan", "Kontak": "Tidak ditemukan"}
    if result["pengirim"] is None:
        result["pengirim"] = empty
    if result["penerima"] is None:
        result["penerima"] = empty

    return result










            


    
