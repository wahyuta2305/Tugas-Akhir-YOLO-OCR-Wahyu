import re
from glob import glob
from rapidfuzz import fuzz
# =====================[ REGEX & KEYWORDS ]=======================
# (1) Deteksi cepat untuk pembersihan nama (seperti V6.6)
contact_pattern_quick = r'(?:\+?62|0?8)[\d\s\-]{8,16}'
CONTACT_RE = re.compile(contact_pattern_quick)

# (2) Ekstraksi kandidat nomor (longgar: +/digit diikuti digit/spasi/dash)
CONTACT_EXTRACT_RE = re.compile(r'\+?\d[\d\s\-]{3,}')
CONTACT_KEYWORDS_RE = re.compile(r'\b(tel|tlp|hp|wa|wa\.|telp|no\.?|kontak)\b', re.I)

# Anchor exact
NAME_ANCHOR_RE = re.compile(r'^\s*(penerima|pengirim)\s*[:\-]?\s*(.+)\s*$', re.IGNORECASE)

labels_to_find_general = {
    "Nama_Keywords": [
        "Pengirim", "Penerima",
        "-Penerima" ,"Penesia", "Penenma", "Pengitim", "Perma", "Penge", "erima", "eritna", "2ENERIMA", "renginm", "Penorima", "seenerima", "Poaasliie",
        "Peagirim", "Pengrim", "Peng", "Penginim", "Pen", "Pengicim", "Pngirm", "Pnrm", "ittn", "iBenigiririis", "rim", "Pengirin", "2engirim", "Pengitim", "Pengittin", "Prarie"
    ],
    "Alamat_Keywords_Spesifik": ["Jl.", "Ji.", "JA.", "Jalan", "Jalam","RT", "RW", "No.", "No", "Gang", "Gg.", "Blok", "Kel.", "Desa", "Kec.", "Kab.", "Kota", "Prov.", "Kode Pos", "kp", "Perumahan"],
    "Alamat_Keywords_Umum": [",", "."],
    "Kontak_Keywords": ["Tel.", "Telp.", "No. Telp", "08"]
}
NAMA_KW_LC = [k.lower().rstrip(':') for k in labels_to_find_general["Nama_Keywords"]]

# Tambahan anchor manual
EXTRA_ANCHORS = {"ibenigiririis", "2enerima"}  # case-insensitive

ELLIPSIS_RE = re.compile(r'\.{3,}\s*$')
NOISE_RUN_RE = re.compile(r'(.)\1{3,}', re.I)  # run huruf sama >=4
VOWEL_RE = re.compile(r'[aeiou]', re.I)

# =====================[ HELPER: KONTAK ]=========================
def normalize_contact_line(line: str) -> str:
    """Rapikan spasi setelah plus: '+6 2' -> '+62', '+ 6281' -> '+6281'."""
    return re.sub(r'\+\s*([0-9])', r'+\1', line or "")

def extract_phone_candidates(line: str):
    """Ambil kandidat blok nomor (boleh ada +, spasi, dash)."""
    return re.findall(CONTACT_EXTRACT_RE, line or "")

def normalize_phone_keep_plus(raw: str) -> str:
    """Pertahankan '+' hanya jika ada di awal token; buang spasi/dash lain."""
    raw = (raw or "").strip()
    keep_plus = raw.startswith('+')
    digits = re.sub(r'\D', '', raw)
    return ('+' + digits) if keep_plus else digits

def is_valid_phone(norm: str, line_has_contact_kw: bool) -> bool:
    """
    Validasi fleksibel:
      - IZINKAN 3 digit jika ada kata kunci kontak, atau prefix +62/62/08.
      - Tanpa keyword/prefix, minimal 5 digit agar tidak ambil angka acak.
    """
    if not norm:
        return False
    # Ada keyword kontak -> minimal 3 digit
    if line_has_contact_kw and len(norm) >= 3:
        return True
    # Prefix resmi -> minimal 3 digit
    if (norm.startswith('+62') or norm.startswith('62') or norm.startswith('08')) and len(norm) >= 3:
        return True
    # Fallback: digit semua -> minimal 5 digit
    if norm.isdigit():
        return len(norm) >= 5
    return False

# =====================[ HELPER: NAMA ]===========================
def strip_trailing_punct_keep_ellipsis(s: str) -> str:
    """Buang tanda baca ujung kecuali ellipsis (...)."""
    s = s.strip()
    if ELLIPSIS_RE.search(s):
        return s
    return re.sub(r'[:;,\.\-]+\s*$', '', s)

def strip_trailing_contact_and_punct(s: str) -> str:
    """(Legacy quick) Jika ada pola kontak di akhir string, potong mulai awal nomor."""
    m = CONTACT_RE.search(s)
    if not m:
        return s
    cut_at = m.start()
    while cut_at > 0 and s[cut_at-1] in ' ,;-':
        cut_at -= 1
    return s[:cut_at].rstrip()

def is_noise_token(tok: str) -> bool:
    """
    Token noise khas OCR:
      - sangat panjang (>=15), atau
      - ada run huruf sama ≥4 (eeee/ssss), atau
      - panjang >=8 dengan rasio vokal sangat rendah.
    """
    t = tok.strip(".,:;()-").strip()
    if not t:
        return False
    if len(t) >= 15:
        return True
    if NOISE_RUN_RE.search(t):
        return True
    vowels = len(VOWEL_RE.findall(t))
    if len(t) >= 8 and vowels / max(1, len(t)) < 0.15:
        return True
    return False

def kill_trailing_noise_sequence(s: str) -> str:
    """Buang buntut token noise dari belakang (tanpa menyentuh kata normal di awal)."""
    parts = s.split()
    out = []
    for w in parts:
        if is_noise_token(w):
            break
        out.append(w)
    return " ".join(out).strip()

def maybe_anchor_like(token: str, ratio_thr=85, partial_thr=78) -> bool:
    """Cek apakah token mirip anchor 'penerima/pengirim' (termasuk typo berat) atau extra anchors."""
    t_raw = token.rstrip(':').strip()
    t = t_raw.lower()
    if t in EXTRA_ANCHORS:
        return True
    if not t.isalpha():
        return False
    for kw in NAMA_KW_LC:
        if fuzz.ratio(t, kw) >= ratio_thr or fuzz.partial_ratio(t, kw) >= partial_thr:
            return True
    return False

def remove_name_keywords(text: str, threshold: int = 85, partial_thr: int = 80) -> str:
    """Buang token yang identik/mirip anchor (termasuk typo). Tidak menyentuh ellipsis."""
    if not text:
        return text
    tokens = text.split()
    kept = []
    for w in tokens:
        if ELLIPSIS_RE.search(w):
            kept.append(w)
            continue
        w_core = w.strip(':,.-()')
        w_norm = w_core.lower()
        anchorish = any(
            (fuzz.ratio(w_norm, kw) >= threshold) or
            (fuzz.partial_ratio(w_norm, kw) >= partial_thr)
            for kw in NAMA_KW_LC
        ) or (w_norm in EXTRA_ANCHORS)
        if anchorish:
            continue
        kept.append(w_core if w_core else w)
    return " ".join(kept).strip()

def clean_final_name(s: str) -> str:
    """Buang nomor di dalam teks (quick), potong nomor di ujung, rapikan (jaga ellipsis)."""
    if not s:
        return s
    s = CONTACT_RE.sub('', s).strip()
    s = strip_trailing_contact_and_punct(s)
    s = strip_trailing_punct_keep_ellipsis(s)
    return s

def postprocess_name(text: str) -> str:
    """
    1) Bersihkan & potong nomor HP di ekor
    2) Hapus pseudo-anchor depan (ittn:, irin:, 2ENERIMA:, iBenigiririis:, ...)
    3) Buang token anchor sisa
    4) Potong buntut noise ekstrim
    5) Rapikan akhir (jaga ellipsis)
    """
    text = clean_final_name(text)

    # (2) pseudo-anchor depan
    parts = text.split(maxsplit=1)
    if parts:
        first = parts[0]
        first_raw = first.rstrip(':')
        if first.endswith(':') and (maybe_anchor_like(first) or first_raw.lower() in EXTRA_ANCHORS):
            text = parts[1] if len(parts) > 1 else ""

    # (3) buang anchor di sisa
    text = remove_name_keywords(text, threshold=85, partial_thr=80)

    # (4) potong buntut noise
    text = kill_trailing_noise_sequence(text)

    # (5) rapikan
    text = strip_trailing_punct_keep_ellipsis(text)
    return text.strip()

def detect_anchor_fuzzy_prefix(line: str):
    """
    Deteksi anchor fuzzy di awal baris (token sebelum ':' atau spasi),
    contoh: 'Panorima:', 'enerima:', 'Pengittn:', atau EXTRA_ANCHORS.
    """
    s = line.strip()
    if not s:
        return False, ""
    m = re.match(r'^([^\s:]+)\s*[:\-]?\s*(.*)$', s)
    if not m:
        return False, ""
    tok, rest = m.group(1), m.group(2)
    if tok.lower() in EXTRA_ANCHORS or maybe_anchor_like(tok):
        return True, rest
    return False, ""