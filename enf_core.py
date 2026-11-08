"""
enf_core.py — استخراج بصمة تردد الشبكة الكهربائية (ENF) من ملف صوتي.
"""

import numpy as np
import soundfile as sf
from scipy.signal import resample as scipy_resample
from scipy.signal import butter, filtfilt, stft


def _load_audio(path, sr_target):
    data, sr_native = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr_native != sr_target:
        n_target = int(round(len(data) * sr_target / sr_native))
        data = scipy_resample(data, n_target)
    return data.astype(np.float32), sr_target


def _parabolic_interpolate(freqs_axis, magnitudes, peak_idx):
    if peak_idx <= 0 or peak_idx >= len(magnitudes) - 1:
        return freqs_axis[peak_idx]
    y_m1, y_0, y_p1 = magnitudes[peak_idx - 1], magnitudes[peak_idx], magnitudes[peak_idx + 1]
    denom = (y_m1 - 2 * y_0 + y_p1)
    if abs(denom) < 1e-12:
        return freqs_axis[peak_idx]
    p = 0.5 * (y_m1 - y_p1) / denom
    df = freqs_axis[1] - freqs_axis[0] if len(freqs_axis) > 1 else 0
    return freqs_axis[peak_idx] + p * df


def extract_enf(audio_path, nominal_freq=50.0, band=0.4, sr_target=1000,
                 window_seconds=10.0, overlap_ratio=0.9):
    y, sr = _load_audio(audio_path, sr_target)

    low = (nominal_freq - band) / (sr / 2)
    high = (nominal_freq + band) / (sr / 2)
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, y)

    nperseg = int(window_seconds * sr)
    noverlap = int(nperseg * overlap_ratio)

    f, t, Zxx = stft(filtered, fs=sr, nperseg=nperseg, noverlap=noverlap)

    band_mask = (f >= nominal_freq - band) & (f <= nominal_freq + band)
    band_indices = np.where(band_mask)[0]

    freqs_over_time = []
    for i in range(Zxx.shape[1]):
        full_mag = np.abs(Zxx[:, i])
        band_mag = full_mag[band_indices]
        if band_mag.sum() < 1e-9:
            freqs_over_time.append(nominal_freq)
            continue
        local_peak = int(np.argmax(band_mag))
        global_peak_idx = band_indices[local_peak]
        refined = _parabolic_interpolate(f, full_mag, global_peak_idx)
        freqs_over_time.append(refined)

    return t, np.array(freqs_over_time)


def find_best_match(reference_times, reference_freqs, query_freqs):
    ref = np.asarray(reference_freqs) - np.mean(reference_freqs)
    qry = np.asarray(query_freqs) - np.mean(query_freqs)

    if len(qry) > len(ref) or len(qry) == 0:
        return None, 0.0

    correlation = np.correlate(ref, qry, mode="valid")
    norm = np.linalg.norm(qry) * np.array([
        np.linalg.norm(ref[i:i + len(qry)]) for i in range(len(correlation))
    ])
    norm[norm == 0] = 1e-9
    scores = correlation / norm

    best_i = int(np.argmax(scores))
    confidence = float(np.clip(scores[best_i], 0, 1))
    return best_i, confidence


def detect_jump(freqs, threshold_std=3.0):
    diffs = np.abs(np.diff(freqs))
    if len(diffs) < 3:
        return None
    mean_d, std_d = np.mean(diffs), np.std(diffs)
    if std_d < 1e-9:
        return None
    threshold = mean_d + threshold_std * std_d
    suspicious = np.where(diffs > threshold)[0]
    if len(suspicious) == 0:
        return None
    return int(suspicious[0]) + 1


# ---------------------------------------------------------------------------
# تحليل أكثر موثوقية: ارتباط مطبَّع (Pearson) + اتساق مواضع الأجزاء
# ---------------------------------------------------------------------------

def sliding_ncc(ref, qry):
    """ارتباط بيرسون المطبَّع بين qry وكل نافذة من ref (قيمة بين -1 و1)."""
    ref = np.asarray(ref, dtype=np.float64)
    qry = np.asarray(qry, dtype=np.float64)
    m = len(qry)
    if m == 0 or len(ref) < m:
        return np.array([])
    q = qry - qry.mean()
    qn = np.linalg.norm(q)
    if qn < 1e-12:
        return np.zeros(len(ref) - m + 1)
    num = np.correlate(ref, q, mode="valid")
    c1 = np.concatenate([[0.0], np.cumsum(ref)])
    c2 = np.concatenate([[0.0], np.cumsum(ref ** 2)])
    s1 = c1[m:] - c1[:-m]
    s2 = c2[m:] - c2[:-m]
    var = np.maximum(s2 - (s1 ** 2) / m, 1e-18)
    return num / (np.sqrt(var) * qn)


def _best_match_ncc(ref, qry):
    scores = sliding_ncc(ref, qry)
    if len(scores) == 0:
        return None, 0.0, 0.0
    bi = int(np.argmax(scores))
    best = float(scores[bi])
    # أفضل نتيجة ثانية بعيدة عن الأولى (لقياس تفرّد التطابق)
    excl = max(len(qry) // 2, 3)
    masked = scores.copy()
    masked[max(0, bi - excl): bi + excl + 1] = -1
    second = float(masked.max()) if len(masked) else -1.0
    return bi, best, best - second



def _detrend(x):
    t = np.arange(len(x), dtype=np.float64)
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ coef


def _consistent_at(ref, q, idx, min_r=0.6, seg_len=45):
    """يتأكد أن كل جزء من التسجيل يطابق نفس الموضع idx بالأرشيف، بعد إزالة
    الاتجاه الخطي من كل جزء (حتى لا ينخدع الارتباط بالصعود/الهبوط البطيء).
    يمنع قبول تسجيل مدموج كأصلي لمجرد أن الارتباط الكلي عالٍ."""
    n = len(q)
    if idx is None or idx < 0 or idx + n > len(ref):
        return False
    k = max(2, min(10, n // seg_len))
    edges = np.linspace(0, n, k + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 8:
            return False
        sq, sr_ = _detrend(q[a:b]), _detrend(ref[idx + a: idx + b])
        if np.std(sq) < 1e-12 or np.std(sr_) < 1e-12:
            return False
        if np.corrcoef(sq, sr_)[0, 1] < min_r:
            return False
    return True


def analyze_query(ref_freqs, q_freqs, min_conf=0.92, min_margin=0.15,
                  part_conf=0.92, part_margin=0.12, min_part=40, split_step=5, tol=4):
    """
    status: original | tampered | undetermined

    1) لو التسجيل كله يطابق الأرشيف بوضوح → أصلي.
    2) وإلا نجرّب نقاط قص مرشّحة؛ فإن طابق كل جزء الأرشيف بوضوح وبموضعين
       غير متصلين → متلاعب فيه (مع نقطة القص والفترة المفقودة).
    3) غير ذلك → غير محدد (لا نخمّن).
    """
    ref = np.asarray(ref_freqs, dtype=np.float64)
    q = np.asarray(q_freqs, dtype=np.float64)
    n = len(q)

    bi, conf, margin = _best_match_ncc(ref, q)
    if bi is None:
        return {"status": "undetermined", "reason": "التسجيل أطول من الأرشيف أو فارغ",
                "confidence": 0.0}

    if conf >= min_conf and margin >= min_margin and _consistent_at(ref, q, bi):
        return {"status": "original", "match_idx": int(bi), "confidence": conf, "margin": margin}

    best = None
    if n >= 2 * min_part:
        for sp in range(min_part, n - min_part + 1, split_step):
            i1, c1, m1 = _best_match_ncc(ref, q[:sp])
            i2, c2, m2 = _best_match_ncc(ref, q[sp:])
            if i1 is None or i2 is None:
                continue
            if min(c1, c2) < part_conf or min(m1, m2) < part_margin:
                continue
            score = min(c1, c2)
            if best is None or score > best["score"]:
                best = {"score": score, "sp": sp, "i1": i1, "i2": i2, "c1": c1, "c2": c2}

    if best is not None:
        sp, i1, i2 = best["sp"], best["i1"], best["i2"]
        end1 = i1 + sp - 1
        gap = i2 - end1 - 1
        if abs(gap) > tol:  # موضعان غير متصلين
            return {
                "status": "tampered",
                "split_index": int(sp),
                "part1_start_idx": int(i1),
                "part1_end_idx": int(end1),
                "part2_start_idx": int(i2),
                "gap_points": int(gap),
                "confidence": float((best["c1"] + best["c2"]) / 2),
            }
        return {"status": "original", "match_idx": int(i1),
                "confidence": float((best["c1"] + best["c2"]) / 2), "margin": float(margin)}

    return {"status": "undetermined", "match_idx": int(bi), "confidence": conf, "margin": margin,
            "reason": "لا يوجد تطابق قوي ومتفرّد بالأرشيف"}
