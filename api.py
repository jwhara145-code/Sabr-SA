"""
api.py — واجهة REST API حقيقية لمنصة سَبْر، تُستخدم من أي فرونت-إند
(بما فيها Lovable). تُغلِّف نفس المحرك العلمي المختبَر (enf_core.py)
ونفس قاعدة البيانات (db.py) بدون أي تغيير بالمنطق المُثبَت.
"""

import os
import tempfile
import subprocess
import datetime
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from enf_core import extract_enf, analyze_query
import db

app = FastAPI(title="Sabr API", description="منصة سَبْر — واجهة برمجية للتحقق من صحة الأدلة الرقمية")

# يسمح لأي فرونت-إند (مثل Lovable) بالاتصال بهذا الـAPI من دومين مختلف
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _convert_to_wav(raw_path: str) -> str:
    out_path = raw_path + "_converted.wav"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", raw_path, "-ar", "16000", "-ac", "1", "-vn", out_path],
        capture_output=True,
    )
    if result.returncode != 0 or not os.path.exists(out_path):
        raise HTTPException(status_code=400, detail="تعذّر تحويل الملف الصوتي. تأكدي أن الملف سليم.")
    return out_path


@app.get("/stats")
def get_stats():
    total_points = db.archive_count()
    oldest, newest = db.archive_time_range()
    v_count = db.verification_count()
    recent = db.get_recent_verifications(1)
    last_result = recent[0][2] if recent else None
    return {
        "archive_points": total_points,
        "archive_start": oldest.isoformat() if oldest else None,
        "archive_end": newest.isoformat() if newest else None,
        "verification_count": v_count,
        "last_result": last_result,
        "db_size_kb": db.storage_size_kb(),
        "storage": db.backend_name(),
    }


@app.get("/archive/preview")
def archive_preview(limit: int = 200):
    _, freqs = db.load_reference_archive()
    shown = freqs[-limit:] if len(freqs) > limit else freqs
    return {"frequencies": [float(f) for f in shown]}


@app.post("/archive")
async def add_reference(
    file: UploadFile = File(...),
    start_time: str = Form(...),  # ISO format: "2026-09-16T10:00:00"
):
    try:
        start_dt = datetime.datetime.fromisoformat(start_time)
    except ValueError:
        raise HTTPException(status_code=400, detail="صيغة الوقت غير صحيحة، استخدمي ISO 8601")

    suffix = "." + file.filename.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        raw_path = tmp.name

    wav_path = _convert_to_wav(raw_path)
    try:
        t, f = extract_enf(wav_path)
        saved_count = db.save_reference_points(start_dt, t, f)
    finally:
        os.unlink(raw_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    return {"saved_points": saved_count}


MIN_RECOMMENDED_SECONDS = 300  # أقل طول موصى به لتحليل موثوق


@app.post("/verify")
async def verify_recording(file: UploadFile = File(...)):
    archive_times, archive_freqs = db.load_reference_archive()
    if len(archive_freqs) == 0:
        raise HTTPException(status_code=400, detail="الأرشيف المرجعي فارغ — أضيفي تسجيلًا مرجعيًا أولًا.")

    suffix = "." + file.filename.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        raw_path = tmp.name

    wav_path = _convert_to_wav(raw_path)
    try:
        q_times, q_freqs = extract_enf(wav_path)
        n = len(q_freqs)
        res = analyze_query(archive_freqs, q_freqs)
        last = len(archive_times) - 1

        response = {
            "frequencies": [float(x) for x in q_freqs],
            "analysis_points": n,
            "short_recording": n < MIN_RECOMMENDED_SECONDS,
            "min_recommended_seconds": MIN_RECOMMENDED_SECONDS,
            "status": res["status"],
            "confidence": float(res.get("confidence", 0.0)),
        }

        if res["status"] == "original":
            i = res["match_idx"]
            m_start, m_end = archive_times[i], archive_times[min(i + n - 1, last)]
            response.update({"matched_start": m_start.isoformat(), "matched_end": m_end.isoformat()})
            db.log_verification(file.filename, "أصلي", response["confidence"],
                                f"{m_start.isoformat()} → {m_end.isoformat()}")

        elif res["status"] == "tampered":
            i1s, i1e, i2s = res["part1_start_idx"], res["part1_end_idx"], res["part2_start_idx"]
            p1_start, p1_end = archive_times[min(max(i1s, 0), last)], archive_times[min(max(i1e, 0), last)]
            p2_start = archive_times[min(max(i2s, 0), last)]
            gap_seconds = (p2_start - p1_end).total_seconds()
            response.update({
                "split_index": res["split_index"],
                "part1_match": {"start": p1_start.isoformat(), "end": p1_end.isoformat()},
                "part2_match": {"start": p2_start.isoformat()},
                "gap_seconds": gap_seconds,
            })
            db.log_verification(file.filename, "متلاعب فيه", response["confidence"],
                                f"جزء مفقود/مُعاد ترتيبه: {gap_seconds:.0f} ثانية")

        else:
            response["reason"] = res.get("reason", "لا يوجد تطابق قوي ومتفرّد بالأرشيف")
            db.log_verification(file.filename, "غير محدد", response["confidence"], response["reason"])

        return response
    finally:
        os.unlink(raw_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)


@app.get("/log")
def get_log(limit: int = 20):
    rows = db.get_recent_verifications(limit)
    return [
        {"checked_at": r[0], "file_name": r[1], "result": r[2], "confidence": r[3], "details": r[4]}
        for r in rows
    ]


@app.get("/")
def root():
    return {"name": "Sabr API", "status": "running", "endpoints": ["/stats", "/archive", "/verify", "/log"]}
