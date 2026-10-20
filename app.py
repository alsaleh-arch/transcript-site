"""Arabic-first video/audio transcription web app powered by faster-whisper."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="فيديو إلى نص | Transcript", page_icon="🎙️", layout="centered")

LANGUAGES = {
    "تلقائي (يتعرف على اللغة)": None,
    "العربية": "ar",
    "English": "en",
    "Français": "fr",
    "Español": "es",
    "Türkçe": "tr",
    "اردو": "ur",
    "हिन्दी": "hi",
}

MODELS = {
    "متوازن — دقة أفضل": "base",
    "سريع — مناسب للتجربة": "tiny",
}

ACCEPTED_EXTENSIONS = [
    "mp4", "mov", "mkv", "webm", "avi", "m4v", "mp3", "m4a", "wav", "aac", "ogg", "flac"
]

st.markdown(
    """
    <style>
    .stApp { direction: rtl; }
    .block-container { max-width: 860px; padding-top: 2.1rem; padding-bottom: 3rem; }
    html, body, [class*="css"], p, label, button, input, textarea {
        font-family: Tahoma, "Segoe UI", system-ui, sans-serif;
    }
    .hero {
        border-radius: 22px;
        background: linear-gradient(120deg, #153b59, #1b5c69 63%, #277c75);
        color: #fff;
        padding: 35px 32px;
        margin-bottom: 23px;
        box-shadow: 0 12px 30px rgba(12, 44, 60, .12);
    }
    .hero .eyebrow { font-size: 13px; font-weight: 600; color: #bcebe4; margin-bottom: 12px; }
    .hero h1 { font-size: 2.2rem; margin: 0 0 12px; color: white; font-weight: 750; line-height: 1.45; }
    .hero p { margin: 0; color: #e0f0ef; line-height: 1.9; font-size: 15px; }
    .step-header { font-weight: 700; font-size: 1.1rem; margin: 0 0 8px; }
    .quiet { color: #7d8793; font-size: 13px; }
    [data-testid="stFileUploader"] { border-radius: 15px; }
    div.stButton > button[kind="primary"] { border-radius: 10px; min-height: 49px; font-weight: 700; }
    div.stDownloadButton > button { border-radius: 10px; }
    </style>
    <div class="hero">
        <div class="eyebrow">🎙️ TRANSCRIPT STUDIO</div>
        <h1>حوّل أي فيديو إلى نص مكتوب</h1>
        <p>ارفع الفيديو أو الملف الصوتي، واضغط «ابدأ التفريغ». انسخ الكلام أو نزّله كنص أو ملف ترجمة بتوقيت.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def subtitle_time(seconds: float, *, srt: bool = True) -> str:
    """Format a floating-point second value as an SRT or VTT timestamp."""
    total_ms = max(0, round(seconds * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, milliseconds = divmod(rest, 1000)
    sep = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{milliseconds:03d}"


def build_srt(chunks: list[dict]) -> str:
    """Build SRT subtitles from Whisper segments."""
    return "\n\n".join(
        f"{i}\n{subtitle_time(c['start'])} --> {subtitle_time(c['end'])}\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    ) + "\n"


def build_vtt(chunks: list[dict]) -> str:
    """Build WebVTT subtitles from Whisper segments."""
    return "WEBVTT\n\n" + "\n\n".join(
        f"{subtitle_time(c['start'], srt=False)} --> {subtitle_time(c['end'], srt=False)}\n{c['text']}"
        for c in chunks
    ) + "\n"


@st.cache_resource(show_spinner=False)
def get_model(model_name: str) -> WhisperModel:
    """Cache the local CPU recognition model between Streamlit reruns."""
    return WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=2, num_workers=1)


st.markdown('<div class="step-header">١. اختر الملف</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "ارفع الفيديو أو الصوت هنا",
    type=ACCEPTED_EXTENSIONS,
    help="صيغ شائعة للفيديو والصوت. الحد الافتراضي في الاستضافة 200 ميجابايت للملف.",
)

c1, c2 = st.columns(2)
with c1:
    lang_label = st.selectbox("لغة الكلام", list(LANGUAGES), index=0)
with c2:
    model_label = st.selectbox("سرعة / دقة", list(MODELS), index=0)

st.caption("الملف يُعالج على الخادم. أول تشغيل قد يحتاج تحميل نموذج التعرف على الكلام. النتيجة تعتمد على جودة الصوت.")

current_file = (uploaded.name, uploaded.size) if uploaded else None
if st.session_state.get("current_file") != current_file:
    st.session_state["current_file"] = current_file
    for key in ("transcription", "editable_transcript"):
        st.session_state.pop(key, None)

start = st.button("✨ ابدأ التفريغ", type="primary", use_container_width=True, disabled=uploaded is None)

if start and uploaded is not None:
    tmp_path = None
    try:
        suffix = Path(uploaded.name).suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_path = tmp_file.name
            uploaded.seek(0)
            shutil.copyfileobj(uploaded, tmp_file)

        with st.spinner("جارٍ تجهيز نموذج التفريغ وتحليل الصوت..."):
            model = get_model(MODELS[model_label])
            segments, info = model.transcribe(
                tmp_path,
                language=LANGUAGES[lang_label],
                beam_size=3,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            bar = st.progress(0.0, text="جارٍ تحويل الكلام إلى نص...")
            chunks = []
            duration = max(float(info.duration or 0), 1.0)
            for segment in segments:
                clean = segment.text.strip()
                if clean:
                    chunks.append({"start": float(segment.start), "end": float(segment.end), "text": clean})
                bar.progress(min(1.0, max(0.0, float(segment.end) / duration)), text="جارٍ تحويل الكلام إلى نص...")
            bar.empty()

        if not chunks:
            st.warning("لم أتعرف على كلام واضح في الملف. تأكد من وجود صوت واضح وجرّب تحديد اللغة.")
        else:
            full_text = " ".join(c["text"] for c in chunks)
            st.session_state["transcription"] = {
                "text": full_text,
                "chunks": chunks,
                "language": info.language,
                "filename": Path(uploaded.name).stem,
            }
            st.session_state["editable_transcript"] = full_text
            st.success("تم التفريغ بنجاح ✅")
    except Exception as exc:
        st.error("تعذّر تفريغ هذا الملف. جرّب MP4 أو MP3، أو ملفًا أصغر، وتأكد من اتصال الاستضافة بالإنترنت.")
        with st.expander("تفاصيل تقنية للمساعدة في حل المشكلة"):
            st.code(f"{type(exc).__name__}: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

result = st.session_state.get("transcription")
if result:
    st.divider()
    st.markdown('<div class="step-header">٢. النص الناتج — يمكنك تعديله قبل التنزيل</div>', unsafe_allow_html=True)
    st.text_area("التفريغ النصي", key="editable_transcript", height=280)
    edited_text = st.session_state["editable_transcript"]
    st.caption("ملاحظة: أي تعديل يدوي على النص يظهر في TXT، أما SRT وVTT فيحتفظان بالتوقيت والنص الأصليين.")

    name = result["filename"]
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "⬇️ تنزيل TXT", data=("\ufeff" + edited_text).encode("utf-8"),
            file_name=f"{name}-transcript.txt", mime="text/plain; charset=utf-8", use_container_width=True,
        )
    with d2:
        st.download_button(
            "⬇️ تنزيل SRT", data=("\ufeff" + build_srt(result["chunks"])).encode("utf-8"),
            file_name=f"{name}-subtitles.srt", mime="text/plain; charset=utf-8", use_container_width=True,
        )
    with d3:
        st.download_button(
            "⬇️ تنزيل VTT", data=build_vtt(result["chunks"]).encode("utf-8"),
            file_name=f"{name}-subtitles.vtt", mime="text/vtt; charset=utf-8", use_container_width=True,
        )

st.markdown("---")
st.caption("مدعوم بنموذج Whisper مفتوح المصدر. يدعم لغات متعددة، لكن لا يضمن الدقة الكاملة أو تشغيل جميع ترميزات الفيديو.")
