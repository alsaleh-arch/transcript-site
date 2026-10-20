"""Arabic-first Streamlit video/audio transcription from files or social post URLs."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import streamlit as st
import yt_dlp
from faster_whisper import WhisperModel
from yt_dlp.utils import DownloadError

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

# Only enable known video platform URLs. Do not hand arbitrary URLs to a downloader.
ALLOWED_HOSTS = {
    "x.com", "www.x.com", "mobile.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com",
    "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
    "instagram.com", "www.instagram.com",
    "tiktok.com", "www.tiktok.com", "vm.tiktok.com", "vt.tiktok.com",
}
MAX_LINK_DURATION_SECONDS = 30 * 60
MAX_LINK_FILE_BYTES = 100 * 1024 * 1024

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
    [data-testid="stFileUploader"] { border-radius: 15px; }
    div.stButton > button[kind="primary"] { border-radius: 10px; min-height: 49px; font-weight: 700; }
    div.stDownloadButton > button { border-radius: 10px; }
    </style>
    <div class="hero">
        <div class="eyebrow">🎙️ TRANSCRIPT STUDIO</div>
        <h1>حوّل الفيديو أو رابطه إلى نص مكتوب</h1>
        <p>ارفع فيديو من جهازك أو الصق رابط منشور عام فيه فيديو، ثم نزّل النص أو ملف الترجمة بتوقيت.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def subtitle_time(seconds: float, *, srt: bool = True) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, milliseconds = divmod(rest, 1000)
    sep = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{milliseconds:03d}"


def build_srt(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"{i}\n{subtitle_time(c['start'])} --> {subtitle_time(c['end'])}\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    ) + "\n"


def build_vtt(chunks: list[dict]) -> str:
    return "WEBVTT\n\n" + "\n\n".join(
        f"{subtitle_time(c['start'], srt=False)} --> {subtitle_time(c['end'], srt=False)}\n{c['text']}"
        for c in chunks
    ) + "\n"


@st.cache_resource(show_spinner=False)
def get_model(model_name: str) -> WhisperModel:
    return WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=2, num_workers=1)


def validate_video_url(url: str) -> str:
    """Only accept HTTPS links from supported public social/video platforms."""
    url = url.strip()
    try:
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError as exc:
        raise ValueError("الرابط غير صالح. انسخ رابط المنشور كاملًا.") from exc
    if parsed.scheme != "https" or hostname not in ALLOWED_HOSTS or port not in (None, 443):
        raise ValueError("الرابط غير مدعوم. استخدم رابط HTTPS من X أو YouTube أو TikTok أو Instagram.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("الرابط غير صالح.")
    if not parsed.path or parsed.path == "/":
        raise ValueError("الصق رابط المنشور أو الفيديو نفسه، وليس الصفحة الرئيسية.")
    return url


def download_audio_from_link(url: str, directory: Path) -> tuple[Path, str]:
    """Extract one publicly accessible video, convert its audio to MP3, and return it."""
    url = validate_video_url(url)

    def reject_long_video(info: dict, *, incomplete: bool = False) -> str | None:
        duration = info.get("duration")
        if duration and float(duration) > MAX_LINK_DURATION_SECONDS:
            return "مدة الفيديو تتجاوز 30 دقيقة. اختر فيديو أقصر."
        if info.get("is_live"):
            return "البث المباشر غير مدعوم."
        return None

    def check_download_size(status: dict) -> None:
        if status.get("status") == "downloading" and status.get("downloaded_bytes", 0) > MAX_LINK_FILE_BYTES:
            raise ValueError("حجم الفيديو يتجاوز 100 ميجابايت. استخدم مقطعًا أصغر.")

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(directory / "source.%(ext)s"),
        "noplaylist": True,
        "playlistend": 1,
        "max_filesize": MAX_LINK_FILE_BYTES,
        "match_filter": reject_long_video,
        "progress_hooks": [check_download_size],
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}],
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 20,
        "retries": 2,
        "fragment_retries": 2,
        "restrictfilenames": True,
        "overwrites": True,
        "nopart": False,
    }
    with yt_dlp.YoutubeDL(opts) as downloader:
        info = downloader.extract_info(url, download=True)
    if not info:
        raise ValueError("ما قدرنا نستخرج فيديو من الرابط. تأكد أن المنشور يحتوي فيديو عامًا.")

    audio_path = directory / "source.mp3"
    if not audio_path.is_file():
        raise ValueError("تم الوصول للرابط، لكن ما قدرنا نجهّز الصوت منه.")
    if audio_path.stat().st_size > MAX_LINK_FILE_BYTES:
        raise ValueError("الملف الصوتي كبير جدًا.")

    video_id = str(info.get("id") or "video")
    safe_id = re.sub(r"[^0-9A-Za-z_-]", "-", video_id)[:70] or "video"
    return audio_path, safe_id


st.markdown('<div class="step-header">١. اختر مصدر الفيديو</div>', unsafe_allow_html=True)
source_mode = st.radio(
    "طريقة إدخال الفيديو",
    ["📁 رفع ملف من الجهاز", "🔗 لصق رابط فيديو"],
    horizontal=True,
)

uploaded = None
video_url = ""
if source_mode.startswith("📁"):
    uploaded = st.file_uploader(
        "ارفع الفيديو أو الصوت هنا",
        type=ACCEPTED_EXTENSIONS,
        help="صيغ الفيديو والصوت الشائعة. الحد الافتراضي للاستضافة 200 ميجابايت للملف.",
    )
else:
    video_url = st.text_input(
        "رابط منشور الفيديو",
        placeholder="https://x.com/info__usa/status/2112282876817625489?s=20",
    )
    st.caption(
        "روابط المنشورات العامة من X (تويتر)، YouTube، TikTok وInstagram. "
        "لا ندعم المنشورات الخاصة أو الروابط التي تشترط تسجيل الدخول، وبعض المنصات قد تمنع الوصول الآلي. "
        "الحد: 30 دقيقة و100 ميجابايت للتنزيل."
    )

c1, c2 = st.columns(2)
with c1:
    lang_label = st.selectbox("لغة الكلام", list(LANGUAGES), index=0)
with c2:
    model_label = st.selectbox("سرعة / دقة", list(MODELS), index=0)

st.caption("التفريغ يجري على خادم التطبيق. أول تشغيل قد يحتاج تحميل نموذج Whisper. استخدم مقاطع تملك الحق في معالجتها.")

current_source = (
    ("file", uploaded.name, uploaded.size) if uploaded is not None
    else ("url", video_url.strip()) if source_mode.startswith("🔗")
    else None
)
if st.session_state.get("current_source") != current_source:
    st.session_state["current_source"] = current_source
    for key in ("transcription", "editable_transcript"):
        st.session_state.pop(key, None)

has_input = uploaded is not None or (source_mode.startswith("🔗") and bool(video_url.strip()))
start = st.button("✨ ابدأ التفريغ", type="primary", use_container_width=True, disabled=not has_input)

if start and has_input:
    try:
        with tempfile.TemporaryDirectory(prefix="transcript_") as workdir:
            tempdir = Path(workdir)
            if uploaded is not None:
                suffix = Path(uploaded.name).suffix.lower() or ".mp4"
                media_path = tempdir / ("uploaded" + suffix)
                uploaded.seek(0)
                with media_path.open("wb") as dst:
                    shutil.copyfileobj(uploaded, dst)
                result_name = Path(uploaded.name).stem
            else:
                with st.spinner("جارٍ فحص الرابط واستخراج صوت الفيديو..."):
                    media_path, result_name = download_audio_from_link(video_url, tempdir)

            with st.spinner("جارٍ تجهيز نموذج التفريغ وتحليل الصوت..."):
                model = get_model(MODELS[model_label])
                segments, info = model.transcribe(
                    str(media_path),
                    language=LANGUAGES[lang_label],
                    beam_size=3,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                progress = st.progress(0.0, text="جارٍ تحويل الكلام إلى نص...")
                chunks = []
                duration = max(float(info.duration or 0), 1.0)
                for segment in segments:
                    clean = segment.text.strip()
                    if clean:
                        chunks.append({"start": float(segment.start), "end": float(segment.end), "text": clean})
                    progress.progress(
                        min(1.0, max(0.0, float(segment.end) / duration)),
                        text="جارٍ تحويل الكلام إلى نص...",
                    )
                progress.empty()

        if not chunks:
            st.warning("لم أتعرف على كلام واضح في المقطع. تأكد من وجود صوت واضح وجرب تحديد اللغة.")
        else:
            full_text = " ".join(chunk["text"] for chunk in chunks)
            st.session_state["transcription"] = {"chunks": chunks, "filename": result_name}
            st.session_state["editable_transcript"] = full_text
            st.success("تم التفريغ بنجاح ✅")
    except ValueError as exc:
        st.error(str(exc))
    except DownloadError as exc:
        st.error(
            "تعذّر استخراج الفيديو من الرابط. ربما المنشور خاص، أو يحتاج تسجيل دخول، "
            "أو أن المنصة تمنع الوصول من الخادم. يمكنك تنزيل الفيديو بنفسك ثم رفعه من خيار «رفع ملف»."
        )
        with st.expander("تفاصيل تقنية للمساعدة في حل المشكلة"):
            st.code(f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        st.error("تعذرت معالجة الفيديو. تحقق من الرابط أو جرب رفع الملف مباشرة.")
        with st.expander("تفاصيل تقنية للمساعدة في حل المشكلة"):
            st.code(f"{type(exc).__name__}: {exc}")

result = st.session_state.get("transcription")
if result:
    st.divider()
    st.markdown('<div class="step-header">٢. النص الناتج — يمكنك تعديله قبل التنزيل</div>', unsafe_allow_html=True)
    st.text_area("التفريغ النصي", key="editable_transcript", height=280)
    edited_text = st.session_state["editable_transcript"]
    st.caption("التعديل اليدوي يظهر في TXT فقط، أما SRT وVTT فيحتفظان بالنص والتوقيت الأصليين.")

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
st.caption("مدعوم بنموذج Whisper مفتوح المصدر ومكتبة yt-dlp. بعض الروابط قد لا تُفتح بسبب سياسات أو قيود المنصة.")
