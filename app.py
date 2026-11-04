"""
OmniStream - Ultimate Downloader
Single-file FastAPI app: YouTube, Instagram Reels and Facebook video downloader powered by yt-dlp.

Install:  pip install fastapi uvicorn yt-dlp
Run:      python app.py
Open:     http://127.0.0.1:8000

Optional: install ffmpeg (https://ffmpeg.org) on your PATH to enable merged
1080p+ MP4 downloads and true MP3 conversion. Without ffmpeg the app falls back
to single-file MP4 streams and M4A audio automatically.
"""

import os
import re
import shutil
import tempfile
from urllib.parse import quote, urlparse

import uvicorn
import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

app = FastAPI(title="OmniStream", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Length"],
)

ALLOWED_HOSTS = (
    "youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
    "instagram.com",
    "facebook.com",
    "fb.watch",
    "fb.com",
)

STANDARD_HEIGHTS = {240, 360, 480, 720, 1080, 1440, 2160}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class InfoRequest(BaseModel):
    url: str


class DownloadRequest(BaseModel):
    url: str
    quality: str = "best"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def validate_url(raw: str) -> str:
    url = (raw or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Please paste a video link first.")
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host or "." not in host:
        raise HTTPException(
            status_code=400,
            detail="That doesn't look like a valid link. Please check the URL and try again.",
        )
    if not any(host == d or host.endswith("." + d) for d in ALLOWED_HOSTS):
        raise HTTPException(
            status_code=400,
            detail="Unsupported website. Please use a YouTube, Instagram or Facebook link.",
        )
    return url


def detect_platform(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if "youtu" in host:
        return "YouTube"
    if "instagram" in host:
        return "Instagram"
    return "Facebook"


def clean_error(exc: Exception) -> str:
    msg = ANSI_RE.sub("", str(exc))
    msg = re.sub(r"^\s*ERROR:\s*", "", msg).strip()
    msg = msg.replace("\n", " ")
    if not msg:
        return "Something went wrong while reading that link."
    lowered = msg.lower()
    if "unsupported url" in lowered:
        return "This link doesn't point to a downloadable video."
    if "private" in lowered or "login" in lowered or "sign in" in lowered or "cookies" in lowered:
        return "This video is private or requires a login, so it can't be downloaded."
    if "unavailable" in lowered or "not available" in lowered or "removed" in lowered:
        return "This video is unavailable or has been removed."
    if len(msg) > 300:
        msg = msg[:297] + "..."
    return msg


def safe_filename(title: str, ext: str) -> str:
    base = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", title or "video")
    base = re.sub(r"\s+", " ", base).strip(" .")
    if not base:
        base = "video"
    return f"{base[:120]}.{ext}"


def guess_media_type(ext: str) -> str:
    return {
        "mp4": "video/mp4",
        "webm": "video/webm",
        "mkv": "video/x-matroska",
        "mp3": "audio/mpeg",
        "m4a": "audio/mp4",
        "opus": "audio/ogg",
        "ogg": "audio/ogg",
    }.get(ext.lower(), "application/octet-stream")


def extract_info_safe(url: str, download: bool = False, extra_opts: dict | None = None) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 25,
        "retries": 3,
        "nocheckcertificate": False,
    }
    if extra_opts:
        opts.update(extra_opts)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=download)
    if info is None:
        raise ValueError("No media information could be extracted.")
    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise ValueError("No video found at this link.")
        info = entries[0]
    return info


def build_format_options(info: dict, ffmpeg: bool) -> list:
    heights = set()
    for f in info.get("formats") or []:
        vcodec = f.get("vcodec")
        h = f.get("height")
        if h and vcodec not in (None, "none"):
            heights.add(int(h))

    chosen = [h for h in sorted(heights, reverse=True) if h in STANDARD_HEIGHTS]
    if not chosen:
        chosen = sorted(heights, reverse=True)[:5]
    chosen = chosen[:6]

    options = []
    for h in chosen:
        options.append({"id": str(h), "label": f"{h}p MP4", "type": "video", "ext": "mp4"})

    if not options and any(
        (f.get("vcodec") not in (None, "none")) for f in (info.get("formats") or [])
    ):
        options.append({"id": "best", "label": "Best Quality MP4", "type": "video", "ext": "mp4"})
    elif not options and info.get("url"):
        options.append({"id": "best", "label": "Best Quality MP4", "type": "video", "ext": "mp4"})

    options.append(
        {
            "id": "audio",
            "label": "Audio MP3" if ffmpeg else "Audio M4A",
            "type": "audio",
            "ext": "mp3" if ffmpeg else "m4a",
        }
    )
    return options


def build_download_opts(quality: str, tmp_dir: str, ffmpeg: bool) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 25,
        "retries": 5,
        "fragment_retries": 5,
        "restrictfilenames": True,
        "outtmpl": os.path.join(tmp_dir, "%(id)s.%(ext)s"),
        "overwrites": True,
    }

    if quality == "audio":
        if ffmpeg:
            opts["format"] = "bestaudio/best"
            opts["postprocessors"] = [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ]
        else:
            opts["format"] = "bestaudio[ext=m4a]/bestaudio/best"
        return opts

    height = int(quality) if quality.isdigit() else None
    if ffmpeg:
        if height:
            opts["format"] = (
                f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/"
                f"bestvideo[height<={height}]+bestaudio/"
                f"best[height<={height}]/best"
            )
        else:
            opts["format"] = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best"
        opts["merge_output_format"] = "mp4"
    else:
        if height:
            opts["format"] = (
                f"best[height<={height}][ext=mp4]/best[height<={height}]/best[ext=mp4]/best"
            )
        else:
            opts["format"] = "best[ext=mp4]/best"
    return opts


def find_output_file(tmp_dir: str) -> str:
    candidates = []
    for name in os.listdir(tmp_dir):
        if name.endswith((".part", ".ytdl", ".temp", ".tmp")):
            continue
        path = os.path.join(tmp_dir, name)
        if os.path.isfile(path):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("The download finished but no file was produced.")
    candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return candidates[0]


# --------------------------------------------------------------------------- #
# API routes
# --------------------------------------------------------------------------- #
@app.post("/api/info")
def api_info(req: InfoRequest):
    url = validate_url(req.url)
    ffmpeg = has_ffmpeg()
    try:
        info = extract_info_safe(url, download=False)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=clean_error(exc))

    thumbnail = info.get("thumbnail")
    if not thumbnail:
        thumbs = [t for t in (info.get("thumbnails") or []) if t.get("url")]
        if thumbs:
            thumbnail = thumbs[-1]["url"]

    return {
        "title": info.get("title") or "Untitled video",
        "thumbnail": thumbnail,
        "duration": info.get("duration"),
        "uploader": info.get("uploader") or info.get("channel") or "",
        "platform": detect_platform(url),
        "ffmpeg": ffmpeg,
        "formats": build_format_options(info, ffmpeg),
    }


@app.post("/api/download")
def api_download(req: DownloadRequest):
    url = validate_url(req.url)
    quality = (req.quality or "best").strip().lower()
    if not (quality in ("best", "audio") or quality.isdigit()):
        raise HTTPException(status_code=400, detail="Invalid quality selection.")

    ffmpeg = has_ffmpeg()
    tmp_dir = tempfile.mkdtemp(prefix="omnistream_")

    try:
        opts = build_download_opts(quality, tmp_dir, ffmpeg)
        info = extract_info_safe(url, download=True, extra_opts=opts)
        path = find_output_file(tmp_dir)
    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=clean_error(exc))

    ext = os.path.splitext(path)[1].lstrip(".") or "mp4"
    filename = safe_filename(info.get("title") or "video", ext)
    size = os.path.getsize(path)

    def iter_file():
        try:
            with open(path, "rb") as fh:
                while True:
                    chunk = fh.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    headers = {
        "Content-Disposition": (
            f"attachment; filename=\"download.{ext}\"; filename*=UTF-8''{quote(filename)}"
        ),
        "Content-Length": str(size),
        "Cache-Control": "no-store",
    }
    return StreamingResponse(
        iter_file(),
        media_type=guess_media_type(ext),
        headers=headers,
        background=BackgroundTask(shutil.rmtree, tmp_dir, True),
    )


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>OmniStream - Ultimate Downloader</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
  <script src="https://unpkg.com/lucide@0.469.0/dist/umd/lucide.min.js"></script>
  <script>
    tailwind.config = {
      theme: { extend: { fontFamily: { sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'] } } }
    };
  </script>
  <style>
    body { background: #070b17; }
    .glass {
      background: rgba(30, 41, 59, 0.45);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
      border: 1px solid rgba(148, 163, 184, 0.15);
    }
    .glow-text {
      background: linear-gradient(90deg, #818cf8, #a78bfa, #e879f9);
      -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .orb { position: fixed; border-radius: 9999px; filter: blur(110px); opacity: .35; pointer-events: none; z-index: 0; }
    @keyframes shimmer { 0% { background-position: -400px 0; } 100% { background-position: 400px 0; } }
    .skeleton {
      background: linear-gradient(90deg, rgba(51,65,85,.5) 25%, rgba(100,116,139,.45) 50%, rgba(51,65,85,.5) 75%);
      background-size: 800px 100%; animation: shimmer 1.4s infinite linear; border-radius: .75rem;
    }
    @keyframes toastIn { from { opacity: 0; transform: translateX(24px); } to { opacity: 1; transform: translateX(0); } }
    @keyframes toastOut { from { opacity: 1; } to { opacity: 0; transform: translateX(24px); } }
    .toast-in { animation: toastIn .28s ease-out forwards; }
    .toast-out { animation: toastOut .25s ease-in forwards; }
    @keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
    .fade-up { animation: fadeUp .45s ease-out both; }
    .spin { animation: spin 1s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body class="font-sans text-slate-200 min-h-screen antialiased relative overflow-x-hidden">
  <div class="orb bg-indigo-600" style="top:-120px; left:-100px; width:420px; height:420px;"></div>
  <div class="orb bg-fuchsia-600" style="bottom:-140px; right:-100px; width:460px; height:460px;"></div>
  <div class="orb bg-violet-700" style="top:35%; left:55%; width:300px; height:300px; opacity:.2;"></div>

  <!-- Toasts -->
  <div id="toastRoot" class="fixed top-4 right-4 z-50 flex flex-col gap-3 w-[calc(100%-2rem)] max-w-sm pointer-events-none"></div>

  <main class="relative z-10 max-w-3xl mx-auto px-4 sm:px-6 py-10 sm:py-16">
    <!-- Hero -->
    <section class="text-center fade-up">
      <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full glass text-xs font-medium text-indigo-300 mb-6">
        <i data-lucide="sparkles" class="w-3.5 h-3.5"></i> Fast &bull; Free &bull; No sign-up
      </div>
      <h1 class="text-4xl sm:text-6xl font-extrabold tracking-tight leading-tight">
        <span class="glow-text">OmniStream</span><br class="sm:hidden" />
        <span class="text-white"> - Ultimate Downloader</span>
      </h1>
      <p class="mt-4 text-slate-400 text-base sm:text-lg max-w-xl mx-auto">
        Paste a link, pick your quality, and download. One clean tool for all your favorite platforms.
      </p>

      <div class="mt-7 flex items-center justify-center gap-3">
        <div class="glass rounded-2xl px-4 py-2.5 flex items-center gap-2 text-sm font-medium">
          <svg class="w-5 h-5 text-red-500" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M23.5 6.2a3 3 0 0 0-2.1-2.1C19.5 3.6 12 3.6 12 3.6s-7.5 0-9.4.5A3 3 0 0 0 .5 6.2C0 8.1 0 12 0 12s0 3.9.5 5.8a3 3 0 0 0 2.1 2.1c1.9.5 9.4.5 9.4.5s7.5 0 9.4-.5a3 3 0 0 0 2.1-2.1c.5-1.9.5-5.8.5-5.8s0-3.9-.5-5.8ZM9.6 15.6V8.4l6.2 3.6-6.2 3.6Z"/></svg>
          YouTube
        </div>
        <div class="glass rounded-2xl px-4 py-2.5 flex items-center gap-2 text-sm font-medium">
          <svg class="w-5 h-5 text-pink-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="5"/><circle cx="12" cy="12" r="4"/><circle cx="17.5" cy="6.5" r="1" fill="currentColor"/></svg>
          Instagram
        </div>
        <div class="glass rounded-2xl px-4 py-2.5 flex items-center gap-2 text-sm font-medium">
          <svg class="w-5 h-5 text-blue-500" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M24 12.07C24 5.4 18.63 0 12 0S0 5.4 0 12.07C0 18.1 4.39 23.1 10.13 24v-8.44H7.08v-3.49h3.05V9.41c0-3.02 1.79-4.69 4.53-4.69 1.31 0 2.69.24 2.69.24v2.97h-1.51c-1.49 0-1.96.93-1.96 1.88v2.26h3.33l-.53 3.49h-2.8V24C19.61 23.1 24 18.1 24 12.07Z"/></svg>
          Facebook
        </div>
      </div>
    </section>

    <!-- Step 1: Input -->
    <section class="mt-10 fade-up" style="animation-delay:.1s">
      <div class="glass rounded-3xl p-3 sm:p-4 shadow-2xl shadow-indigo-900/20">
        <div class="flex flex-col sm:flex-row gap-3">
          <div class="relative flex-1">
            <i data-lucide="link" class="w-5 h-5 text-slate-500 absolute left-4 top-1/2 -translate-y-1/2"></i>
            <input id="urlInput" type="url" inputmode="url" autocomplete="off" spellcheck="false"
              placeholder="Paste a YouTube, Instagram or Facebook video link..."
              class="w-full bg-slate-900/70 border border-slate-700/70 focus:border-indigo-400 focus:ring-4 focus:ring-indigo-500/20 outline-none rounded-2xl pl-12 pr-28 py-4 text-base placeholder-slate-500 transition" />
            <button id="pasteBtn" type="button"
              class="absolute right-2 top-1/2 -translate-y-1/2 inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-sm font-medium text-slate-200 transition">
              <i data-lucide="clipboard-paste" class="w-4 h-4"></i> Paste
            </button>
          </div>
          <button id="fetchBtn" type="button"
            class="inline-flex items-center justify-center gap-2 px-7 py-4 rounded-2xl font-semibold text-white bg-gradient-to-r from-indigo-500 via-violet-500 to-fuchsia-500 hover:from-indigo-400 hover:via-violet-400 hover:to-fuchsia-400 shadow-lg shadow-indigo-500/30 hover:shadow-indigo-500/50 active:scale-[.98] transition disabled:opacity-60 disabled:cursor-not-allowed">
            <i data-lucide="zap" class="w-5 h-5"></i> Fetch Video
          </button>
        </div>
      </div>

      <!-- Error banner -->
      <div id="errorBanner" class="hidden mt-4 glass rounded-2xl border-red-500/40 px-4 py-3 flex items-start gap-3" style="border-color: rgba(239,68,68,.45); background: rgba(127,29,29,.25);">
        <i data-lucide="alert-triangle" class="w-5 h-5 text-red-400 mt-0.5 shrink-0"></i>
        <p id="errorText" class="text-sm text-red-200 flex-1"></p>
        <button id="errorClose" type="button" class="text-red-300 hover:text-white" aria-label="Dismiss error">
          <i data-lucide="x" class="w-4 h-4"></i>
        </button>
      </div>
    </section>

    <!-- Step 2: Loading skeleton -->
    <section id="loadingCard" class="hidden mt-8">
      <div class="glass rounded-3xl p-5 sm:p-6">
        <div class="flex items-center gap-3 mb-5 text-indigo-300 font-medium">
          <i data-lucide="loader-2" class="w-5 h-5 spin"></i>
          <span>Extracting media streams...</span>
        </div>
        <div class="flex flex-col sm:flex-row gap-5">
          <div class="skeleton w-full sm:w-64 h-40 shrink-0"></div>
          <div class="flex-1 space-y-3">
            <div class="skeleton h-6 w-11/12"></div>
            <div class="skeleton h-6 w-3/5"></div>
            <div class="skeleton h-4 w-1/4 mt-4"></div>
            <div class="flex gap-3 pt-3">
              <div class="skeleton h-11 w-28"></div>
              <div class="skeleton h-11 w-28"></div>
              <div class="skeleton h-11 w-28"></div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Step 3: Result card -->
    <section id="resultCard" class="hidden mt-8">
      <div class="glass rounded-3xl overflow-hidden shadow-2xl shadow-violet-900/20 fade-up">
        <div class="flex flex-col sm:flex-row">
          <div class="relative sm:w-72 shrink-0 bg-slate-900">
            <img id="thumb" alt="Video thumbnail" referrerpolicy="no-referrer" class="w-full h-52 sm:h-full object-cover" />
            <span id="durationBadge" class="absolute bottom-3 right-3 inline-flex items-center gap-1 bg-black/75 text-white text-xs font-semibold px-2 py-1 rounded-lg">
              <i data-lucide="clock" class="w-3 h-3"></i><span id="durationText">0:00</span>
            </span>
            <span id="platformBadge" class="absolute top-3 left-3 bg-indigo-500/90 text-white text-xs font-semibold px-2.5 py-1 rounded-lg"></span>
          </div>
          <div class="p-5 sm:p-6 flex-1 min-w-0">
            <h2 id="videoTitle" class="text-lg sm:text-xl font-bold text-white leading-snug break-words" style="display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;"></h2>
            <p id="videoUploader" class="mt-1 text-sm text-slate-400"></p>

            <p class="mt-5 mb-3 text-xs font-semibold tracking-wider uppercase text-slate-500">Choose quality</p>
            <div id="formatList" class="grid grid-cols-2 sm:grid-cols-3 gap-3"></div>

            <p id="ffmpegHint" class="hidden mt-4 text-xs text-amber-300/80">
              Tip: install <span class="font-semibold">ffmpeg</span> on the server to unlock merged HD video and true MP3 conversion.
            </p>
          </div>
        </div>
      </div>
    </section>

    <footer class="mt-14 text-center text-xs text-slate-500 space-y-1">
      <p>Only download content you own or have permission to save, and respect each platform's terms of service.</p>
      <p>&copy; OmniStream &bull; Powered by yt-dlp</p>
    </footer>
  </main>

  <script>
    (function () {
      var urlInput = document.getElementById('urlInput');
      var pasteBtn = document.getElementById('pasteBtn');
      var fetchBtn = document.getElementById('fetchBtn');
      var errorBanner = document.getElementById('errorBanner');
      var errorText = document.getElementById('errorText');
      var errorClose = document.getElementById('errorClose');
      var loadingCard = document.getElementById('loadingCard');
      var resultCard = document.getElementById('resultCard');
      var toastRoot = document.getElementById('toastRoot');
      var formatList = document.getElementById('formatList');

      var currentUrl = '';
      var busy = false;

      function icons() {
        if (window.lucide && typeof window.lucide.createIcons === 'function') {
          window.lucide.createIcons();
        }
      }

      /* ---------- Toasts ---------- */
      function toast(message, type) {
        type = type || 'success';
        var styles = {
          success: { icon: 'check-circle', color: 'text-emerald-400', border: 'rgba(16,185,129,.45)' },
          error:   { icon: 'alert-triangle', color: 'text-red-400', border: 'rgba(239,68,68,.45)' },
          info:    { icon: 'info', color: 'text-indigo-300', border: 'rgba(129,140,248,.45)' }
        };
        var s = styles[type] || styles.info;
        var el = document.createElement('div');
        el.className = 'toast-in pointer-events-auto glass rounded-2xl px-4 py-3 flex items-start gap-3 shadow-xl';
        el.style.borderColor = s.border;
        el.style.background = 'rgba(15,23,42,.88)';
        var iconEl = document.createElement('i');
        iconEl.setAttribute('data-lucide', s.icon);
        iconEl.className = 'w-5 h-5 mt-0.5 shrink-0 ' + s.color;
        var text = document.createElement('p');
        text.className = 'text-sm text-slate-100 flex-1';
        text.textContent = message;
        el.appendChild(iconEl);
        el.appendChild(text);
        toastRoot.appendChild(el);
        icons();
        setTimeout(function () {
          el.classList.remove('toast-in');
          el.classList.add('toast-out');
          setTimeout(function () { if (el.parentNode) el.parentNode.removeChild(el); }, 260);
        }, 4200);
      }

      /* ---------- Error banner ---------- */
      function showError(message) {
        errorText.textContent = message;
        errorBanner.classList.remove('hidden');
        toast(message, 'error');
      }
      function hideError() { errorBanner.classList.add('hidden'); }
      errorClose.addEventListener('click', hideError);

      /* ---------- Helpers ---------- */
      function formatDuration(seconds) {
        if (!seconds || isNaN(seconds)) return '--:--';
        seconds = Math.floor(seconds);
        var h = Math.floor(seconds / 3600);
        var m = Math.floor((seconds % 3600) / 60);
        var s = seconds % 60;
        var pad = function (n) { return n < 10 ? '0' + n : '' + n; };
        return h > 0 ? h + ':' + pad(m) + ':' + pad(s) : m + ':' + pad(s);
      }

      function looksLikeUrl(value) {
        return /^(https?:\/\/)?([a-z0-9-]+\.)+[a-z]{2,}(\/|$|\?|#)/i.test(value.trim());
      }

      function parseFilename(header, fallback) {
        if (!header) return fallback;
        var star = /filename\*=UTF-8''([^;]+)/i.exec(header);
        if (star) {
          try { return decodeURIComponent(star[1]); } catch (e) { /* ignore */ }
        }
        var plain = /filename="?([^";]+)"?/i.exec(header);
        return plain ? plain[1] : fallback;
      }

      async function readError(res, fallback) {
        try {
          var data = await res.json();
          if (data && typeof data.detail === 'string') return data.detail;
          if (data && Array.isArray(data.detail) && data.detail[0] && data.detail[0].msg) return data.detail[0].msg;
        } catch (e) { /* ignore */ }
        return fallback;
      }

      function setFetching(on) {
        busy = on;
        fetchBtn.disabled = on;
        urlInput.disabled = on;
        fetchBtn.innerHTML = on
          ? '<i data-lucide="loader-2" class="w-5 h-5 spin"></i> Fetching...'
          : '<i data-lucide="zap" class="w-5 h-5"></i> Fetch Video';
        icons();
      }

      /* ---------- Paste ---------- */
      pasteBtn.addEventListener('click', async function () {
        try {
          var text = await navigator.clipboard.readText();
          if (text && text.trim()) {
            urlInput.value = text.trim();
            hideError();
            toast('Link pasted from clipboard.', 'info');
            urlInput.focus();
          } else {
            toast('Your clipboard is empty.', 'info');
          }
        } catch (e) {
          toast('Clipboard access was blocked. Please paste the link manually (Ctrl/Cmd + V).', 'error');
          urlInput.focus();
        }
      });

      /* ---------- Fetch info ---------- */
      async function fetchInfo() {
        if (busy) return;
        var value = urlInput.value.trim();
        hideError();

        if (!value) {
          showError('Please paste a video link first.');
          return;
        }
        if (!looksLikeUrl(value)) {
          showError("That doesn't look like a valid link. Please check the URL and try again.");
          return;
        }

        resultCard.classList.add('hidden');
        loadingCard.classList.remove('hidden');
        setFetching(true);

        try {
          var res = await fetch('/api/info', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: value })
          });
          if (!res.ok) {
            throw new Error(await readError(res, 'Could not fetch this video. Please try another link.'));
          }
          var data = await res.json();
          currentUrl = value;
          renderResult(data);
        } catch (err) {
          loadingCard.classList.add('hidden');
          showError(err && err.message ? err.message : 'Network error. Please check your connection and try again.');
        } finally {
          setFetching(false);
        }
      }

      fetchBtn.addEventListener('click', fetchInfo);
      urlInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') { e.preventDefault(); fetchInfo(); }
      });

      /* ---------- Render result ---------- */
      function renderResult(data) {
        var thumb = document.getElementById('thumb');
        thumb.onerror = function () {
          thumb.onerror = null;
          thumb.src = 'data:image/svg+xml;utf8,' + encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="100%" height="100%" fill="#0f172a"/><text x="50%" y="50%" fill="#475569" font-family="sans-serif" font-size="22" text-anchor="middle" dominant-baseline="middle">No preview available</text></svg>'
          );
        };
        thumb.src = data.thumbnail || '';
        if (!data.thumbnail) thumb.onerror();

        document.getElementById('videoTitle').textContent = data.title || 'Untitled video';
        document.getElementById('videoUploader').textContent = data.uploader ? 'by ' + data.uploader : '';
        document.getElementById('platformBadge').textContent = data.platform || '';
        document.getElementById('durationText').textContent = formatDuration(data.duration);
        document.getElementById('durationBadge').style.display = data.duration ? '' : 'none';
        document.getElementById('ffmpegHint').classList.toggle('hidden', !!data.ffmpeg);

        formatList.innerHTML = '';
        (data.formats || []).forEach(function (f) {
          var btn = document.createElement('button');
          btn.type = 'button';
          btn.dataset.quality = f.id;
          btn.dataset.label = f.label;
          var isAudio = f.type === 'audio';
          btn.className = 'group inline-flex items-center justify-center gap-2 px-4 py-3 rounded-xl text-sm font-semibold border transition active:scale-[.97] disabled:opacity-50 disabled:cursor-not-allowed ' +
            (isAudio
              ? 'bg-fuchsia-500/10 border-fuchsia-400/40 text-fuchsia-200 hover:bg-fuchsia-500/25 hover:border-fuchsia-300'
              : 'bg-indigo-500/10 border-indigo-400/40 text-indigo-100 hover:bg-indigo-500/25 hover:border-indigo-300');
          btn.innerHTML = '<i data-lucide="' + (isAudio ? 'music' : 'film') + '" class="w-4 h-4"></i><span></span>';
          btn.querySelector('span').textContent = f.label;
          btn.addEventListener('click', function () { startDownload(btn, f); });
          formatList.appendChild(btn);
        });

        loadingCard.classList.add('hidden');
        resultCard.classList.remove('hidden');
        icons();
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }

      /* ---------- Download ---------- */
      async function startDownload(button, format) {
        if (busy) return;
        busy = true;
        var all = formatList.querySelectorAll('button');
        all.forEach(function (b) { b.disabled = true; });
        var originalHtml = button.innerHTML;
        button.innerHTML = '<i data-lucide="loader-2" class="w-4 h-4 spin"></i><span>Preparing...</span>';
        icons();
        toast('Preparing ' + format.label + '. Large files can take a moment...', 'info');

        try {
          var res = await fetch('/api/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: currentUrl, quality: format.id })
          });
          if (!res.ok) {
            throw new Error(await readError(res, 'Download failed. Please try again.'));
          }
          var blob = await res.blob();
          var fallbackName = 'omnistream-download.' + (format.ext || 'mp4');
          var filename = parseFilename(res.headers.get('Content-Disposition'), fallbackName);

          var objectUrl = URL.createObjectURL(blob);
          var a = document.createElement('a');
          a.href = objectUrl;
          a.download = filename;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          setTimeout(function () { URL.revokeObjectURL(objectUrl); }, 60000);

          toast('Download started: ' + filename, 'success');
        } catch (err) {
          var msg = err && err.message ? err.message : 'Download failed. Please try again.';
          showError(msg);
        } finally {
          button.innerHTML = originalHtml;
          all.forEach(function (b) { b.disabled = false; });
          busy = false;
          icons();
        }
      }

      icons();
    })();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_PAGE)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    uvicorn.run(app, host=host, port=port)
