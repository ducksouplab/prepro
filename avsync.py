"""
Audio-video synchronisation of DuckSoup recordings.

The problem
-----------
DuckSoup writes each recording with GStreamer (appsrc -> rtpbin -> depayloader -> mp4mux). While a
participant waits for the partner, their WebRTC tracks are already arriving. When the pipeline
starts (`pipeline_started` in the logs), the audio that was queued during the wait goes into the file,
but the video only starts at the first key frame (the H264 depayloader drops everything before it).
Both tracks are written from timestamp 0, so the audio of such a file is LATE relative to its own video
by the amount of queued audio. The offset is constant over the file (the same in every minute of a
4-min recording) and grows with the wait: ~0 for waits < 0.5 s, 0.6-1.2 s for 2-4 s, 1.2-3 s for 8-15 s
(2024 prolific data, 604 files). Queued audio cannot be longer than the wait, which bounds the offset.

How the offset is estimated (per raw file)
------------------------------------------
1. Durations (physical evidence, `duration_offset`). Nothing can be recorded after the pipeline stops, so
   audio beyond the wall-clock recording time (`pipeline_stopped` - `pipeline_started` in the logs) was
   recorded before the start: duration offset = audio duration - wall time. Without a usable wall time (no
   stop event, or a video longer than it, i.e. a stop that belongs to another recording): audio - video.
   An excess longer than the wait (`pipeline_started` - audio `in_track_received`) + WAIT_MARGIN_S cannot
   be queued audio and is not used. Precise (~20 ms) but blind to audio missing elsewhere (audio that ends
   early hides part of the offset), so the content decides when it can.
2. Content (independent check). The audio envelope is cross-correlated with a mouth signal of the same
   file. Talking moves the mouth, so the correlation peaks at the audio-video lag. Precise to about one
   video frame; it includes the natural lead of the mouth over the sound (0-0.15 s). Mouth signals, best
   first (`mouth_backend="auto"` takes the first one available):
     - "mediapipe": jaw opening (`jawOpen` blendshape) of MediaPipe's face landmarker; needs the
       `mediapipe` package and the model file (`face_model`, the PREPRO_FACE_MODEL environment variable,
       or models/face_landmarker.task next to this module; download link in the README);
     - "opencv": motion of the lower half of the face found by OpenCV's frontal-face detector;
     - "frame": whole-frame motion (no face detection; also used when no face is found).
3. Decision (`decide`): the duration offset when a reliable content estimate confirms it (more precise);
   the content lag when a reliable content estimate disagrees; otherwise (content inconclusive or not
   measurable) the duration offset, flagged as unverified. Every value and the reason are in the manifest.
   Validation (2024 prolific data, 40 files of 10 problem interactions, MediaPipe): every reliable content
   estimate agreed with the same measurement on the rendered output (0 mismatches); the durations alone
   would have been wrong for 6 files (4 caught by the wait bound, 2 corrected by the content).

Only needs numpy and ffmpeg; MediaPipe and OpenCV are optional (better mouth signals).
"""

import os
import subprocess

import numpy as np

FPS = 30                  # analysis frame rate
WIDTH = 320               # analysis frame width (px)
MAX_LAG_S = 5.0           # largest offset searched / applied
SILENCE_DB = -90.0        # audio frames below this level are ignored (digital silence, gaps)
MIN_VALID_S = 30.0        # minimum analysable overlap of motion and audio
Z_STRONG = 4.0            # peak prominence above which a content estimate is reliable on its own
Z_WEAK = 2.5              # ... above which it is reliable if at least two thirds of the file agree
THIRDS_TOL_S = 0.1        # agreement of a third with the whole-file estimate
AGREE_TOL_S = 0.25        # content and duration estimates agree within this (the content lag includes the
                          # natural lead of the mouth over the sound, 0-0.15 s on in-sync DuckSoup files)
DURATION_MIN_S = 0.1      # smaller duration excesses are packet jitter, not an offset
VIDEO_WALL_TOL_S = 0.25   # a video longer than the wall time by more than this: the stop is not this recording's
WAIT_MARGIN_S = 0.3       # queued audio <= wait + this (event timing, packetisation)


# ---------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------
def _ffmpeg_input(path, start, duration):
    a = ["ffmpeg", "-v", "error", "-nostdin"]
    if start:
        a += ["-ss", f"{start:.6f}"]
    a += ["-i", path]
    if duration:
        a += ["-t", f"{duration:.6f}"]
    return a


def _video_size(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    w, h = (int(v) for v in out.strip().split("\n")[0].split(",")[:2])
    return w, h


def _frames(path, start, duration, width, pix_fmt, channels):
    """Frames at FPS on the timestamp grid from t = 0 (as align.py renders them), streamed from ffmpeg."""
    iw, ih = _video_size(path)
    h = max(2, int(round(ih * width / iw / 2)) * 2)
    p = subprocess.Popen(_ffmpeg_input(path, start, duration) + [
        "-map", "0:v:0", "-vf", f"fps=fps={FPS}:start_time=0,scale={width}:{h},format={pix_fmt}",
        "-f", "rawvideo", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = h * width * channels
    shape = (h, width, channels) if channels > 1 else (h, width)
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(shape)
    finally:
        p.stdout.close()
        p.wait()


def audio_envelope(path, n, start=0.0, duration=None, sr=16000):
    """Speech-band level (dB) per analysis frame; NaN for silence. Length n."""
    # aresample=async: samples placed at their timestamps from t = 0 (gaps filled with silence), as align.py
    # renders them; decoding the samples back to back would compress the audio over timestamp gaps
    raw = subprocess.run(_ffmpeg_input(path, start, duration) + [
        "-map", "0:a:0", "-af", f"aresample=async=1:first_pts=0,aresample={sr},highpass=f=150,lowpass=f=4000",
        "-ac", "1", "-f", "f32le", "-"], capture_output=True).stdout
    x = np.frombuffer(raw, np.float32)
    hop = sr // FPS
    x = x[:len(x) // hop * hop].reshape(-1, hop)
    e = 10 * np.log10((x.astype(np.float64) ** 2).mean(1) + 1e-12)
    out = np.full(n, np.nan)
    m = min(n, len(e))
    out[:m] = e[:m]
    out[out < SILENCE_DB] = np.nan
    return out


def _absdiff(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def face_model_path(face_model=None):
    """MediaPipe face landmarker model: argument, PREPRO_FACE_MODEL, or models/face_landmarker.task here."""
    for p in (face_model, os.environ.get("PREPRO_FACE_MODEL"),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")):
        if p and os.path.exists(p):
            return p
    return None


def available_backend(mouth_backend="auto", face_model=None):
    """The mouth backend that will actually be used ('mediapipe', 'opencv' or 'frame')."""
    order = ["mediapipe", "opencv", "frame"] if mouth_backend == "auto" else [mouth_backend, "frame"]
    for b in order:
        if b == "mediapipe":
            try:
                import mediapipe  # noqa: F401
            except ImportError:
                continue
            if face_model_path(face_model):
                return b
        elif b == "opencv":
            try:
                import cv2  # noqa: F401
                return b
            except ImportError:
                continue
        elif b == "frame":
            return b
    return "frame"


def mouth_signal(path, start=0.0, duration=None, mouth_backend="auto", face_model=None, width=WIDTH):
    """(mouth signal per analysis frame, whole-frame motion per frame, fraction of frames with a face, backend).

    The mouth signal is NaN where no face is found; whole-frame motion is always computed (cheap), as the
    fallback for files without a usable face."""
    backend = available_backend(mouth_backend, face_model)
    mouth, frame_motion, prev = [], [], None
    if backend == "mediapipe":
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=face_model_path(face_model)),
                                            running_mode=vision.RunningMode.VIDEO, num_faces=1,
                                            output_face_blendshapes=True)
        with vision.FaceLandmarker.create_from_options(opts) as lm:
            for i, f in enumerate(_frames(path, start, duration, max(width, 480), "rgb24", 3)):
                g = f.mean(2)
                frame_motion.append(np.abs(g - prev).mean() if prev is not None else np.nan)
                prev = g
                r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)),
                                        int(round(i * 1000 / FPS)))
                if r.face_blendshapes:
                    mouth.append(next(c.score for c in r.face_blendshapes[0] if c.category_name == "jawOpen"))
                else:
                    mouth.append(np.nan)
    elif backend == "opencv":
        import cv2
        cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        box, every = None, 10
        for i, f in enumerate(_frames(path, start, duration, width, "gray", 1)):
            if i % every == 0:
                side = max(16, f.shape[1] // 8)
                found = cas.detectMultiScale(f, 1.15, 4, minSize=(side, side))
                if len(found):
                    b = np.asarray(max(found, key=lambda r: r[2] * r[3]), float)
                    box = b if box is None else 0.7 * box + 0.3 * b
            val = np.nan
            if prev is not None and box is not None:
                x, y, w, h = box
                y0, y1 = int(y + 0.62 * h), int(min(f.shape[0], y + h))
                x0, x1 = int(max(0, x + 0.25 * w)), int(min(f.shape[1], x + 0.75 * w))
                if y1 - y0 >= 4 and x1 - x0 >= 4:
                    val = _absdiff(f[y0:y1, x0:x1], prev[y0:y1, x0:x1])
            mouth.append(val)
            frame_motion.append(_absdiff(f, prev) if prev is not None else np.nan)
            prev = f
    else:
        for f in _frames(path, start, duration, width, "gray", 1):
            frame_motion.append(_absdiff(f, prev) if prev is not None else np.nan)
            mouth.append(np.nan)
            prev = f
    mouth, frame_motion = np.asarray(mouth, float), np.asarray(frame_motion, float)
    face_frac = float(np.isfinite(mouth).mean()) if len(mouth) else 0.0
    return mouth, frame_motion, face_frac, backend


# ---------------------------------------------------------------------------------------------
# Lag
# ---------------------------------------------------------------------------------------------
def _prep(s, rate=FPS):
    """High-pass (1 s moving average removed), z-scored, NaN -> 0; returns (signal, valid mask)."""
    s = np.asarray(s, float).copy()
    m = np.isfinite(s)
    if m.sum() < 2 or len(s) < 2 * rate:
        return np.zeros_like(s), np.zeros_like(m)
    s[~m] = s[m].mean()
    s = s - np.convolve(s, np.ones(rate) / rate, mode="same")
    s[~m] = 0.0
    sd = s[m].std()
    return (s - s[m].mean()) / (sd if sd > 0 else 1.0), m


def xcorr_lag(mot, env, max_lag_s=MAX_LAG_S, rate=FPS):
    """Lag (s) of `env` relative to `mot` (> 0: env later), peak prominence z, number of analysable samples.
    Overlap-normalised, sub-sample (parabolic) peak. Both signals sampled at `rate` Hz."""
    L = int(round(max_lag_s * rate))
    n = len(mot)
    if n < 2 * L + 3:
        return np.nan, 0.0, 0
    a, ma = _prep(mot, rate)
    b, mb = _prep(env, rate)
    valid = int((ma & mb).sum())
    if valid < rate:
        return np.nan, 0.0, valid
    nf = 1 << int(np.ceil(np.log2(2 * n)))       # FFT cross-correlation (signals can be long at 200 Hz)
    c = np.fft.irfft(np.fft.rfft(b, nf) * np.conj(np.fft.rfft(a, nf)), nf)
    cnt = np.fft.irfft(np.fft.rfft(mb.astype(float), nf) * np.conj(np.fft.rfft(ma.astype(float), nf)), nf)
    idx = np.r_[nf - L:nf, 0:L + 1]                 # lags -L..L
    c = c[idx] / np.maximum(np.round(cnt[idx]), 1.0)
    i = int(np.argmax(c))
    frac = 0.0
    if 0 < i < len(c) - 1:
        den = c[i - 1] - 2 * c[i] + c[i + 1]
        frac = 0.5 * (c[i - 1] - c[i + 1]) / den if den < 0 else 0.0
    sd = c.std()
    z = float((c[i] - np.median(c)) / sd) if sd > 0 else 0.0
    return (i - L + frac) / rate, z, valid


def estimate_lag(mot, env, max_lag_s=MAX_LAG_S):
    """Whole-signal lag plus the lag of each third, and whether the estimate is reliable."""
    lag, z, valid = xcorr_lag(mot, env, max_lag_s)
    n = len(mot)
    thirds = []
    for k in range(3):
        sl = slice(k * n // 3, (k + 1) * n // 3)
        if (sl.stop - sl.start) >= 20 * FPS:
            thirds.append(xcorr_lag(mot[sl], env[sl], max_lag_s)[0])
    agree = sum(1 for t in thirds if np.isfinite(t) and np.isfinite(lag) and abs(t - lag) <= THIRDS_TOL_S)
    enough = bool(np.isfinite(lag) and valid >= MIN_VALID_S * FPS)
    reliable = enough and (z >= Z_STRONG or (z >= Z_WEAK and agree >= 2))
    # strong: prominent AND consistent over the file; needed to overrule the durations (see decide)
    strong = enough and z >= Z_STRONG and agree >= 2
    return dict(lag=float(lag), z=z, valid_s=valid / FPS, thirds=[round(float(t), 3) for t in thirds],
                thirds_agree=agree, reliable=reliable, strong=strong)


def measure(path, start=0.0, duration=None, max_lag_s=MAX_LAG_S, mouth_backend="auto", face_model=None):
    """Audio-video lag of a file (or of [start, start + duration]); positive: audio later than video.
    Returns the estimate_lag dict plus `roi` (the mouth backend used, or 'frame' when no usable face was
    found) and `face_frac`; or dict(error=...) with error 'no_video', 'no_audio' or 'too_short'
    (< MIN_VALID_S, too short to measure)."""
    try:
        mouth, frame_motion, face_frac, backend = mouth_signal(path, start, duration, mouth_backend, face_model)
    except (ValueError, IndexError):
        return dict(error="no_video")
    n = len(frame_motion)
    if n < 2:
        return dict(error="no_video")
    if n < MIN_VALID_S * FPS:
        return dict(error="too_short")
    env = audio_envelope(path, n, start, duration)
    if not np.isfinite(env).any():
        return dict(error="no_audio")
    use_face = backend != "frame" and face_frac >= 0.5
    out = estimate_lag(mouth if use_face else frame_motion, env, max_lag_s)
    out.update(roi=backend if use_face else "frame", face_frac=round(face_frac, 3))
    return out


# ---------------------------------------------------------------------------------------------
# Twins: the dry and wet files of the same recording
# ---------------------------------------------------------------------------------------------
TWIN_MAX_LAG_S = 10.0     # largest wet-vs-dry offset searched
TWIN_AUDIO_RATE = 200     # Hz, audio envelope rate for the twin comparison (5 ms)
TWIN_VIDEO_TOL_S = 1.5 / FPS   # thirds agreement for the video lag (identical content: within ~1 frame)
TWIN_AUDIO_TOL_S = 0.01   # ... for the audio lag
SAME_AUDIO_R = 0.9        # waveform match above which the wet audio is the dry microphone signal (no audio effect)


def thumbnail_components(path, ref_basis=None, k=8, size=(24, 18)):
    """Image content per analysis frame: tiny gray thumbnails projected on their first k principal components
    (head pose, position, lighting: what both twins share; compression noise is averaged out).
    ref_basis: (mean, components) from the dry twin, so that both files are projected on the same axes.
    Returns (scores (n, k), (mean, components))."""
    thumbs = np.array([f.astype(np.float32).ravel() for f in _frames(path, 0.0, None, size[0], "gray", 1)])
    if len(thumbs) < 2:
        raise ValueError("no video")
    if ref_basis is None:
        mean = thumbs.mean(0)
        sub = thumbs[::max(1, len(thumbs) // 3000)] - mean
        comps = np.linalg.svd(sub, full_matrices=False)[2][:k]
        ref_basis = (mean, comps)
    mean, comps = ref_basis
    return (thumbs - mean) @ comps.T, ref_basis


def multi_xcorr_lag(A, B, max_lag_s, rate):
    """xcorr_lag summed over the columns of A and B (z-scored components): lag, z, valid."""
    L = int(round(max_lag_s * rate))
    n = min(len(A), len(B))
    if n < 2 * L + 3:
        return np.nan, 0.0, 0
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    tot = np.zeros(2 * L + 1)
    for j in range(A.shape[1]):
        a, _ = _prep(A[:n, j], rate)
        b, _ = _prep(B[:n, j], rate)
        c = np.fft.irfft(np.fft.rfft(b, nf) * np.conj(np.fft.rfft(a, nf)), nf)
        tot += c[np.r_[nf - L:nf, 0:L + 1]] / n
    i = int(np.argmax(tot))
    frac = 0.0
    if 0 < i < len(tot) - 1:
        den = tot[i - 1] - 2 * tot[i] + tot[i + 1]
        frac = 0.5 * (tot[i - 1] - tot[i + 1]) / den if den < 0 else 0.0
    sd = tot.std()
    return (i - L + frac) / rate, (float((tot[i] - np.median(tot)) / sd) if sd > 0 else 0.0), n


def audio_envelope_fine(path, rate=TWIN_AUDIO_RATE, sr=16000):
    """Speech-band level (dB) at `rate` Hz, NaN for silence (as audio_envelope, finer)."""
    raw = subprocess.run(_ffmpeg_input(path, 0.0, None) + [
        "-map", "0:a:0", "-af", f"aresample=async=1:first_pts=0,aresample={sr},highpass=f=150,lowpass=f=4000",
        "-ac", "1", "-f", "f32le", "-"], capture_output=True).stdout
    x = np.frombuffer(raw, np.float32)
    hop = sr // rate
    x = x[:len(x) // hop * hop].reshape(-1, hop)
    e = 10 * np.log10((x.astype(np.float64) ** 2).mean(1) + 1e-12)
    e[e < SILENCE_DB] = np.nan
    return e


def _pcm(path, sr=16000):
    raw = subprocess.run(_ffmpeg_input(path, 0.0, None) + [
        "-map", "0:a:0", "-af", f"aresample=async=1:first_pts=0,aresample={sr}", "-ac", "1", "-f", "f32le", "-"],
        capture_output=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


def waveform_match(dry_path, wet_path, lag_s, sr=16000, n_chunks=12, chunk_s=1.0, search_s=1.5):
    """Is the wet audio the same signal as the dry audio (the same packets, no effect)? For n_chunks loud 1-s
    chunks of the dry audio, the best normalised correlation of the waveform with the wet audio within
    +-search_s of the expected lag (so that packets dropped by the wet muxer do not matter). Returns the median:
    ~1 for the same signal (decoded identically), low for any transformation (pitch shift, filter, effect)."""
    x, y = _pcm(dry_path, sr), _pcm(wet_path, sr)
    c, S = int(chunk_s * sr), int(search_s * sr)
    if len(x) < 4 * c or len(y) < 4 * c:
        return np.nan
    energy = (x[:len(x) // c * c].reshape(-1, c) ** 2).mean(1)
    order = [i for i in np.argsort(-energy) if energy[i] > 0][:n_chunks]
    rs = []
    for i in order:
        seg = x[i * c:(i + 1) * c]
        mid = i * c + int(round(lag_s * sr))
        lo, hi = max(0, mid - S), min(len(y), mid + c + S)
        win = y[lo:hi]
        if len(win) < c + 1 or seg.std() == 0:
            continue
        nf = 1 << int(np.ceil(np.log2(len(win) + c)))
        cc = np.fft.irfft(np.fft.rfft(win, nf) * np.conj(np.fft.rfft(seg, nf)), nf)[:len(win) - c + 1]
        w2 = np.convolve(win ** 2, np.ones(c), "valid")[:len(cc)]
        rs.append(float(np.max(cc / np.sqrt(np.maximum(w2, 1e-12) * (seg ** 2).sum()))))
    return float(np.median(rs)) if rs else np.nan


def _consistent_lag(a, b, rate, max_lag_s, tol):
    """Whole-signal lag of b relative to a (1-d signals, or 2-d: summed over columns), lags of the thirds,
    reliability (z >= Z_STRONG and >= 2 thirds within tol) and drift (the thirds differ by more than tol)."""
    f = multi_xcorr_lag if np.ndim(a) == 2 else xcorr_lag
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    lag, z, valid = f(a, b, max_lag_s, rate)
    thirds = [f(a[k * n // 3:(k + 1) * n // 3], b[k * n // 3:(k + 1) * n // 3], max_lag_s, rate)[0]
              for k in range(3) if n // 3 >= 2 * max_lag_s * rate + 3]
    agree = sum(1 for t in thirds if np.isfinite(t) and np.isfinite(lag) and abs(t - lag) <= tol)
    finite = [t for t in thirds if np.isfinite(t)]
    drift = bool(len(finite) >= 2 and max(finite) - min(finite) > tol)
    reliable = bool(np.isfinite(lag) and valid >= MIN_VALID_S * rate and z >= Z_STRONG and (agree >= 2 or not thirds))
    return dict(lag=float(lag), z=z, thirds=[round(float(t), 3) for t in thirds], reliable=reliable, drift=drift)


def twin_lags(dry_path, wet_path, max_lag_s=TWIN_MAX_LAG_S):
    """Offsets of a wet file relative to its dry twin (same camera and microphone; DuckSoup writes them with
    separate muxers, and the wet video goes through the effect first, so the wet file can start with a different
    amount of queued audio and its video can be offset, while both are stamped from 0).

    'video': image content (thumbnail components); 'audio': speech envelope (5 ms). Each with lag (s, > 0: the
    content comes later in the wet file than in the dry file), z, thirds, reliable, drift; or error.
    'audio' also has `waveform_r` and `same_audio`: the wet audio is the dry microphone signal (no audio effect):
    the envelopes match (z >= Z_STRONG, even if the wet muxer dropped packets) AND so do the waveforms
    (waveform_r >= SAME_AUDIO_R; a voice effect such as a pitch shift keeps the envelope but not the waveform)."""
    out = {}
    try:
        A, basis = thumbnail_components(dry_path)
        B, _ = thumbnail_components(wet_path, basis)
        out["video"] = (_consistent_lag(A, B, FPS, max_lag_s, TWIN_VIDEO_TOL_S)
                        if min(len(A), len(B)) >= MIN_VALID_S * FPS else dict(error="too_short"))
    except (ValueError, IndexError):
        out["video"] = dict(error="no_video")
    a, b = audio_envelope_fine(dry_path), audio_envelope_fine(wet_path)
    if min(len(a), len(b)) >= MIN_VALID_S * TWIN_AUDIO_RATE:
        out["audio"] = _consistent_lag(a, b, TWIN_AUDIO_RATE, max_lag_s, TWIN_AUDIO_TOL_S)
        r = waveform_match(dry_path, wet_path, out["audio"]["lag"]) if np.isfinite(out["audio"]["lag"]) else np.nan
        out["audio"]["waveform_r"] = r
        out["audio"]["same_audio"] = bool(out["audio"]["z"] >= Z_STRONG and np.isfinite(r) and r >= SAME_AUDIO_R)
    else:
        out["audio"] = dict(error="too_short", same_audio=False)
    return out


# ---------------------------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------------------------
def duration_offset(audio_s, video_s, wall_s=None, wait_s=None):
    """Audio recorded before the video started, from the durations: (offset, basis, note).

    audio_s, video_s : stream durations of the raw file (video_s None: audio-only recording)
    wall_s           : pipeline_stopped - pipeline_started (None: no stop event)
    wait_s           : pipeline_started - audio in_track_received (None: unknown)
    basis: 'wall' (audio - wall time), 'video' (audio - video), or None (no evidence).
    Negative excesses (audio ending early) and jitter (< DURATION_MIN_S) are not offsets -> 0."""
    if audio_s is None:
        return 0.0, None, ""
    note = ""
    if wall_s is not None and (video_s is None or video_s <= wall_s + VIDEO_WALL_TOL_S):
        excess, basis = audio_s - wall_s, "wall"
    elif video_s is not None:
        excess, basis = audio_s - video_s, "video"
        if wall_s is not None:
            note = f"video {video_s - wall_s:.1f} s longer than the wall time: stop event not used"
    else:
        return 0.0, None, ""
    if excess <= DURATION_MIN_S:
        return 0.0, basis, note
    if wait_s is not None and excess > wait_s + WAIT_MARGIN_S:
        return 0.0, basis, (note + "; " if note else "") + (
            f"audio excess {excess:.3f} s longer than the wait {wait_s:.3f} s: not queued audio")
    return float(excess), basis, note


def decide(dur_offset, content, agree_tol=AGREE_TOL_S, max_offset=MAX_LAG_S):
    """Audio offset to remove (s, > 0: audio is late), its source and a note.

    dur_offset : from duration_offset (>= 0)
    content    : from measure / estimate_lag, or None (not measured)"""
    ok = bool(content) and not content.get("error") and content.get("reliable")
    c = content["lag"] if ok else None
    if ok and abs(c - dur_offset) <= agree_tol:
        # the content confirms the durations (incl. both ~0): the more precise duration value is used
        off, src, note = dur_offset, ("duration+content" if dur_offset > 0 else "content"), ""
    elif ok and abs(c) <= agree_tol:
        # in sync: the extra audio seen in the durations is not at the start
        off, src, note = 0.0, "content", f"duration excess {dur_offset:.3f} s not at the start"
    elif ok:
        off, src, note = c, "content", (f"duration estimate {dur_offset:.3f} s disagrees" if dur_offset > 0
                                        else "offset not visible in the durations")
    elif dur_offset > 0:
        off, src = dur_offset, "duration"
        note = "unverified (content check " + ((content or {}).get("error") or
                                               ("inconclusive" if content else "not run")) + ")"
    else:
        off, src, note = 0.0, "none", ""
    if abs(off) > max_offset:
        return 0.0, "none", f"offset {off:.3f} s larger than {max_offset} s: not applied"
    return float(off), src, note
