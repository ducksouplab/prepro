"""
Align and encode the recordings of one DuckSoup interaction in a single ffmpeg pass.

What this replaces (ducksoup.ds_process before this module):
    re_encode_folder (x264 CRF 18, scaled to 1280x720 whatever the aspect ratio)
    -> change_fps_folder (x264 default CRF 23)
    -> trim_folder (x264 default CRF 23), i.e. three lossy generations of video and of audio, and:
    - trim_folder took the start of each recording from the time in its file name;
    - it cut every file to start at the LATEST file start of the folder.

Two problems with that alignment:
    1. The time in the file name is when the participant joined, not when the recording started.
       DuckSoup starts recording everyone at the same moment, when the interaction starts
       (log message `pipeline_started` = `interaction_started`), usually a few seconds after
       the last participant joined. Cutting by the difference of join times therefore shifted
       the partners relative to each other by that difference (0-10 s).
    2. When a participant reconnects, DuckSoup writes a new file (-c-2-, -c-3-, ...). The latest
       file start is then the last reconnection: everything recorded before it was cut, for both
       partners, and segments that ended before it were dropped (0-byte files before prepro
       commit dd24404, silently skipped after).

What this module does, per interaction folder:
    - start of each recording = the participant's `pipeline_started` time in the interaction
      logs (the *.log files next to recordings/), matched to the file by its name time;
      if no log is found, the time in the file name (old behaviour, flagged in the manifest);
    - t0 = the start of the interaction (`interaction_started` in the log); without a log, the moment
      every participant is recording (latest first-segment start);
    - for each participant and each version (dry / wet), all segments are placed on one
      timeline starting at t0; reconnection gaps are filled with black frames and silence;
      each output ends at the end of that participant's last segment (or at `duration`);
    - one encode: native resolution (largest frame size of the participant's segments, other
      sizes are scaled to fit and padded, aspect ratio kept), constant `fps`, x264 at `crf`
      (0 = lossless), audio AAC in the mp4 and, optionally, a lossless WAV next to it;
    - audio-video sync within each recording: some DuckSoup files hold audio recorded before their
      video starts (the audio queued while the participant waited), with both tracks written from
      timestamp 0, so their audio is late. The offset of each file is estimated from its durations
      (audio vs the wall-clock time between `pipeline_started` and `pipeline_stopped`, bounded by how
      long the audio track had been arriving before the start) and from its
      content (audio envelope vs lower-face motion), and the audio is moved by it (avsync.py);
      each rendered mp4 is then measured again (av_check);
    - a manifest (alignment.csv) with the segments, gaps, covered time, audio coverage, audio
      offsets (with their evidence) and the sync check of every output, so that analyses can mask
      gaps instead of reading the filler as data.

Output file names are the file name of the participant's first used segment, so that code
parsing DuckSoup names keeps working.

Needs ffmpeg/ffprobe (with libx264) on the PATH and numpy; OpenCV is optional (face region for
the sync measurement, whole-frame motion without it).
"""

import csv
import datetime
import glob
import json
import os
import subprocess

try:
    from . import avsync
except ImportError:
    import avsync

# DuckSoup file name: i-<id>-a-<date>-<time>-s-<namespace>-n-<round>-<dyad>-u-<user>-c-<connection>-<dry|wet>.<ext>
# Fields are read from the end, as in ducksoup.py, because the namespace may contain '-'.
DATE_POS, HOUR_POS = 3, 4

MANIFEST_NAME = "alignment.csv"

# per-file audio offset fields (audio_offset) carried into the plan segments and the manifest
OFFSET_FIELDS = ["audio_offset", "offset_source", "offset_note", "audio_path", "video_offset", "twin",
                 "wall_s", "wait_s", "duration_offset", "duration_basis",
                 "content_lag", "content_z", "content_reliable", "content_strong", "content_roi", "content_error"]


# ---------------------------------------------------------------------------------------------
# File names and logs
# ---------------------------------------------------------------------------------------------
def parse_name(path):
    """Fields of a DuckSoup recording name."""
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("-")
    start = datetime.datetime.strptime(parts[DATE_POS] + " " + parts[HOUR_POS], "%Y%m%d %H%M%S.%f")
    manip = parts[-1] if parts[-1] in ("dry", "wet") else "other"
    conn = 1
    if "c" in parts:
        k = len(parts) - 1 - parts[::-1].index("c")
        if k + 1 < len(parts) and parts[k + 1].isdigit():
            conn = int(parts[k + 1])
    user = parts[parts.index("u") + 1] if "u" in parts else base
    return dict(name_time=start, user=user, conn=conn, manip=manip)


LOG_EVENTS = ("pipeline_started", "pipeline_stopped", "interaction_started", "in_track_received", "template_data")


def _fx(options):
    """The `Fx:` field of a DuckSoup template_data options string ('' = no effect)."""
    if "Fx:" not in options:
        return None
    rest = options.split("Fx:", 1)[1]
    end = min([i for i in (rest.find(" Muxer:"), rest.find(" Extension:")) if i >= 0] or [len(rest)])
    return rest[:end].strip()


def read_logs(log_folder):
    """From the DuckSoup logs in log_folder: dict with
    'starts'      {user: sorted datetimes of `pipeline_started`},
    'stops'       {user: sorted datetimes of `pipeline_stopped`},
    'audio_in'    {user: sorted datetimes of the audio `in_track_received` (track arriving at the server)},
    'audio_fx'    {user: sorted (datetime, audio effect) of the pipeline templates ('' = no audio effect)},
    'interaction' sorted datetimes of `interaction_started`."""
    out = dict(starts={}, stops={}, audio_in={}, audio_fx={}, interaction=[])
    key = dict(pipeline_started="starts", pipeline_stopped="stops")
    for log in sorted(glob.glob(os.path.join(log_folder, "*.log"))):
        with open(log, errors="replace") as fh:
            for line in fh:
                if not any(f'"{e}"' in line for e in LOG_EVENTS):
                    continue
                try:
                    j = json.loads(line)
                    t = datetime.datetime.strptime(j["time"], "%Y%m%d-%H%M%S.%f")
                except (ValueError, KeyError):
                    continue
                msg, user = j.get("message"), j.get("user")
                if msg in key and user:
                    out[key[msg]].setdefault(user, []).append(t)
                elif msg == "in_track_received" and user and j.get("kind") == "audio":
                    out["audio_in"].setdefault(user, []).append(t)
                elif msg == "template_data" and user and "audioOptions" in j:
                    fx = _fx(j["audioOptions"])
                    if fx is not None:
                        out["audio_fx"].setdefault(user, []).append((t, fx))
                elif msg == "interaction_started":
                    out["interaction"].append(t)
    for k in ("starts", "stops", "audio_in", "audio_fx"):
        out[k] = {u: sorted(v) for u, v in out[k].items()}
    out["interaction"].sort()
    return out


def recording_stop(user, start, pipeline_starts, pipeline_stops):
    """`pipeline_stopped` of the recording that started at `start` (a `pipeline_started` time of `user`):
    the first stop after it, provided the user's pipeline did not start again in between; else None."""
    nxt = [t for t in pipeline_starts.get(user, []) if t > start]
    for t in pipeline_stops.get(user, []):
        if t > start:
            return t if not nxt or t <= nxt[0] else None
    return None


def recording_audio_fx(user, start, audio_fx):
    """Audio effect of the pipeline that recorded from `start` ('' = none), from the last template logged before
    it; None if not in the log."""
    before = [fx for t, fx in audio_fx.get(user, []) if t <= start]
    return before[-1] if before else None


def recording_wait(user, start, pipeline_starts, audio_in):
    """How long the user's audio track had been arriving when the recording started at `start` (s):
    start - the last audio `in_track_received` before it and after the user's previous pipeline start
    (i.e. of this connection); None if not in the log. Audio queued before the start is at most this long."""
    prev = [t for t in pipeline_starts.get(user, []) if t < start]
    lo = prev[-1] if prev else None
    arr = [t for t in audio_in.get(user, []) if t <= start and (lo is None or t > lo)]
    return (start - arr[-1]).total_seconds() if arr else None


def recording_starts(name_times, pipeline_starts, before=2.0, after=60.0):
    """Start of the recording of every file, from its user and the time in its name.

    name_times: {user: iterable of name times} (one per connection; dry and wet files of the same
    connection share the name time and therefore the start).
    Each `pipeline_started` event of a user is given to one connection only, in order: the first
    unused event within [name_time - before, name_time + after] s. Without one, the name time is used.
    Returns {(user, name_time): (start, "log" | "file_name")}."""
    out = {}
    for user, times in name_times.items():
        events, k = pipeline_starts.get(user, []), 0
        for T in sorted(set(times)):
            while k < len(events) and (events[k] - T).total_seconds() < -before:
                k += 1
            if k < len(events) and (events[k] - T).total_seconds() <= after:
                out[(user, T)] = (events[k], "log")
                k += 1
            else:
                out[(user, T)] = (T, "file_name")
    return out


# ---------------------------------------------------------------------------------------------
# ffprobe
# ---------------------------------------------------------------------------------------------
def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def probe(path):
    """Durations and stream properties of a recording (no decoding, except key frames for the size)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration:stream=codec_type,duration,width,height,channels,sample_rate",
                          "-of", "json", path], capture_output=True, text=True)
    j = json.loads(out.stdout or "{}")
    fmt = _float(j.get("format", {}).get("duration"))
    info = dict(duration=fmt, has_video=False, has_audio=False, video_duration=None, audio_duration=None,
                width=None, height=None, channels=None, sample_rate=None)
    for s in j.get("streams", []):
        if s.get("codec_type") == "video" and not info["has_video"]:
            info.update(has_video=True, video_duration=_float(s.get("duration")) or fmt)
        elif s.get("codec_type") == "audio" and not info["has_audio"]:
            info.update(has_audio=True, audio_duration=_float(s.get("duration")) or fmt,
                        channels=s.get("channels"), sample_rate=_float(s.get("sample_rate")))
    if info["has_video"]:
        # the resolution can change during a WebRTC recording; changes start on a key frame
        sizes = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
                                "-show_entries", "frame=width,height", "-of", "csv=p=0", path],
                               capture_output=True, text=True).stdout.split()
        wh = [tuple(int(v) for v in s.split(",")[:2]) for s in sizes if s.count(",") >= 1]
        if wh:
            info["width"], info["height"] = max(wh, key=lambda x: x[0] * x[1])
    return info


# ---------------------------------------------------------------------------------------------
# Plan: which part of which file goes where on the timeline
# ---------------------------------------------------------------------------------------------
def audio_offset(path, info, wall_s, wait_s=None, av_sync=True, max_offset=avsync.MAX_LAG_S, mouth_backend="auto",
                 face_model=None):
    """Audio offset of one raw file (s, > 0: its audio is late relative to its video), see avsync.py.
    wall_s: pipeline_stopped - pipeline_started; wait_s: pipeline_started - audio in_track_received.
    Returns a dict that goes into the plan and the manifest."""
    dur_off, basis, dur_note = avsync.duration_offset(info["audio_duration"] if info["has_audio"] else None,
                                                      info["video_duration"] if info["has_video"] else None,
                                                      wall_s, wait_s)
    content = None
    if av_sync and info["has_video"] and info["has_audio"]:
        content = avsync.measure(path, max_lag_s=max_offset, mouth_backend=mouth_backend, face_model=face_model)
    off, src, note = avsync.decide(dur_off, content, max_offset=max_offset)
    c = content or {}
    return dict(audio_offset=off, offset_source=src, offset_note="; ".join(x for x in (dur_note, note) if x),
                wall_s=wall_s, wait_s=wait_s, duration_offset=dur_off, duration_basis=basis,
                content_lag=None if c.get("error") or not c else c["lag"], content_z=c.get("z"),
                content_reliable=c.get("reliable"), content_strong=c.get("strong"), content_roi=c.get("roi"),
                content_error=c.get("error"))


def twin_offsets(wet, dry, own, max_offset=avsync.MAX_LAG_S):
    """Offsets of a wet file, from its dry twin (the same camera and microphone recorded through another muxer).

    - video: the wet video is moved by its lag relative to the dry video (image content, frame-exact), so that
      it is on the dry file's clock, which is verified (log start; the audio is synced on it);
    - audio: when the wet audio is the dry microphone signal (no audio effect in the pipeline template of the log;
      without a log, waveforms clearly identical), the wet output takes the dry twin's audio, with the dry
      offset: identical sound, and immune to the packets the wet muxer dropped or re-timed.
      Otherwise the wet audio is measured on its own (own) and moved together with the wet video.
    Falls back to the file's own measurement when the twin comparison is inconclusive."""
    if not (wet["has_video"] and dry["has_video"]):
        return dict(own(wet), audio_path=wet["path"], video_offset=0.0, twin=None)
    tw = avsync.twin_lags(dry["path"], wet["path"])
    v, a = tw["video"], tw["audio"]
    notes = []
    if v.get("reliable") and abs(v["lag"]) <= max_offset:
        voff = v["lag"]
        if v.get("drift"):
            notes.append(f"wet video drifts vs dry ({min(v['thirds']):.2f} to {max(v['thirds']):.2f} s), "
                         "constant shift applied")
    else:
        voff = 0.0
        notes.append("wet video vs dry: " + (v.get("error") or "inconclusive") + ", not moved")
    # is the wet audio the dry microphone signal? The log says whether the pipeline had an audio effect;
    # without it, the signals must be clearly identical (avsync.twin_lags)
    if wet.get("audio_fx") is not None:
        same, basis = wet["audio_fx"] == "", "log"
    else:
        same, basis = bool(a.get("same_audio")), "signal"
    twin = dict(video_lag=v.get("lag"), video_z=v.get("z"), video_thirds=v.get("thirds"), video_drift=v.get("drift"),
                audio_lag=a.get("lag"), audio_z=a.get("z"), waveform_r=a.get("waveform_r"), audio_fx=wet.get("audio_fx"),
                same_audio=same, same_audio_basis=basis)
    if same and dry["has_audio"]:
        out = {k: dry[k] for k in OFFSET_FIELDS if k not in ("offset_note", "audio_path", "video_offset", "twin")}
        out.update(offset_source="dry twin: " + dry["offset_source"], audio_path=dry["path"],
                   offset_note="; ".join(notes + ["audio from the dry twin"] + ([dry["offset_note"]] if dry["offset_note"] else [])))
    else:
        out = own(wet)
        # the wet audio was measured against the wet video, which is moved by voff: the audio moves with it
        out.update(audio_offset=out["audio_offset"] + voff, audio_path=wet["path"],
                   offset_note="; ".join(notes + ([out["offset_note"]] if out["offset_note"] else [])))
    out.update(video_offset=voff, twin=twin)
    return out


def plan_interaction(source_folder, extension=".mp4", manips=("dry", "wet"), log_folder=None, duration=None,
                     av_sync=True, max_offset=avsync.MAX_LAG_S, mouth_backend="auto", face_model=None, twin_sync=True):
    """Plan the outputs of one interaction folder.

    av_sync    : measure each file's audio-video offset from its content (see avsync.py) to verify / correct
                 the offset seen in the durations; False: durations only (flagged as unverified)
    max_offset : offsets larger than this (s) are reported, not applied
    mouth_backend, face_model : mouth signal of the content measurement (see avsync.py)
    twin_sync  : align each wet file on its dry twin (see twin_offsets) instead of measuring it on its own

    Returns (plans, notes): plans is a list of dicts, one per (user, manip), with the segments
    to use, their position on the common timeline, their audio offset and the gaps; notes lists skipped files."""
    if log_folder is None:
        log_folder = os.path.dirname(os.path.normpath(source_folder))
    logs = read_logs(log_folder)
    pipeline_starts, interaction_starts = logs["starts"], logs["interaction"]
    metas, notes = {}, []
    for f in sorted(glob.glob(os.path.join(source_folder, "*" + extension))):
        try:
            metas[f] = parse_name(f)
        except (ValueError, IndexError):
            notes.append((f, "unparsable_name"))
    # all files, empty ones included, take part in the matching so that events are used in order
    name_times = {}
    for m in metas.values():
        name_times.setdefault(m["user"], []).append(m["name_time"])
    starts = recording_starts(name_times, pipeline_starts)
    files = []
    for f, meta in metas.items():
        if manips and meta["manip"] not in manips and meta["manip"] != "other":
            continue
        if os.path.getsize(f) == 0:
            notes.append((f, "empty_file"))
            continue
        info = probe(f)
        length = info["video_duration"] if info["has_video"] else info["audio_duration"]
        if not length:
            notes.append((f, "unreadable"))
            continue
        start, source = starts[(meta["user"], meta["name_time"])]
        wall = wait = audio_fx = None
        if source == "log":
            audio_fx = recording_audio_fx(meta["user"], start, logs["audio_fx"])
            stop = recording_stop(meta["user"], start, pipeline_starts, logs["stops"])
            wall = (stop - start).total_seconds() if stop else None
            wait = recording_wait(meta["user"], start, pipeline_starts, logs["audio_in"])
        files.append(dict(path=f, start=start, start_source=source, length=length, wall=wall, wait=wait,
                          audio_fx=audio_fx, **meta, **info))
    if not files:
        return [], notes
    # audio-video offsets: dry (and other) files first, then the wet files, aligned on their dry twin
    own = lambda x: audio_offset(x["path"], x, x["wall"], x["wait"], av_sync, max_offset, mouth_backend, face_model)
    for x in files:
        if x["manip"] != "wet":
            x.update(own(x), audio_path=x["path"], video_offset=0.0, twin=None)
    for x in files:
        if x["manip"] == "wet":
            twin = next((y for y in files if y["manip"] == "dry" and y["user"] == x["user"]
                         and y["name_time"] == x["name_time"]), None)
            x.update(twin_offsets(x, twin, own, max_offset) if twin_sync and twin is not None
                     else dict(own(x), audio_path=x["path"], video_offset=0.0, twin=None))
    by_path = {x["path"]: x for x in files}
    for x in files:
        src = by_path[x["audio_path"]]
        x["has_audio"], x["audio_source_s"] = src["has_audio"], src["audio_duration"]

    # t0 = start of the interaction in the log (every pipeline starts then); without a log: the moment
    # everybody is recording (latest first-segment start). A participant whose first usable recording
    # starts after t0 (e.g. a first file that is empty) gets a leading gap.
    first = {}
    for x in files:
        first[x["user"]] = min(first.get(x["user"], x["start"]), x["start"])
    t0 = max(first.values())
    t0_source = "first_recordings"
    earliest = min(first.values())
    candidates = [t for t in interaction_starts if t <= t0 + datetime.timedelta(seconds=2)]
    if candidates and (earliest - candidates[-1]).total_seconds() < 2:
        t0, t0_source = max(candidates[-1], earliest), "log"
    end_cap = t0 + datetime.timedelta(seconds=duration) if duration else None

    plans = []
    for key in sorted({(x["user"], x["manip"]) for x in files}):
        segs = sorted((x for x in files if (x["user"], x["manip"]) == key), key=lambda x: x["start"])
        used, cursor = [], 0.0  # cursor = end of the previous segment on the timeline (s from t0)
        for x in segs:
            begin = (x["start"] - t0).total_seconds()
            end = begin + x["length"]
            if end_cap is not None:
                end = min(end, duration)
            cut = max(0.0, cursor - begin, -begin)  # before t0, or overlapping the previous segment
            if end - (begin + cut) <= 0.04:
                notes.append((x["path"], "before_t0" if end <= 0 else "overlap_or_after_end"))
                continue
            used.append(dict(path=x["path"], t_begin=begin + cut, t_end=end, cut=cut, gap_before=begin + cut - cursor,
                             has_audio=x["has_audio"],
                             # the audio is moved earlier by its offset, so its end moves too
                             audio_end=begin + (x["audio_source_s"] or 0) - x["audio_offset"],
                             start_source=x["start_source"], conn=x["conn"], width=x["width"], height=x["height"],
                             channels=x["channels"], video_s=x["video_duration"], audio_s=x["audio_source_s"],
                             **{k: x[k] for k in OFFSET_FIELDS}))
            cursor = end
        if not used:
            continue
        used[0]["gap_before"] = used[0]["t_begin"]  # > 0 only if this participant's first segment starts after t0
        has_video = any(x["has_video"] for x in segs)
        w = max((u["width"] or 0) for u in used) if has_video else None
        h = max((u["height"] or 0) for u in used) if has_video else None
        audio_covered = sum(max(0.0, min(u["t_end"], u["audio_end"]) - u["t_begin"]) for u in used if u["has_audio"])
        plans.append(dict(user=key[0], manip=key[1], t0=t0, t0_source=t0_source, segments=used, has_video=has_video,
                          width=w + (w % 2) if w else None, height=h + (h % 2) if h else None,
                          channels=next((u["channels"] for u in used if u["channels"]), 2),
                          out_duration=used[-1]["t_end"],
                          covered=sum(u["t_end"] - u["t_begin"] for u in used),
                          audio_covered=audio_covered,
                          gaps=[round(u["gap_before"], 3) for u in used if u["gap_before"] > 0.04],
                          out_name=os.path.splitext(os.path.basename(used[0]["path"]))[0]))
    return plans, notes


# ---------------------------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------------------------
def build_command(plan, out_video, out_wav=None, fps=30, crf=0, preset="veryfast", audio_codec="aac",
                  audio_bitrate="320k", sample_rate=48000, wav_codec="pcm_s16le", threads=2):
    """ffmpeg command (list) that renders one plan.

    Segment and gap boundaries are snapped to the output frame grid (multiples of 1/fps from t0), so
    that frame rounding cannot accumulate over segments; the content of each segment is shifted by its
    snapping remainder (cut a bit more, or padded by less than one frame), so it stays at its true time
    (audio to the sample, video to the frame). The audio of a segment is also moved by its
    `audio_offset`, and its video by its `video_offset` (see avsync.py and twin_offsets); its audio can come
    from another file (`audio_path`, the dry twin of a wet file)."""
    W, H = plan["width"], plan["height"]
    fps_f = float(fps)
    layout = "mono" if plan["channels"] == 1 else "stereo"
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-y"]
    filters, labels, k_in, prev_end = [], [], 0, 0
    for i, s in enumerate(plan["segments"]):
        b, e = round(s["t_begin"] * fps_f), round(s["t_end"] * fps_f)
        if e <= b:
            continue
        gap_frames = b - prev_end
        if gap_frames > 0:
            gap = gap_frames / fps_f
            if plan["has_video"]:
                filters.append(f"color=c=black:s={W}x{H}:r={fps}:d={gap:.6f},format=yuv420p,setsar=1,"
                               f"trim=end_frame={gap_frames},setpts=PTS-STARTPTS[gv{i}]")
            filters.append(f"anullsrc=r={sample_rate}:cl={layout},atrim=duration={gap:.6f},asetpts=PTS-STARTPTS[ga{i}]")
            labels.append((f"[gv{i}]" if plan["has_video"] else "") + f"[ga{i}]")
        prev_end = e
        d, n_frames = (e - b) / fps_f, e - b
        r = b / fps_f - s["t_begin"]          # > 0: the grid point is after the true start -> cut more
        cut, pad = s["cut"] + max(r, 0.0), max(-r, 0.0)
        seek = ["-ss", f"{cut:.6f}"] if cut > 0 else []
        cmd += seek + ["-reinit_filter", "0", "-i", s["path"]]
        k_v = k_a = k_in
        if s.get("audio_path", s["path"]) != s["path"]:
            # the audio comes from another file (the dry twin of a wet file), cut at the same point
            k_a = k_in + 1
            cmd += seek + ["-i", s["audio_path"]]
        # video offset > 0 (content late in its file): moved earlier (its first frames dropped); < 0 (content early,
        # e.g. a wet video that started after its dry twin): delayed, with BLACK frames before it (as in gaps, so
        # that no frozen face is read as data)
        voff = s.get("video_offset", 0.0)
        lead = round(max(-voff, 0.0) * fps_f)
        if plan["has_video"]:
            filters.append(f"[{k_v}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
                           f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,setpts=PTS+{pad - max(voff, 0.0):.6f}/TB,"
                           f"fps=fps={fps}:start_time=0,format=yuv420p,"
                           + (f"tpad=start={lead}:start_mode=add:color=black," if lead else "") +
                           f"tpad=stop_mode=add:stop_duration={d:.6f},trim=end_frame={n_frames},setpts=PTS-STARTPTS[v{i}]")
        if s["has_audio"]:
            # audio offset > 0 (audio late in the file): its first `offset` s are dropped (negative timestamps,
            # trimmed by aresample); < 0: the audio is delayed (padded with silence by aresample)
            shift = pad - s.get("audio_offset", 0.0)
            filters.append(f"[{k_a}:a]asetpts=PTS+{shift:.6f}/TB,aresample=async=1:first_pts=0,"
                           f"aformat=sample_rates={sample_rate}:channel_layouts={layout},"
                           f"apad=whole_dur={d:.6f},atrim=duration={d:.6f},asetpts=PTS-STARTPTS[a{i}]")
        else:
            filters.append(f"anullsrc=r={sample_rate}:cl={layout},atrim=duration={d:.6f},asetpts=PTS-STARTPTS[a{i}]")
        labels.append((f"[v{i}]" if plan["has_video"] else "") + f"[a{i}]")
        k_in = max(k_v, k_a) + 1
    n = len(labels)
    v = 1 if plan["has_video"] else 0
    filters.append("".join(labels) + f"concat=n={n}:v={v}:a=1" + ("[v]" if v else "") + "[acat]")
    filters.append("[acat]asplit=2[a][aw]" if out_wav else "[acat]anull[a]")
    cmd += ["-filter_complex", ";".join(filters), "-threads", str(threads)]
    if out_video:
        if v:
            cmd += ["-map", "[v]", "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
        cmd += ["-map", "[a]", "-c:a", audio_codec, "-b:a", audio_bitrate, "-movflags", "+faststart", out_video]
    else:
        cmd += ["-map", "[a]", "-f", "null", "-"]
    if out_wav:
        cmd += ["-map", "[aw]", "-c:a", wav_codec, out_wav]
    return cmd


def render(plan, out_video, out_wav=None, overwrite=False, **kw):
    """Render one plan; writes to temporary files first so that an interrupted run leaves no output."""
    targets = [p for p in (out_video, out_wav) if p]
    if not overwrite and all(os.path.exists(p) for p in targets):
        return "exists"
    tmp = {p: os.path.join(os.path.dirname(p), ".tmp_" + os.path.basename(p)) for p in targets}
    cmd = build_command(plan, tmp.get(out_video), tmp.get(out_wav), **kw)
    p = subprocess.run(cmd, capture_output=True, text=True)
    ok = p.returncode == 0 and all(os.path.exists(t) and os.path.getsize(t) > 0 for t in tmp.values())
    if not ok:
        for t in tmp.values():
            if os.path.exists(t):
                os.remove(t)
        err = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ""
        return "ffmpeg_error: " + err[:300]
    for final, t in tmp.items():
        os.replace(t, final)
    return "ok"


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


def check_output(path, max_offset=avsync.MAX_LAG_S, mouth_backend="auto", face_model=None):
    """Audio-video lag measured on a rendered output: (lag_s, z, status). status: 'ok' (reliable and in sync
    within avsync.AGREE_TOL_S), 'mismatch' (reliable, out of sync), 'inconclusive', or an error."""
    m = avsync.measure(path, max_lag_s=max_offset, mouth_backend=mouth_backend, face_model=face_model)
    if m.get("error"):
        return None, None, m["error"]
    if not m["reliable"]:
        return _r(m["lag"]), _r(m["z"], 2), "inconclusive"
    return _r(m["lag"]), _r(m["z"], 2), ("ok" if abs(m["lag"]) <= avsync.AGREE_TOL_S else "mismatch")


def align_folder(source_folder, folder_tag, target_folder, trimed_path="trimed/", extension=".mp4",
                 manips=("dry", "wet"), log_folder=None, duration=None, write_wav=True, video_out=True,
                 overwrite=False, verbose=True, av_sync=True, av_check=True, max_offset=avsync.MAX_LAG_S,
                 mouth_backend="auto", face_model=None, twin_sync=True, **encode):
    """Align and encode one interaction folder (see module docstring).

    source_folder : folder with the original recordings of one interaction (ends with '/')
    folder_tag    : name of the interaction, used as output sub-folder (ends with '/')
    target_folder : root of the preprocessed data (ends with '/')
    log_folder    : folder with the DuckSoup *.log files (default: parent of source_folder)
    duration      : optional maximum length of the outputs, in s from t0
    write_wav     : also write a lossless WAV of the audio of each output
    video_out     : write the mp4 (False: WAV only, e.g. audio-only experiments)
    av_sync       : measure the audio-video offset of each raw file from its content to verify / correct the
                    offset seen in its durations (avsync.py); False: durations only
    av_check      : measure the audio-video lag of each rendered mp4 and report it (av_check_* columns)
    max_offset    : audio offsets larger than this (s) are reported, not applied
    mouth_backend : mouth signal of the sync measurement: "auto" (best available), "mediapipe", "opencv", "frame"
    face_model    : MediaPipe face landmarker model (.task) for mouth_backend "mediapipe" / "auto"
    twin_sync     : align each wet file on its dry twin (video by image content; audio taken from the dry twin
                    when it is the same microphone signal), see twin_offsets
    encode        : fps, crf, preset, audio_codec, audio_bitrate, sample_rate, wav_codec,
                    threads (ffmpeg threads for the encode and the sync measurement; default 2)
    Returns the manifest rows (also written to <out folder>/alignment.csv)."""
    out_dir = os.path.join(target_folder, trimed_path, folder_tag)
    avsync.THREADS = max(1, int(encode.get("threads", 2)))      # the sync measurement decodes with the same threads
    plans, notes = plan_interaction(source_folder, extension, manips, log_folder, duration, av_sync, max_offset,
                                    mouth_backend, face_model, twin_sync)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for plan in plans:
        out_video = os.path.join(out_dir, plan["out_name"] + ".mp4") if (video_out and plan["has_video"]) else None
        out_wav = os.path.join(out_dir, plan["out_name"] + ".wav") if (write_wav or not out_video) else None
        segs = plan["segments"]
        offsets = [s["audio_offset"] for s in segs]
        if verbose:
            print(f"aligning {folder_tag}{plan['out_name']}: {len(segs)} segment(s), gaps {plan['gaps']} s, "
                  f"audio offsets {[round(o, 3) for o in offsets]} s, "
                  f"video offsets {[round(s['video_offset'], 3) for s in segs]} s", flush=True)
        status = render(plan, out_video, out_wav, overwrite=overwrite, **encode)
        check = (None, None, "not_run")
        if av_check and out_video and status in ("ok", "exists") and any(s["has_audio"] for s in segs):
            check = check_output(out_video, max_offset, mouth_backend, face_model)
        notes_out = [f"c{s['conn']}: {s['offset_note']}" for s in segs if s["offset_note"]]
        if check[2] == "mismatch":
            notes_out.append(f"output audio-video lag {check[0]} s")
        rows.append(dict(folder_tag=folder_tag.strip("/"), user=plan["user"], manip=plan["manip"],
                         output=os.path.basename(out_video or out_wav), status=status,
                         t0=plan["t0"].strftime("%Y%m%d-%H%M%S.%f")[:-3], t0_source=plan["t0_source"],
                         start_source=",".join(sorted({s["start_source"] for s in segs})),
                         n_segments=len(segs), duration_s=round(plan["out_duration"], 3),
                         covered_s=round(plan["covered"], 3), audio_covered_s=round(plan["audio_covered"], 3),
                         n_gaps=len(plan["gaps"]), gaps_s=round(sum(plan["gaps"]), 3),
                         width=plan["width"], height=plan["height"],
                         audio_offset_s=round(max(offsets, key=abs), 3),
                         video_offset_s=round(max((s["video_offset"] for s in segs), key=abs), 3),
                         offset_source=",".join(dict.fromkeys(s["offset_source"] for s in segs)),
                         av_check_lag_s=check[0], av_check_z=check[1], av_check=check[2],
                         notes="; ".join(notes_out),
                         segments=json.dumps([dict(file=os.path.basename(s["path"]), connection=s["conn"],
                                                   t_begin=round(s["t_begin"], 3), t_end=round(s["t_end"], 3),
                                                   cut_s=round(s["cut"], 3), wall_s=_r(s["wall_s"]),
                                                   wait_s=_r(s["wait_s"]),
                                                   video_s=_r(s["video_s"]), audio_s=_r(s["audio_s"]),
                                                   duration_offset_s=_r(s["duration_offset"]),
                                                   duration_basis=s["duration_basis"],
                                                   content_lag_s=_r(s["content_lag"]), content_z=_r(s["content_z"], 2),
                                                   content_reliable=s["content_reliable"],
                                                   content_strong=s["content_strong"], content_roi=s["content_roi"],
                                                   content_error=s["content_error"],
                                                   audio_offset_s=round(s["audio_offset"], 3),
                                                   offset_source=s["offset_source"], offset_note=s["offset_note"],
                                                   audio_source=("own" if s["audio_path"] == s["path"]
                                                                 else os.path.basename(s["audio_path"])),
                                                   video_offset_s=round(s["video_offset"], 3),
                                                   twin=None if s["twin"] is None else
                                                   {k: (round(float(v), 3) if isinstance(v, float) else v)
                                                    for k, v in s["twin"].items()})
                                              for s in segs])))
        if verbose and (status not in ("ok", "exists") or notes_out):
            print(f"  -> {status}; " + "; ".join(notes_out), flush=True)
    for f, why in notes:
        rows.append(dict(folder_tag=folder_tag.strip("/"), output=os.path.basename(f), status="skipped_" + why))
        if verbose:
            print(f"skipped ({why}): {f}", flush=True)
    if rows:
        fields = ["folder_tag", "user", "manip", "output", "status", "t0", "t0_source", "start_source", "n_segments",
                  "duration_s", "covered_s", "audio_covered_s", "n_gaps", "gaps_s", "width", "height",
                  "audio_offset_s", "video_offset_s", "offset_source", "av_check_lag_s", "av_check_z", "av_check", "notes",
                  "segments"]
        with open(os.path.join(out_dir, MANIFEST_NAME), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return rows
