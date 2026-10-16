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
    - a manifest (alignment.csv) with the segments, gaps, covered time and audio coverage of
      every output, so that analyses can mask gaps instead of reading the filler as data.

Output file names are the file name of the participant's first used segment, so that code
parsing DuckSoup names keeps working.

Only needs ffmpeg/ffprobe (with libx264) on the PATH and the standard library.
"""

import csv
import datetime
import glob
import json
import os
import subprocess

# DuckSoup file name: i-<id>-a-<date>-<time>-s-<namespace>-n-<round>-<dyad>-u-<user>-c-<connection>-<dry|wet>.<ext>
# Fields are read from the end, as in ducksoup.py, because the namespace may contain '-'.
DATE_POS, HOUR_POS = 3, 4

MANIFEST_NAME = "alignment.csv"


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


def read_logs(log_folder):
    """From the DuckSoup logs in log_folder: ({user: sorted datetimes of `pipeline_started`},
    sorted datetimes of `interaction_started`)."""
    starts, interaction = {}, []
    for log in sorted(glob.glob(os.path.join(log_folder, "*.log"))):
        with open(log, errors="replace") as fh:
            for line in fh:
                if '"pipeline_started"' not in line and '"interaction_started"' not in line:
                    continue
                try:
                    j = json.loads(line)
                    t = datetime.datetime.strptime(j["time"], "%Y%m%d-%H%M%S.%f")
                except (ValueError, KeyError):
                    continue
                if j.get("message") == "pipeline_started" and j.get("user"):
                    starts.setdefault(j["user"], []).append(t)
                elif j.get("message") == "interaction_started":
                    interaction.append(t)
    return {u: sorted(v) for u, v in starts.items()}, sorted(interaction)


def recording_start(name_time, user, pipeline_starts, before=2.0, after=60.0):
    """Start of the recording of a file: the user's first `pipeline_started` within
    [name_time - before, name_time + after] seconds, else the time in the file name."""
    for t in pipeline_starts.get(user, []):
        dt = (t - name_time).total_seconds()
        if -before <= dt <= after:
            return t, "log"
    return name_time, "file_name"


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
def plan_interaction(source_folder, extension=".mp4", manips=("dry", "wet"), log_folder=None, duration=None):
    """Plan the outputs of one interaction folder.

    Returns (plans, notes): plans is a list of dicts, one per (user, manip), with the segments
    to use, their position on the common timeline and the gaps; notes lists skipped files."""
    if log_folder is None:
        log_folder = os.path.dirname(os.path.normpath(source_folder))
    pipeline_starts, interaction_starts = read_logs(log_folder)
    files, notes = [], []
    for f in sorted(glob.glob(os.path.join(source_folder, "*" + extension))):
        try:
            meta = parse_name(f)
        except (ValueError, IndexError):
            notes.append((f, "unparsable_name"))
            continue
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
        start, source = recording_start(meta["name_time"], meta["user"], pipeline_starts)
        files.append(dict(path=f, start=start, start_source=source, length=length, **meta, **info))
    if not files:
        return [], notes

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
                             has_audio=x["has_audio"], audio_end=begin + (x["audio_duration"] or 0),
                             start_source=x["start_source"], conn=x["conn"], width=x["width"], height=x["height"],
                             channels=x["channels"]))
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
                  audio_bitrate="320k", sample_rate=48000, wav_codec="pcm_s16le", threads=0):
    """ffmpeg command (list) that renders one plan."""
    W, H = plan["width"], plan["height"]
    layout = "mono" if plan["channels"] == 1 else "stereo"
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-y"]
    filters, labels, k_in = [], [], 0
    for i, s in enumerate(plan["segments"]):
        gap = s["gap_before"]
        if gap > 0.04:
            if plan["has_video"]:
                filters.append(f"color=c=black:s={W}x{H}:r={fps}:d={gap:.3f},format=yuv420p,setsar=1[gv{i}]")
            filters.append(f"anullsrc=r={sample_rate}:cl={layout},atrim=duration={gap:.3f},asetpts=PTS-STARTPTS[ga{i}]")
            labels.append((f"[gv{i}]" if plan["has_video"] else "") + f"[ga{i}]")
        d = s["t_end"] - s["t_begin"]
        if s["cut"] > 0:
            cmd += ["-ss", f"{s['cut']:.3f}"]
        cmd += ["-reinit_filter", "0", "-i", s["path"]]
        if plan["has_video"]:
            filters.append(f"[{k_in}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
                           f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=fps={fps}:start_time=0,format=yuv420p,"
                           f"tpad=stop_mode=add:stop_duration={d:.3f},trim=duration={d:.3f},setpts=PTS-STARTPTS[v{i}]")
        if s["has_audio"]:
            filters.append(f"[{k_in}:a]aresample=async=1:first_pts=0,"
                           f"aformat=sample_rates={sample_rate}:channel_layouts={layout},"
                           f"apad=whole_dur={d:.3f},atrim=duration={d:.3f},asetpts=PTS-STARTPTS[a{i}]")
        else:
            filters.append(f"anullsrc=r={sample_rate}:cl={layout},atrim=duration={d:.3f},asetpts=PTS-STARTPTS[a{i}]")
        labels.append((f"[v{i}]" if plan["has_video"] else "") + f"[a{i}]")
        k_in += 1
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


def align_folder(source_folder, folder_tag, target_folder, trimed_path="trimed/", extension=".mp4",
                 manips=("dry", "wet"), log_folder=None, duration=None, write_wav=True, video_out=True,
                 overwrite=False, verbose=True, **encode):
    """Align and encode one interaction folder (see module docstring).

    source_folder : folder with the original recordings of one interaction (ends with '/')
    folder_tag    : name of the interaction, used as output sub-folder (ends with '/')
    target_folder : root of the preprocessed data (ends with '/')
    log_folder    : folder with the DuckSoup *.log files (default: parent of source_folder)
    duration      : optional maximum length of the outputs, in s from t0
    write_wav     : also write a lossless WAV of the audio of each output
    video_out     : write the mp4 (False: WAV only, e.g. audio-only experiments)
    encode        : fps, crf, preset, audio_codec, audio_bitrate, sample_rate, wav_codec, threads
    Returns the manifest rows (also written to <out folder>/alignment.csv)."""
    out_dir = os.path.join(target_folder, trimed_path, folder_tag)
    plans, notes = plan_interaction(source_folder, extension, manips, log_folder, duration)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for plan in plans:
        out_video = os.path.join(out_dir, plan["out_name"] + ".mp4") if (video_out and plan["has_video"]) else None
        out_wav = os.path.join(out_dir, plan["out_name"] + ".wav") if (write_wav or not out_video) else None
        if verbose:
            print(f"aligning {folder_tag}{plan['out_name']}: {len(plan['segments'])} segment(s), "
                  f"gaps {plan['gaps']} s", flush=True)
        status = render(plan, out_video, out_wav, overwrite=overwrite, **encode)
        rows.append(dict(folder_tag=folder_tag.strip("/"), user=plan["user"], manip=plan["manip"],
                         output=os.path.basename(out_video or out_wav), status=status,
                         t0=plan["t0"].strftime("%Y%m%d-%H%M%S.%f")[:-3], t0_source=plan["t0_source"],
                         start_source=",".join(sorted({s["start_source"] for s in plan["segments"]})),
                         n_segments=len(plan["segments"]), duration_s=round(plan["out_duration"], 3),
                         covered_s=round(plan["covered"], 3), audio_covered_s=round(plan["audio_covered"], 3),
                         n_gaps=len(plan["gaps"]), gaps_s=round(sum(plan["gaps"]), 3),
                         width=plan["width"], height=plan["height"],
                         segments=json.dumps([dict(file=os.path.basename(s["path"]), connection=s["conn"],
                                                   t_begin=round(s["t_begin"], 3), t_end=round(s["t_end"], 3),
                                                   cut_s=round(s["cut"], 3)) for s in plan["segments"]])))
        if verbose and status != "ok":
            print(f"  -> {status}", flush=True)
    for f, why in notes:
        rows.append(dict(folder_tag=folder_tag.strip("/"), output=os.path.basename(f), status="skipped_" + why))
        if verbose:
            print(f"skipped ({why}): {f}", flush=True)
    if rows:
        fields = ["folder_tag", "user", "manip", "output", "status", "t0", "t0_source", "start_source", "n_segments",
                  "duration_s", "covered_s", "audio_covered_s", "n_gaps", "gaps_s", "width", "height", "segments"]
        with open(os.path.join(out_dir, MANIFEST_NAME), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return rows
