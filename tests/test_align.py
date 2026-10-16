"""
Tests for align.py on synthetic DuckSoup interactions (needs ffmpeg with libx264 and libopus).

Each synthetic recording shows a white flash and plays a beep at known ABSOLUTE times (the
same events for both participants), so after alignment the events must appear at
(event time - t0) in every output, whatever the join times, reconnections and cuts.

Run:  python -m pytest tests/test_align.py -v
"""

import datetime
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import align  # noqa: E402

FPS_IN, FPS_OUT, SR = 25, 30, 48000
T = datetime.datetime(2024, 1, 12, 15, 0, 0)          # interaction start (pipeline_started)
EVENTS = [1.0, 3.5, 7.2, 9.6, 12.3, 15.1, 18.8]      # absolute times (s from T) of flashes and beeps
NS = "mktest_session"


def ts(t):
    return (T + datetime.timedelta(seconds=t)).strftime("%Y%m%d-%H%M%S.%f")[:-3]


def name(join_t, user, conn, manip, dyad="p1p2", rnd=1):
    return f"i-0123abcd-a-{ts(join_t)}-s-{NS}-n-{rnd}-{dyad}-u-{user}-c-{conn}-{manip}.mp4"


def make_clip(path, rec_start, length, size=(320, 240), audio_length=None, size2=None):
    """Clip recorded from rec_start (s from T) for `length` s. Flash + beep at EVENTS.
    size2: second half of the clip at another resolution (mid-stream size change)."""
    def part(out, off, dur, wh):
        en = "+".join(f"between(t+{off},{e},{e + 0.1})" for e in EVENTS)
        ad = dur if audio_length is None else max(0.05, min(dur, audio_length - (off - rec_start)))
        subprocess.run(["ffmpeg", "-v", "error", "-y",
                        "-f", "lavfi", "-i", f"color=c=black:s={wh[0]}x{wh[1]}:r={FPS_IN}:d={dur}",
                        # sample-accurate beeps (the `enable` timeline of audio filters works per audio frame)
                        "-f", "lavfi", "-i", f"aevalsrc=exprs='0.5*sin(2*PI*1000*t)*({en})':s={SR}:d={ad}",
                        "-filter_complex",
                        f"[0:v]drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='{en}'[v];[1:a]anull[a]",
                        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-g", "25", "-pix_fmt", "yuv420p",
                        "-c:a", "libopus", out], check=True)
    if size2 is None:
        part(path, rec_start, length, size)
    else:
        h = length / 2
        a, b = path + ".a.mp4", path + ".b.mp4"
        part(a, rec_start, h, size)
        part(b, rec_start + h, h, size2)
        lst = path + ".txt"
        with open(lst, "w") as fh:
            fh.write(f"file '{a}'\nfile '{b}'\n")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", path],
                       check=True)
        for f in (a, b, lst):
            os.remove(f)


def write_log(folder, pipeline_starts):
    with open(os.path.join(folder, "1-p1p2-a-test.log"), "w") as fh:
        fh.write(json.dumps({"level": "info", "time": ts(0), "message": "interaction_started"}) + "\n")
        for user, t in pipeline_starts:
            fh.write(json.dumps({"level": "info", "user": user, "time": ts(t), "message": "pipeline_started"}) + "\n")


def flashes(video):
    """Onset times (s) of white frames."""
    w, h = 8, 6
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"scale={w}:{h},format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    m = np.frombuffer(raw, np.uint8).reshape(-1, w * h).mean(1) > 128
    on = np.flatnonzero(m[1:] & ~m[:-1]) + 1
    return on / FPS_OUT, len(m) / FPS_OUT


def beeps(wav):
    import soundfile as sf
    x, sr = sf.read(wav)
    x = np.abs(x if x.ndim == 1 else x.mean(1))
    hop = int(sr * 0.005)
    env = x[:len(x) // hop * hop].reshape(-1, hop).max(1) > 0.1
    on = np.flatnonzero(env[1:] & ~env[:-1]) + 1
    return on * hop / sr, len(x) / sr


def expected(t0, intervals):
    """Events (s from t0) that fall in the covered intervals (s from T)."""
    return np.array([e - t0 for e in EVENTS if any(a <= e < b - 0.15 for a, b in intervals)])


def run(tmp_path, files, log=None, **kw):
    rec = tmp_path / "1-p1p2" / "recordings"
    rec.mkdir(parents=True)
    for spec in files:
        p = str(rec / spec["name"])
        if spec.get("empty"):
            open(p, "w").close()
        else:
            make_clip(p, spec["rec_start"], spec["length"], spec.get("size", (320, 240)),
                      spec.get("audio_length"), spec.get("size2"))
    if log is not None:
        write_log(str(tmp_path / "1-p1p2"), log)
    out = tmp_path / "out"
    rows = align.align_folder(str(rec) + "/", "1-p1p2/", str(out) + "/", verbose=False, crf=0, **kw)
    return rows, out / "trimed" / "1-p1p2"


def check(out_dir, row, t0, intervals, tol=2 / FPS_OUT, audio_tol=0.01):
    exp = expected(t0, intervals)
    f, dur_v = flashes(str(out_dir / row["output"]))
    b, dur_a = beeps(str(out_dir / row["output"]).replace(".mp4", ".wav"))
    assert len(f) == len(exp), (row["output"], f, exp)
    assert np.allclose(f, exp, atol=tol), (f, exp)
    assert len(b) == len(exp), (b, exp)
    assert np.allclose(b, exp, atol=audio_tol), (b, exp)
    assert abs(dur_v - row["duration_s"]) < 2 / FPS_OUT and abs(dur_a - row["duration_s"]) < 0.02


def test_join_times_do_not_shift_partners(tmp_path):
    """p1 joins 5 s and p2 2 s before the interaction starts; both recordings start at T.
    prepro's trim_folder would have cut p1 by 3 s; here both outputs start at T."""
    files = [dict(name=name(-5, "p1", 1, "dry"), rec_start=0, length=20),
             dict(name=name(-2, "p2", 1, "dry"), rec_start=0, length=20)]
    rows, out = run(tmp_path, files, log=[("p1", 0), ("p2", 0)])
    ok = [r for r in rows if r["status"] == "ok"]
    assert len(ok) == 2 and all(r["start_source"] == "log" for r in ok)
    for r in ok:
        check(out, r, 0, [(0, 20)])


def test_reconnection_is_joined_with_a_gap(tmp_path):
    """p2 drops at 8 s and is recording again from 11 s (rejoined at 9 s): one output with a 3 s gap."""
    files = [dict(name=name(-3, "p1", 1, "dry"), rec_start=0, length=20),
             dict(name=name(-1, "p2", 1, "dry"), rec_start=0, length=8),
             dict(name=name(9, "p2", 2, "dry"), rec_start=11, length=9)]
    rows, out = run(tmp_path, files, log=[("p1", 0), ("p2", 0), ("p2", 11)])
    r2 = next(r for r in rows if r.get("user") == "p2")
    assert r2["status"] == "ok" and r2["n_segments"] == 2 and r2["n_gaps"] == 1
    assert abs(r2["gaps_s"] - 3) < 0.05 and abs(r2["covered_s"] - 17) < 0.05 and abs(r2["duration_s"] - 20) < 0.05
    check(out, r2, 0, [(0, 8), (11, 20)])
    check(out, next(r for r in rows if r.get("user") == "p1"), 0, [(0, 20)])


def test_resolution_change_empty_file_and_fragment_before_start(tmp_path):
    """p1 has an empty file and a fragment recorded before p2 was there; p2 changes resolution mid-file."""
    files = [dict(name=name(-30, "p1", 1, "dry"), rec_start=-28, length=10),   # ends before t0: skipped
             dict(name=name(-15, "p1", 2, "dry"), empty=True),
             dict(name=name(-12, "p1", 3, "dry"), rec_start=-10, length=30),   # cut by 10 s
             dict(name=name(-2, "p2", 1, "dry"), rec_start=0, length=20, size=(320, 240), size2=(640, 480))]
    rows, out = run(tmp_path, files, log=[("p1", -28), ("p1", -13), ("p1", -10), ("p2", 0)])  # the empty file had a pipeline too
    status = sorted(r["status"] for r in rows)
    assert status.count("ok") == 2 and "skipped_empty_file" in status and "skipped_before_t0" in status
    r1 = next(r for r in rows if r.get("user") == "p1")
    r2 = next(r for r in rows if r.get("user") == "p2")
    assert (r2["width"], r2["height"]) == (640, 480)
    check(out, r1, 0, [(0, 20)])
    # p2's synthetic clip is two Opus encodes joined with -c copy: the join itself leaves ~20 ms of
    # encoder padding in the source, which the aligner keeps (it follows the audio timestamps)
    check(out, r2, 0, [(0, 20)], audio_tol=0.03)


def test_quick_reconnections_each_get_their_own_start(tmp_path):
    """p2 reconnects twice in a row; the third file is named (joined) before the second one's recording
    started, so a 'first event after the name' rule would give both files the same start."""
    files = [dict(name=name(-3, "p1", 1, "dry"), rec_start=0, length=20),
             dict(name=name(-1, "p2", 1, "dry"), rec_start=0, length=8),
             dict(name=name(9, "p2", 2, "dry"), rec_start=10, length=0.7),
             dict(name=name(10, "p2", 3, "dry"), rec_start=12.5, length=7.5)]
    rows, out = run(tmp_path, files, log=[("p1", 0), ("p2", 0), ("p2", 10), ("p2", 12.5)])
    r2 = next(r for r in rows if r.get("user") == "p2")
    seg = json.loads(r2["segments"])
    assert [round(x["t_begin"], 2) for x in seg] == [0, 10, 12.5] and all(x["cut_s"] == 0 for x in seg)
    check(out, r2, 0, [(0, 8), (10, 10.7), (12.5, 20)])


def test_missing_first_recording_gives_a_leading_gap(tmp_path):
    """p2's first file is empty and p2 records again from 6 s: p1 is NOT cut (t0 = interaction start
    from the log), p2's output starts with a 6 s gap."""
    files = [dict(name=name(-3, "p1", 1, "dry"), rec_start=0, length=20),
             dict(name=name(-1, "p2", 1, "dry"), empty=True),
             dict(name=name(4, "p2", 2, "dry"), rec_start=6, length=14)]
    rows, out = run(tmp_path, files, log=[("p1", 0), ("p2", 0), ("p2", 6)])
    r1 = next(r for r in rows if r.get("user") == "p1")
    r2 = next(r for r in rows if r.get("user") == "p2")
    assert r1["t0_source"] == "log" and abs(r1["duration_s"] - 20) < 0.05
    assert abs(r2["gaps_s"] - 6) < 0.05 and abs(r2["duration_s"] - 20) < 0.05
    check(out, r1, 0, [(0, 20)])
    check(out, r2, 0, [(6, 20)])


def test_audio_ending_early_is_reported_and_padded(tmp_path):
    files = [dict(name=name(-1, "p1", 1, "dry"), rec_start=0, length=20, audio_length=8),
             dict(name=name(-1, "p2", 1, "dry"), rec_start=0, length=20)]
    rows, out = run(tmp_path, files, log=[("p1", 0), ("p2", 0)])
    r1 = next(r for r in rows if r.get("user") == "p1")
    assert r1["status"] == "ok" and abs(r1["audio_covered_s"] - 8) < 0.1 and abs(r1["duration_s"] - 20) < 0.05
    f, _ = flashes(str(out / r1["output"]))
    assert np.allclose(f, expected(0, [(0, 20)]), atol=2 / FPS_OUT)
    b, dur_a = beeps(str(out / r1["output"]).replace(".mp4", ".wav"))
    assert np.allclose(b, expected(0, [(0, 8)]), atol=0.01) and abs(dur_a - 20) < 0.02


def test_without_log_falls_back_to_file_names(tmp_path):
    """No log: starts from the file names (legacy behaviour), flagged in the manifest."""
    files = [dict(name=name(0, "p1", 1, "dry"), rec_start=0, length=20),
             dict(name=name(2, "p2", 1, "dry"), rec_start=2, length=18)]
    rows, out = run(tmp_path, files, log=None)
    assert all(r["start_source"] == "file_name" for r in rows)
    for r in rows:
        check(out, r, 2, [(2, 20)])


def test_dry_and_wet_and_duration_cap(tmp_path):
    files = [dict(name=name(-1, u, 1, m), rec_start=0, length=20) for u in ("p1", "p2") for m in ("dry", "wet")]
    rows, out = run(tmp_path, files, log=[("p1", 0), ("p2", 0)], duration=12)
    assert len([r for r in rows if r["status"] == "ok"]) == 4
    for r in rows:
        assert abs(r["duration_s"] - 12) < 0.05
        check(out, r, 0, [(0, 12)])
