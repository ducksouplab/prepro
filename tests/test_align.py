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
# absolute times (s from T) of flashes and beeps; irregular, so that no lag but the true one lines them up
EVENTS = [1.0, 3.5, 7.2, 9.6, 12.3, 15.1, 18.8, 21.4, 22.9, 26.7, 28.2, 31.9, 33.3, 36.8, 39.1, 42.6, 44.0,
          47.3, 49.9, 51.2, 54.6, 57.1, 58.8, 61.5, 63.0]
NS = "mktest_session"
# another speaker's syllables (interferer), irregular and independent of EVENTS
OTHER_VOICE = [0.4, 2.3, 5.8, 6.6, 8.4, 11.1, 13.9, 14.6, 17.2, 20.1, 24.0, 25.5, 27.3, 30.3, 32.0, 34.9, 38.0,
               40.6, 41.7, 43.2]


def ts(t):
    return (T + datetime.timedelta(seconds=t)).strftime("%Y%m%d-%H%M%S.%f")[:-3]


def name(join_t, user, conn, manip, dyad="p1p2", rnd=1):
    return f"i-0123abcd-a-{ts(join_t)}-s-{NS}-n-{rnd}-{dyad}-u-{user}-c-{conn}-{manip}.mp4"


def make_clip(path, rec_start, length, size=(320, 240), audio_length=None, size2=None, audio_lead=0.0,
              audio_tail=0.0, video_skip=0.0, interferer=False):
    """Clip recorded from rec_start (s from T) for `length` s. Flash + beep at EVENTS.
    audio_length: the audio stops after this many s (of the video's time).
    audio_lead  : the audio track also holds the `audio_lead` s BEFORE the video started (DuckSoup's queued
                  audio), and both tracks start at timestamp 0: the audio is late by audio_lead in the file.
    audio_tail  : the audio track runs `audio_tail` s longer than the video at the end (not an offset).
    video_skip  : the video starts `video_skip` s late but is stamped from 0 (its content is early in the file),
                  as a wet video can be relative to its dry twin.
    interferer  : add another, LOUDER "voice" (beeps at other times, no flash) and a higher noise floor.
    size2: second half of the clip at another resolution (mid-stream size change)."""
    def part(out, off, dur, wh, lead=0.0, tail=0.0, skip=0.0):
        en_v = "+".join(f"between(t+{off + skip},{e},{e + 0.1})" for e in EVENTS)
        dur_v = dur - skip
        en_a = "+".join(f"between(t+{off - lead},{e},{e + 0.1})" for e in EVENTS)
        live = dur if audio_length is None else max(0.05, min(dur, audio_length - (off - rec_start)))
        ad = lead + live + tail
        other = ""
        if interferer:   # another speaker, 6 dB louder, at its own times (not locked to the target); noise -30 dB
            oth = "+".join(f"between(t+{off - lead},{e},{e + 0.25})" for e in OTHER_VOICE)
            other = f"+1.0*sin(2*PI*700*t)*({oth})+0.03*(2*random(1)-1)"
        subprocess.run(["ffmpeg", "-v", "error", "-y",
                        "-f", "lavfi", "-i", f"color=c=0x303030:s={wh[0]}x{wh[1]}:r={FPS_IN}:d={dur_v}",
                        # sample-accurate beeps (the `enable` timeline of audio filters works per audio frame),
                        # over a quiet noise floor (-55 dB) so that the audio is not digital silence
                        "-f", "lavfi", "-i",
                        f"aevalsrc=exprs='0.5*sin(2*PI*1000*t)*({en_a})+0.002*(2*random(0)-1){other}':s={SR}:d={ad}",
                        "-filter_complex",
                        f"[0:v]drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='{en_v}'[v];[1:a]anull[a]",
                        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-g", "25", "-pix_fmt", "yuv420p",
                        "-c:a", "libopus", out], check=True)
    if size2 is None:
        part(path, rec_start, length, size, audio_lead, audio_tail, video_skip)
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


def write_log(folder, pipeline_starts, pipeline_stops=(), audio_in=(), audio_fx=()):
    with open(os.path.join(folder, "1-p1p2-a-test.log"), "w") as fh:
        fh.write(json.dumps({"level": "info", "time": ts(0), "message": "interaction_started"}) + "\n")
        for user, t in pipeline_starts:
            fh.write(json.dumps({"level": "info", "user": user, "time": ts(t), "message": "pipeline_started"}) + "\n")
        for user, t in pipeline_stops:
            fh.write(json.dumps({"level": "info", "user": user, "time": ts(t), "message": "pipeline_stopped"}) + "\n")
        for user, t, fx in audio_fx:  # pipeline template (logged when the pipeline is created)
            fh.write(json.dumps({"level": "info", "user": user, "time": ts(t), "message": "template_data",
                                 "audioOptions": "{SkipFixedCaps:false DefaultBitrate:64000 Fx:" + fx +
                                                 " Muxer:oggmux Extension:ogg Decoder:opusdec}"}) + "\n")
        for user, t in audio_in:      # the user's audio track starts arriving at the server
            fh.write(json.dumps({"level": "debug", "user": user, "kind": "audio", "time": ts(t),
                                 "message": "in_track_received"}) + "\n")


def flashes(video):
    """Onset times (s) of white frames."""
    w, h = 8, 6
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"scale={w}:{h},format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    m = np.frombuffer(raw, np.uint8).reshape(-1, w * h).mean(1) > 128
    on = np.flatnonzero(m[1:] & ~m[:-1]) + 1
    return on / FPS_OUT, len(m) / FPS_OUT


def black_until(video):
    """Time (s) of the first frame that is not completely black (synthetic clips: dark gray background)."""
    w, h = 8, 6
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"scale={w}:{h},format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    m = np.frombuffer(raw, np.uint8).reshape(-1, w * h).mean(1) > 8
    return np.argmax(m) / FPS_OUT


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


def run(tmp_path, files, log=None, stops=(), audio_in=(), audio_fx=(), **kw):
    rec = tmp_path / "1-p1p2" / "recordings"
    rec.mkdir(parents=True)
    for spec in files:
        p = str(rec / spec["name"])
        if spec.get("empty"):
            open(p, "w").close()
        else:
            make_clip(p, spec["rec_start"], spec["length"], spec.get("size", (320, 240)),
                      spec.get("audio_length"), spec.get("size2"), spec.get("audio_lead", 0.0),
                      spec.get("audio_tail", 0.0), spec.get("video_skip", 0.0), spec.get("interferer", False))
    if log is not None:
        write_log(str(tmp_path / "1-p1p2"), log, stops, audio_in, audio_fx)
    out = tmp_path / "out"
    rows = align.align_folder(str(rec) + "/", "1-p1p2/", str(out) + "/", verbose=False, crf=0, **kw)
    return rows, out / "trimed" / "1-p1p2"


def check(out_dir, row, t0, intervals, tol=2 / FPS_OUT, audio_tol=0.01, video_intervals=None):
    """Flashes and beeps at their absolute times in the output; video_intervals: where there is picture
    (default: the same intervals as the sound)."""
    exp = expected(t0, intervals)
    exp_v = exp if video_intervals is None else expected(t0, video_intervals)
    f, dur_v = flashes(str(out_dir / row["output"]))
    b, dur_a = beeps(str(out_dir / row["output"]).replace(".mp4", ".wav"))
    assert len(f) == len(exp_v), (row["output"], f, exp_v)
    assert np.allclose(f, exp_v, atol=tol), (f, exp_v)
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


# ---------------------------------------------------------------------------------------------
# Audio-video sync within a recording (avsync.py)
# ---------------------------------------------------------------------------------------------
L = 45   # clips long enough for the content measurement (>= avsync.MIN_VALID_S)
LL = 66  # ... and for a STRONG content estimate (thirds of >= 20 s), needed to overrule the durations


def seg_info(row):
    return json.loads(row["segments"])


def test_audio_preroll_is_removed(tmp_path):
    """p1's file holds 2 s of audio recorded before its video started (both tracks from timestamp 0):
    the durations (audio vs pipeline_started -> pipeline_stopped) and the content agree on 2 s; the audio is
    moved 2 s earlier and the rendered output is in sync."""
    files = [dict(name=name(-8, "p1", 1, "dry"), rec_start=0, length=L, audio_lead=2.0),
             dict(name=name(-1, "p2", 1, "dry"), rec_start=0, length=L)]
    rows, out = run(tmp_path, files, log=[("p1", 0), ("p2", 0)], stops=[("p1", L), ("p2", L)],
                    audio_in=[("p1", -7.9), ("p2", -0.9)])
    r1 = next(r for r in rows if r.get("user") == "p1")
    r2 = next(r for r in rows if r.get("user") == "p2")
    s1 = seg_info(r1)[0]
    assert s1["duration_basis"] == "wall" and abs(s1["duration_offset_s"] - 2) < 0.05 and s1["wait_s"] == 7.9
    assert s1["content_reliable"] and abs(s1["content_lag_s"] - 2) < 0.05
    assert r1["offset_source"] == "duration+content" and abs(r1["audio_offset_s"] - 2) < 0.05
    assert r1["av_check"] == "ok" and abs(r1["av_check_lag_s"]) < 0.05
    assert r2["audio_offset_s"] == 0 and r2["av_check"] == "ok"
    for r in (r1, r2):
        check(out, r, 0, [(0, L)])


def test_audio_preroll_without_stop_event(tmp_path):
    """No `pipeline_stopped` in the log: the durations fall back to audio - video."""
    files = [dict(name=name(-8, "p1", 1, "dry"), rec_start=0, length=L, audio_lead=1.5)]
    rows, out = run(tmp_path, files, log=[("p1", 0)])
    r1 = rows[0]
    s1 = seg_info(r1)[0]
    assert s1["wall_s"] is None and s1["duration_basis"] == "video"
    assert r1["offset_source"] == "duration+content" and abs(r1["audio_offset_s"] - 1.5) < 0.05
    check(out, r1, 0, [(0, L)])


def test_extra_audio_at_the_end_is_not_an_offset(tmp_path):
    """The audio track is 2 s longer than the video, but at the END. The participant did not wait (their audio
    arrived 0.2 s before the start), so 2 s cannot be queued audio: the durations give no offset, the content
    agrees, and the audio is not moved."""
    files = [dict(name=name(-1, "p1", 1, "dry"), rec_start=0, length=LL, audio_tail=2.0)]
    rows, out = run(tmp_path, files, log=[("p1", 0)], stops=[("p1", LL)], audio_in=[("p1", -0.2)])
    r1 = rows[0]
    s1 = seg_info(r1)[0]
    assert s1["duration_offset_s"] == 0 and "longer than the wait" in s1["offset_note"]
    assert s1["content_reliable"] and abs(s1["content_lag_s"]) < 0.05
    assert r1["audio_offset_s"] == 0 and r1["offset_source"] == "content"
    f, _ = flashes(str(out / r1["output"]))
    b, _ = beeps(str(out / r1["output"]).replace(".mp4", ".wav"))
    assert np.allclose(b, expected(0, [(0, LL)]), atol=0.01) and np.allclose(f, expected(0, [(0, LL)]), atol=2 / FPS_OUT)


def test_extra_audio_at_the_end_without_wait_in_the_log(tmp_path):
    """Same, without `in_track_received` in the log: the durations suggest 2 s, the content shows that the
    audio is in sync, so it is not moved, and the manifest says why."""
    files = [dict(name=name(-1, "p1", 1, "dry"), rec_start=0, length=LL, audio_tail=2.0)]
    rows, out = run(tmp_path, files, log=[("p1", 0)], stops=[("p1", LL)])
    r1 = rows[0]
    s1 = seg_info(r1)[0]
    assert s1["wait_s"] is None and abs(s1["duration_offset_s"] - 2) < 0.05
    assert r1["audio_offset_s"] == 0 and r1["offset_source"] == "content" and "not at the start" in r1["notes"]
    b, _ = beeps(str(out / r1["output"]).replace(".mp4", ".wav"))
    assert np.allclose(b, expected(0, [(0, LL)]), atol=0.01)


def test_preroll_hidden_by_audio_ending_early(tmp_path):
    """2 s of pre-roll, but the audio also stops 1.5 s before the end: the durations see only 0.5 s;
    the content measures 2 s, which is applied (flagged), and the output is in sync."""
    files = [dict(name=name(-8, "p1", 1, "dry"), rec_start=0, length=LL, audio_lead=2.0, audio_length=LL - 1.5)]
    rows, out = run(tmp_path, files, log=[("p1", 0)], stops=[("p1", LL)], audio_in=[("p1", -7.9)])
    r1 = rows[0]
    s1 = seg_info(r1)[0]
    assert abs(s1["duration_offset_s"] - 0.5) < 0.05
    assert r1["offset_source"] == "content" and abs(r1["audio_offset_s"] - 2) < 0.04 and "disagrees" in r1["notes"]
    assert r1["av_check"] == "ok"
    f, _ = flashes(str(out / r1["output"]))
    b, _ = beeps(str(out / r1["output"]).replace(".mp4", ".wav"))
    assert np.allclose(f, expected(0, [(0, LL)]), atol=2 / FPS_OUT)
    assert np.allclose(b, expected(0, [(0, LL - 1.5)]), atol=0.04)    # content precision: < 1 analysis frame


def test_preroll_on_a_reconnection_segment(tmp_path):
    """p2 reconnects; only the second file has pre-roll (p2 waited before its pipeline restarted)."""
    files = [dict(name=name(-1, "p1", 1, "dry"), rec_start=0, length=80),
             dict(name=name(-1, "p2", 1, "dry"), rec_start=0, length=31),
             dict(name=name(32, "p2", 2, "dry"), rec_start=34, length=46, audio_lead=1.0)]
    rows, out = run(tmp_path, files, log=[("p1", 0), ("p2", 0), ("p2", 34)],
                    stops=[("p1", 80), ("p2", 31), ("p2", 80)],
                    audio_in=[("p1", -0.9), ("p2", -0.9), ("p2", 32.1)])   # p2 waited 1.9 s before restarting
    r2 = next(r for r in rows if r.get("user") == "p2")
    seg = seg_info(r2)
    assert [s["wall_s"] for s in seg] == [31, 46] and [s["wait_s"] for s in seg] == [0.9, 1.9]
    assert seg[0]["audio_offset_s"] == 0 and abs(seg[1]["audio_offset_s"] - 1) < 0.05
    assert seg[1]["offset_source"] == "duration+content"
    check(out, r2, 0, [(0, 31), (34, 80)])


def test_without_content_check_the_duration_offset_is_flagged(tmp_path):
    """av_sync=False: the durations alone decide, and the manifest flags the offset as unverified."""
    files = [dict(name=name(-8, "p1", 1, "dry"), rec_start=0, length=L, audio_lead=2.0)]
    rows, out = run(tmp_path, files, log=[("p1", 0)], stops=[("p1", L)], audio_in=[("p1", -7.9)], av_sync=False,
                    av_check=False)
    r1 = rows[0]
    assert r1["offset_source"] == "duration" and "unverified" in r1["notes"] and r1["av_check"] == "not_run"
    check(out, r1, 0, [(0, L)])


def test_recording_wait():
    t = lambda s: T + datetime.timedelta(seconds=s)
    starts, audio_in = {"p1": [t(0), t(30)]}, {"p1": [t(-5), t(-1), t(27)]}
    assert align.recording_wait("p1", t(0), starts, audio_in) == 1.0      # last arrival before the start
    assert align.recording_wait("p1", t(30), starts, audio_in) == 3.0     # this connection's arrival only
    assert align.recording_wait("p1", t(30), starts, {"p1": [t(-1)]}) is None
    assert align.recording_wait("p2", t(0), starts, audio_in) is None


def test_recording_stop_matching():
    t = lambda s: T + datetime.timedelta(seconds=s)
    starts, stops = {"p1": [t(0), t(30)]}, {"p1": [t(20), t(50)]}
    assert align.recording_stop("p1", t(0), starts, stops) == t(20)
    assert align.recording_stop("p1", t(30), starts, stops) == t(50)
    # the pipeline restarted before any stop: the stop belongs to the later recording
    assert align.recording_stop("p1", t(0), starts, {"p1": [t(50)]}) is None
    assert align.recording_stop("p1", t(30), starts, {}) is None


# ---- pure-logic tests of avsync (no ffmpeg) ----
import avsync  # noqa: E402


def test_duration_offset():
    D = avsync.duration_offset
    assert D(102.0, 100.0, 100.02) == (pytest.approx(1.98), "wall", "")
    assert D(102.0, 100.0, None) == (pytest.approx(2.0), "video", "")
    # the video may end early (it does not matter for the audio): the audio is still compared to the wall time
    assert D(100.13, 98.8, 100.0)[:2] == (pytest.approx(0.13), "wall")
    # a video LONGER than the wall time: the stop is not this recording's -> audio - video
    off, basis, note = D(300.0, 299.0, 70.0)
    assert (off, basis) == (pytest.approx(1.0), "video") and "stop event not used" in note
    assert D(99.0, 100.0, 100.0) == (0.0, "wall", "")                    # audio ending early: no offset
    assert D(100.05, 100.0, 100.0) == (0.0, "wall", "")                  # packet jitter
    assert D(None, 100.0, 100.0) == (0.0, None, "")
    assert D(102.0, None, 100.0) == (pytest.approx(2.0), "wall", "")     # audio-only recording
    # the wait bounds the queued audio
    assert D(102.0, 100.0, 100.0, wait_s=5.0)[0] == pytest.approx(2.0)
    off, _, note = D(101.0, 100.0, 100.0, wait_s=0.25)
    assert off == 0.0 and "longer than the wait" in note


R, S = dict(reliable=True), dict(reliable=True, strong=True)     # reliable / reliable and strong


@pytest.mark.parametrize("dur, content, off, src, note", [
    (2.0, dict(lag=2.05, **R), 2.0, "duration+content", ""),          # confirmed: the duration value is used
    (0.0, dict(lag=0.04, **R), 0.0, "content", ""),
    (1.0, dict(lag=0.05, **S), 0.0, "content", "not at the start"),   # strong content overrules durations
    (0.9, dict(lag=3.0, **S), 3.0, "content", "disagrees"),
    (0.0, dict(lag=1.2, **S), 1.2, "content", "not visible in the durations"),
    (0.0, dict(lag=-0.8, **S), -0.8, "content", "not visible in the durations"),
    (0.9, dict(lag=3.0, **R), 3.0, "content", "disagrees"),           # a reliable estimate decides
    (1.0, dict(lag=0.05, **R), 0.0, "content", "not at the start"),
    (2.0, dict(lag=0.3, reliable=False), 2.0, "duration", "inconclusive"),
    (2.0, dict(error="no_video"), 2.0, "duration", "no_video"),
    (2.0, None, 2.0, "duration", "not run"),
    (0.0, dict(lag=3.0, reliable=False), 0.0, "none", ""),
    (7.0, None, 0.0, "none", "not applied"),
])
def test_decide(dur, content, off, src, note):
    o, s, n = avsync.decide(dur, content)
    assert o == pytest.approx(off) and s == src and note in n


def _speechlike(n, rng):
    """Bursty envelope-like signal (talk spurts) at avsync.FPS."""
    on = np.repeat(rng.random(n // 15 + 1) < 0.5, 15)[:n].astype(float)
    return on * (1 + rng.random(n)) + 0.1 * rng.random(n)


@pytest.mark.parametrize("true_lag", [0.0, 0.5, 1.37, 3.0, -0.9])
def test_estimate_lag_recovers_the_lag(true_lag):
    rng = np.random.default_rng(1)
    n, k = 240 * avsync.FPS, int(round(true_lag * avsync.FPS))
    mouth = _speechlike(n + 400, rng)
    env = np.roll(mouth, k) + 0.8 * rng.random(n + 400)          # audio = mouth delayed by k frames + noise
    est = avsync.estimate_lag(mouth[200:200 + n], env[200:200 + n])
    assert est["reliable"] and est["strong"] and abs(est["lag"] - true_lag) < 0.5 / avsync.FPS + 1e-9


def test_estimate_lag_is_not_reliable_on_unrelated_signals():
    rng = np.random.default_rng(2)
    n = 240 * avsync.FPS
    est = avsync.estimate_lag(_speechlike(n, rng), _speechlike(n, rng))
    assert not est["reliable"] and not est["strong"]


def test_estimate_lag_needs_enough_signal():
    rng = np.random.default_rng(3)
    m = _speechlike(20 * avsync.FPS, rng)
    assert not avsync.estimate_lag(m, m)["reliable"]          # 20 s < MIN_VALID_S


def test_mouth_backend_selection(tmp_path, monkeypatch):
    """'auto' takes the best available backend; a backend that is not available falls back to whole-frame
    motion; the MediaPipe model is looked up in the argument, then PREPRO_FACE_MODEL."""
    monkeypatch.delenv("PREPRO_FACE_MODEL", raising=False)
    assert avsync.available_backend("frame") == "frame"
    assert avsync.face_model_path(str(tmp_path / "missing.task")) is None
    model = tmp_path / "m.task"
    model.write_bytes(b"x")
    assert avsync.face_model_path(str(model)) == str(model)
    monkeypatch.setenv("PREPRO_FACE_MODEL", str(model))
    assert avsync.face_model_path() == str(model)
    try:
        import mediapipe  # noqa: F401
        has_mp = True
    except ImportError:
        has_mp = False
    try:
        import cv2  # noqa: F401
        has_cv = True
    except ImportError:
        has_cv = False
    monkeypatch.delenv("PREPRO_FACE_MODEL")
    expected_auto = "opencv" if has_cv else "frame"          # no model file: never mediapipe
    assert avsync.available_backend("auto", face_model=str(tmp_path / "missing.task")) == expected_auto
    if not has_mp:
        assert avsync.available_backend("mediapipe", face_model=str(model)) == "frame"


def test_measure_on_a_clip_without_face_uses_whole_frame_motion(tmp_path):
    p = str(tmp_path / "clip.mp4")
    make_clip(p, 0, L, audio_lead=1.0)
    m = avsync.measure(p)
    assert m["roi"] == "frame" and m["reliable"] and abs(m["lag"] - 1.0) < 0.05
    short = str(tmp_path / "short.mp4")
    make_clip(short, 0, 10)
    assert avsync.measure(short)["error"] == "too_short"


# ---- dry / wet twins ----
def test_wet_file_is_aligned_on_its_dry_twin(tmp_path):
    """The wet file's video starts 1.3 s late (effect latency) and its muxer got less queued audio (0.8 s instead
    of 2 s). The log says there is no audio effect: the wet output takes the dry audio (with the dry offset) and
    its video is moved onto the dry clock; both outputs are in sync."""
    files = [dict(name=name(-8, "p1", 1, "dry"), rec_start=0, length=LL, audio_lead=2.0),
             dict(name=name(-8, "p1", 1, "wet"), rec_start=0, length=LL, audio_lead=0.8, video_skip=1.3)]
    rows, out = run(tmp_path, files, log=[("p1", 0)], stops=[("p1", LL)], audio_in=[("p1", -7.9)],
                    audio_fx=[("p1", -8, "")])
    dry = next(r for r in rows if r.get("manip") == "dry")
    wet = next(r for r in rows if r.get("manip") == "wet")
    sw = seg_info(wet)[0]
    assert abs(dry["audio_offset_s"] - 2) < 0.05
    assert sw["twin"]["same_audio"] and sw["twin"]["same_audio_basis"] == "log"
    assert sw["audio_source"] == dry["output"] and wet["offset_source"].startswith("dry twin")
    assert abs(wet["video_offset_s"] + 1.3) < 0.05 and abs(wet["audio_offset_s"] - dry["audio_offset_s"]) < 1e-9
    check(out, dry, 0, [(0, LL)])
    check(out, wet, 0, [(0, LL)], video_intervals=[(1.3, LL)])     # the wet picture starts 1.3 s late
    first = black_until(str(out / wet["output"]))
    assert abs(first - 1.3) <= 2 / FPS_OUT                        # ... and is black (not frozen) before


def test_wet_file_with_an_audio_effect_keeps_its_own_audio(tmp_path):
    """Same, but the log declares an audio effect: the wet audio is NOT replaced by the dry one; it is synced on
    its own (content) and moved together with its video."""
    files = [dict(name=name(-8, "p1", 1, "dry"), rec_start=0, length=LL, audio_lead=2.0),
             dict(name=name(-8, "p1", 1, "wet"), rec_start=0, length=LL, audio_lead=0.8, video_skip=1.3)]
    rows, out = run(tmp_path, files, log=[("p1", 0)], stops=[("p1", LL)], audio_in=[("p1", -7.9)],
                    audio_fx=[("p1", -8, "pitch semitones=2")])
    wet = next(r for r in rows if r.get("manip") == "wet")
    sw = seg_info(wet)[0]
    assert not sw["twin"]["same_audio"] and sw["audio_source"] == "own"
    assert abs(wet["video_offset_s"] + 1.3) < 0.05
    check(out, wet, 0, [(0, LL)], audio_tol=0.04, video_intervals=[(1.3, LL)])   # content precision


def test_sync_measure_is_robust_to_noise_and_a_louder_other_voice(tmp_path):
    """Another, louder 'voice' (beeps between the target's, without flashes) and a noise floor: the measured
    lag is still the target's (the visual side is the target only)."""
    p = str(tmp_path / "clip.mp4")
    make_clip(p, 0, L, audio_lead=1.0, interferer=True)
    m = avsync.measure(p)
    assert m["reliable"] and abs(m["lag"] - 1.0) < 0.05
