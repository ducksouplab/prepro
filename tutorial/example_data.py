"""
A tiny made-up DuckSoup session, only so that the tutorial has something to run on.
With your own data you do not need this.

It has the same layout and the same kind of files as real DuckSoup data:

    <root>/data/my_session/1-p1p2/recordings/*.mp4   (participants p1 and p2, dry and wet)
    <root>/data/my_session/1-p1p2/*.log              (the DuckSoup log of the interaction)

and the usual problems: p1's audio is late by 2 s (audio queued while p1 waited), p1's wet video is 1 s off,
and p2 reconnects (two files, with a 4 s gap). Each recording shows a white flash with a beep at known times.
"""

import datetime
import json
import subprocess
from pathlib import Path

T = datetime.datetime(2024, 1, 1, 12, 0, 0)          # start of the interaction
EVENTS = [1.0, 3.5, 7.2, 9.6, 12.3, 15.1, 18.8, 21.4, 22.9, 26.7, 28.2, 31.9, 33.3, 36.8, 39.1, 42.6, 44.0,
          47.3, 49.9, 51.2, 54.6, 57.1, 58.8, 61.5, 63.0, 66.4, 68.1]     # flash + beep (s from T)
LENGTH = 70


def _ts(t):
    return (T + datetime.timedelta(seconds=t)).strftime("%Y%m%d-%H%M%S.%f")[:-3]


def _name(join_t, user, conn, manip):
    return f"i-0123abcd-a-{_ts(join_t)}-s-tutorial-n-1-p1p2-u-{user}-c-{conn}-{manip}.mp4"


def _clip(path, rec_start, length, audio_lead=0.0, video_skip=0.0):
    en_v = "+".join(f"between(t+{rec_start + video_skip},{e},{e + 0.1})" for e in EVENTS)
    en_a = "+".join(f"between(t+{rec_start - audio_lead},{e},{e + 0.1})" for e in EVENTS)
    subprocess.run(["ffmpeg", "-threads", "1", "-v", "error", "-y",
                    "-f", "lavfi", "-i", f"color=c=0x303030:s=320x240:r=25:d={length - video_skip}",
                    "-f", "lavfi", "-i",
                    f"aevalsrc=exprs='0.5*sin(2*PI*1000*t)*({en_a})+0.002*(2*random(0)-1)':s=48000:d={audio_lead + length}",
                    "-filter_complex", f"[0:v]drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='{en_v}'[v];[1:a]anull[a]",
                    "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-g", "25", "-pix_fmt", "yuv420p",
                    "-c:a", "libopus", str(path)], check=True)


def make_example_session(root="example"):
    """Write the example session under <root>/data/my_session/ and return that folder."""
    session = Path(root) / "data" / "my_session"
    rec = session / "1-p1p2" / "recordings"
    rec.mkdir(parents=True, exist_ok=True)
    _clip(rec / _name(-8, "p1", 1, "dry"), 0, LENGTH, audio_lead=2.0)
    _clip(rec / _name(-8, "p1", 1, "wet"), 0, LENGTH, audio_lead=0.7, video_skip=1.0)
    for m in ("dry", "wet"):
        _clip(rec / _name(-1, "p2", 1, m), 0, 30)
        _clip(rec / _name(32, "p2", 2, m), 34, LENGTH - 34)
    events = [dict(time=_ts(0), message="interaction_started")]
    for user, t in [("p1", -7.9), ("p2", -0.9), ("p2", 32.2)]:
        events.append(dict(user=user, kind="audio", time=_ts(t), message="in_track_received"))
    for user, t in [("p1", -8), ("p2", -1), ("p2", 32)]:
        events.append(dict(user=user, time=_ts(t), message="template_data",
                           audioOptions="{DefaultBitrate:64000 Fx: Muxer:oggmux Extension:ogg}"))
    for user, t in [("p1", 0), ("p2", 0), ("p2", 34)]:
        events.append(dict(user=user, time=_ts(t), message="pipeline_started"))
    for user, t in [("p1", LENGTH), ("p2", 30), ("p2", LENGTH)]:
        events.append(dict(user=user, time=_ts(t), message="pipeline_stopped"))
    with open(rec.parent / "1-p1p2-a-tutorial.log", "w") as fh:
        for e in sorted(events, key=lambda e: e["time"]):
            fh.write(json.dumps(e) + "\n")
    return session
