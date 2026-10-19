# prepro
Preprocessing of the recordings made with [DuckSoup](https://github.com/ducksouplab/ducksoup), in **one function call**. Afterwards:
- every participant's video and audio of an interaction are on **one clock** (time 0 = start of the interaction);
- **audio and video are in sync**, within each recording and between the participants;
- reconnections are joined, with gaps where nothing was recorded;
- everything is encoded once, without loss, at native resolution;
- a manifest (`alignment.csv`) documents every decision.

**Tutorial:** [`tutorial/preprocessing_tutorial.ipynb`](tutorial/preprocessing_tutorial.ipynb) runs it on a small synthetic experiment (no data needed), and shows the result and how to use it.

## Installation
```
conda create --name ds_prepro python=3.10
conda activate ds_prepro
pip install numpy pandas soundfile opencv-python
pip install mediapipe                  # optional, recommended: best audio-video sync measurement
brew install ffmpeg                    # on a cluster: tutorial/build_ffmpeg_locally_for_HPC.md

git clone https://github.com/ducksouplab/prepro.git
git clone https://github.com/Pablo-Arias/STIM.git
conda develop "$(pwd)/STIM"
conda develop "$(pwd)/prepro"
```
If `conda develop` does not work on your system, add the two folders to the path at the top of your script:
```python
import sys
sys.path.append("/path/to/STIM")
sys.path.append("/path/to/prepro")
```
With mediapipe, also download its face model, then either set `PREPRO_FACE_MODEL` to its path or keep it in `prepro/models/face_landmarker.task`:
```
curl -o prepro/models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Usage
The raw DuckSoup data of a session is one folder per interaction, holding `recordings/*.mp4` and the interaction's `*.log` files. One call processes them all:
```python
from ducksoup import ds_process_parallel

ds_process_parallel(sources="data/my_session/*/recordings/", target_folder="preproc/my_session/")
```

The output, in `target_folder`:
- `trimed/<interaction>/`: one mp4 and one WAV per participant and version (`dry`: the camera; `wet`: after the effect, as the partner saw it), plus `alignment.csv`;
- `combined/` and `with_audio/`: the participants of each interaction side by side, for checking by eye.

Options you may want:

| option | default | |
|---|---|---|
| `n_cores` | half of the machine's cores | CPU cores the run may use. On a cluster, the cores of the job: `n_cores=int(os.environ["SLURM_CPUS_PER_TASK"])` |
| `lossless` | `True` | lossless video, for analyses (~200 MB per 4-min 800x600 video). `False`: visually lossless and much smaller |
| `combine_videos` | `True` | also make the side-by-side check videos |
| `duration` | `None` | cut the outputs at this many seconds from the start |
| `av_sync` | `True` | measure and correct the audio-video sync. `False`: from the log durations only |
| `verbose` | `True` | print the progress (and the ffmpeg logs of the side-by-side videos) |

All other options: `help(ds_process)`. One interaction at a time: `ds_process(source_folder, folder_tag, target_folder)`.

**Time:** with mediapipe, about 3–5 min of CPU per 4-min recording (sync measurement, lossless encode, check). Run large datasets on a cluster, e.g. one Slurm task per session.

**After a run,** look in `alignment.csv` at:
- outputs whose `av_check` is `mismatch`;
- outputs whose `offset_source` is `duration` (unverified);
- the `notes` column;
- `status` values other than `ok`.

The rest of this README explains what is done and why.

## What the preprocessing does
`ds_process` (single pass, the default) calls `align.align_folder` for each interaction. It aligns and encodes each output (one per participant and version, dry or wet) in **one** ffmpeg pass from the original recordings:

1. **Alignment on the real recording starts.** The time in a DuckSoup file name is when the participant *joined*. Recording starts later, at `pipeline_started` in the interaction logs (the `*.log` files next to `recordings/`). Each file is placed at its `pipeline_started` time, and time 0 is `interaction_started`. Without logs, the file names are used, as before, and the manifest says so.
2. **Reconnections are kept.** When a participant reconnects, DuckSoup writes a new file (`-c-2-`, `-c-3-`, ...). All of a participant's files are placed on one timeline, and the gaps are filled with black frames and silence.
3. **Audio-video sync within each recording.** Audio queued while a participant waited is written at the start of the file, so the audio can be late by up to a few seconds. It is measured and removed (`avsync.py`); see below.
4. **Wet files on the dry clock.** The wet video, which went through the effect, can start later than the dry one. It is aligned on its dry twin, and it carries the dry audio when there is no audio effect; see below.
5. **One encode, native resolution.**
   - Lossless x264 with `lossless=True` (the default: large files, e.g. ~200 MB per 4-min 800x600 video), or `crf_single_pass` with `lossless=False` (default 18: visually lossless and much smaller).
   - Constant `target_fps` and no rescaling: if the resolution changes during a recording, the largest size is used and the aspect ratio is kept.
   - A lossless WAV is written next to each mp4 (`write_wav`).
6. **Manifest.** `trimed/<interaction>/alignment.csv` lists, for each output, the segments used, the gaps, the covered time, the time with audio, the audio and video offsets (with their evidence) and the sync check. Analyses can use it to mask the gaps and audit the sync.

## Audio-video sync within a recording
**The problem.** DuckSoup records each participant with GStreamer (`appsrc` → `rtpbin` → depayloader → `mp4mux`).
A participant's WebRTC tracks arrive as soon as they join, but their recording pipeline only starts when the interaction starts (`pipeline_started`).
When it starts, the audio that was queued while the participant waited goes into the file. The video starts only at the first key frame, because the H264 depayloader drops everything before one.
Both tracks are written from timestamp 0, so **the audio of such a file is late relative to its own video**, and relative to the partner, by a constant amount.
The amount grows with the wait. In the 2024 prolific data (604 files) it was:
- ~0 for waits under 0.5 s;
- 0.6–1.2 s for 2–4 s waits;
- 1.2–3 s for 8–15 s waits.

About 1 file in 8 is affected.

**How it is corrected.** The offset of each raw file is estimated from two independent sources:

1. **Durations (physical).** Nothing can be recorded after the pipeline stops, so audio beyond the wall-clock recording time (`pipeline_stopped` − `pipeline_started` in the logs) was recorded before the start.
   - Audio − video is the fallback when the log has no stop, or when the video is longer than the wall time, which means the stop belongs to another recording.
   - **Bound:** queued audio cannot be longer than the wait (`pipeline_started` − the audio `in_track_received`). A larger excess is reported, not applied.
   - This estimate is precise (~20 ms). It is blind to audio missing elsewhere: audio that ends early hides part of the offset.
2. **Content.** The speech-band audio envelope is cross-correlated with a mouth signal of the same file. Audio and video are decoded on their timestamps, as they are rendered. The result is precise to about one video frame and includes the natural lead of the mouth over the sound (0–0.15 s).
   Mouth signals, best first (`mouth_backend="auto"` uses the first one available):
   - `mediapipe`: jaw opening (`jawOpen`) from MediaPipe's face landmarker. It needs the `mediapipe` package and the model file, which you can give as `face_model=...` or as the `PREPRO_FACE_MODEL` environment variable, or put in `models/face_landmarker.task`. Download it with
     `curl -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`.
     About 100 s per 4-min recording.
   - `opencv`: motion of the lower half of the face found by OpenCV's face detector (`opencv-python`). About 10 s per recording, but weaker.
   - `frame`: whole-frame motion. Used when no face is found in at least half the frames.

   An estimate is **reliable** if its correlation peak is prominent (z ≥ 4), or moderately prominent (z ≥ 2.5) with at least two of the three thirds of the file giving the same lag (±0.1 s).
   It is **strong** if both hold. Files shorter than 30 s are not measured.

**Decision**, per file (`avsync.decide`):

| durations | content | applied offset | `offset_source` |
|---|---|---|---|
| offset X | reliable, ≈ X (±0.25 s) | X (more precise) | `duration+content` |
| none | reliable, ≈ 0 | 0 | `content` |
| offset X | reliable, ≈ 0 | 0 (the extra audio is not at the start) | `content` + note |
| any | reliable, Y ≠ X | Y | `content` + note |
| offset X | inconclusive / not measurable | X, flagged *unverified* | `duration` |
| none | inconclusive | 0 | `none` |

**Validation** (2024 prolific data, MediaPipe backend): 10 interactions, 40 outputs, chosen for their problems (pre-roll up to 2.9 s, reconnections, misleading durations).
- Every reliable content estimate agreed with the same measurement on the rendered output.
- The rendered outputs had 0 mismatches: 24 were verified in sync (residual ≤ 0.25 s, which is the natural mouth lead), and 16 were inconclusive (too little visible talking).
- The durations alone would have been wrong for 6 of the 40 files. The wait bound caught 4 of them (extra audio that was not at the start), and the content corrected the other 2 (pre-roll partly hidden by audio ending early).

Offsets larger than `max_offset` (5 s) are reported and not applied. The audio is then moved by the offset in the same single encode, with sample accuracy: dropped at the start, or delayed with silence if negative. With `av_check=True`, each rendered mp4 is measured again.


**Robustness to noise and other voices.** The mouth signal is the recorded participant's own jaw, so other people's voices only enter through the audio. They are not correlated with that jaw at any consistent lag; a conversational partner is, if anything, *anti*-correlated through turn-taking.
On 4 real recordings with known offsets (0.05–2.9 s), every measurement was correct (within 0.15 s) in each of these conditions:
- white noise at +10, 0 and −5 dB SNR;
- an unrelated speaker at equal loudness;
- the conversational partner's own voice, at equal loudness and at +6 dB (louder than the participant), mixed in at the true time as speaker leakage would be.

A synthetic test (a louder interfering voice plus noise) is part of the test suite.

## Wet files: aligned on their dry twin
The dry and wet files of a recording come from the same camera and microphone, but DuckSoup writes them through separate muxers, and the wet video goes through the effect first. So:
- the wet video can start later than the dry one while being stamped from 0. In the 2024 prolific data most wet files are frame-exact, but some are off by up to 1.3 s, and a few drift by 2–6 frames within the file;
- the wet muxer can receive a different amount of queued audio than the dry one, and it can drop packets during the recording (up to ~1 s).

**Correction** (`align.twin_offsets`, `avsync.twin_lags`; `twin_sync=True` by default):
- **Video.** The wet video is moved onto the dry clock by its lag relative to the dry video. The lag is measured on the image content: tiny thumbnails projected on their main components (head pose, lighting), which both versions share. It is frame-exact.
  - It is reliable when the peak is prominent and the thirds of the file agree.
  - Drift within the file is reported (`notes`).
  - When the video is moved later, the time before it is black, not a frozen frame.
- **Audio.** When the wet pipeline had no audio effect (`template_data` → `audioOptions … Fx:` empty in the log), the wet audio is the dry microphone signal. The wet output then carries the **dry twin's audio**, with the dry file's verified offset: identical sound, immune to the wet muxer's dropped packets.
  - With an audio effect, the wet audio is kept, synced on its own and moved together with its video.
  - Without the log entry, the dry audio is used only if the waveforms are clearly identical.

**Validation:** on 6 real dry/wet pairs, the image-content lags matched the lags measured independently from eye blinks (MediaPipe) to within one frame: −40.8 vs −41, −38.5 vs −38, −3.4 vs −4, the step 0 → −4.9 vs 0 → −5, and 0 vs 0 twice.
A test checks that a pitch-shifted wet voice is not replaced.

## Manifest columns (`alignment.csv`)
Per output:

| column | meaning |
|---|---|
| `folder_tag`, `user`, `manip`, `output`, `status` | interaction, participant, dry/wet, output file, `ok` / `exists` / `ffmpeg_error: …` / `skipped_<reason>` |
| `t0`, `t0_source` | time 0 (`log`: `interaction_started`; `first_recordings` without a log) |
| `start_source` | `log` (`pipeline_started`) or `file_name` (legacy fallback) |
| `n_segments`, `duration_s`, `covered_s`, `audio_covered_s` | segments used, output length, time with video, time with audio |
| `n_gaps`, `gaps_s` | gaps (black frames and silence) |
| `width`, `height` | output size |
| `audio_offset_s`, `video_offset_s` | the largest offsets applied over the output's segments (> 0: the content was late in its file and was moved earlier) |
| `offset_source` | how the audio offset was decided: `duration+content`, `content`, `duration` (unverified), `none`, `dry twin: …` |
| `av_check`, `av_check_lag_s`, `av_check_z` | the audio-video lag measured again on the rendered mp4: `ok`, `mismatch`, `inconclusive`, `too_short`, `not_run` |
| `notes` | what is worth a look: unverified offsets, disagreements, drift of a wet video, offsets not applied |
| `segments` | JSON, one entry per segment (below) |

Per segment (`segments`):
- `file`, `connection`, `t_begin`, `t_end`, `cut_s`;
- `wall_s`, `wait_s`, `video_s`, `audio_s`;
- `duration_offset_s`, `duration_basis`;
- `content_lag_s`, `content_z`, `content_reliable`, `content_strong`, `content_roi` (the mouth signal used), `content_error`;
- `audio_offset_s`, `offset_source`, `offset_note`;
- `audio_source` (`own` or the dry twin's file), `video_offset_s`;
- `twin`: the wet-vs-dry measurement (`video_lag`, `video_z`, `video_thirds`, `video_drift`, `audio_lag`, `audio_z`, `waveform_r`, `audio_fx`, `same_audio`, `same_audio_basis`).

**After a run, look at:**
- outputs with `av_check == "mismatch"`;
- outputs whose `offset_source` is `duration` (unverified);
- the `notes` column;
- `status` values other than `ok` / `exists`.

`ds_process(..., av_sync=False)` uses the durations only (flagged as unverified) and skips the output check. `align_folder` also takes `mouth_backend`, `face_model`, `twin_sync` and `max_offset` (see the tutorial for every parameter).

## Legacy chain
The previous chain (re-encode to 1280x720 → change fps → `trim_folder`: three lossy encodes) is still available with `single_pass=False`. Its `trim_folder` took the recording starts from the file names, which shifted the partners relative to each other by the difference of their join times (0–10 s). It also cut everything recorded before a reconnection, and it did not correct the audio offsets.

## Tests
Synthetic interactions; they need ffmpeg with libx264 and libopus, numpy, soundfile and pytest:
```
python -m pytest tests/test_align.py
```
They cover the following cases:
- join times;
- reconnections and quick reconnections;
- resolution changes, empty files and fragments before the start;
- audio ending early;
- missing logs, dry/wet and duration caps;
- audio pre-roll: with and without a `pipeline_stopped` event, on a reconnection segment, and hidden by audio that ends early;
- the waiting-time bound: extra audio at the end is not shifted;
- `av_sync=False`;
- noise and a louder interfering voice;
- wet files on their dry twin: video offset, black padding, dry audio, and own audio with an audio effect;
- the decision table, the lag estimator on synthetic signals, log matching (`recording_stop`, `recording_wait`), effect parsing, the choice of mouth backend, and whole-frame motion on faceless clips.
