# prepro
This python module provides some scripts to simplify the data preprocessing of datasets collected with ducksoup

## Installation:
To install it please do the following:

First, create a new conda environment with requirements:
```
conda create --name ds_prepro python=3.9
conda activate ds_prepro
pip install soundfile pandas opencv-python
```

Second, install dependencies such as ffmpeg.
```
brew install ffmpeg
```

Third, clone the repository, as well as the one containing some dependencies:
```
git clone https://github.com/ducksouplab/prepro.git
git clone https://github.com/Pablo-Arias/STIM.git 
```

Add these to your path—see below if you have an error with these commands:
```
conda develop "$(pwd)/STIM"
conda develop "$(pwd)/prepro"
```

Sometimes the conda develop is not recognised by the system. In that case, just put these lines of code at the begining of your script:
```
import sys
from pathlib import Path

root = Path(__file__).resolve().parent   # directory that contains the script, where the repositories are located, change as required
sys.path.append(str(root / "STIM"))
sys.path.append(str(root / "prepro"))

# now you can do:
import stim          # whatever the package's top-level name is
import prepro
```


Now you put the raw data collected with ducksoup in a folder called "data/first_data_set and execute the process_video_example.py script:
```
python process_video_example.py

```




## Alignment and encoding (single pass, default since this version)
`ds_process` now aligns and encodes each interaction in **one** ffmpeg pass (`align.py`), directly from the original recordings:

- **Alignment on the real recording starts.** The time in a DuckSoup file name is when the participant *joined*; recording starts later, for everybody at the same moment (`pipeline_started` / `interaction_started` in the interaction logs, the `*.log` files next to `recordings/`). Each file is placed at its `pipeline_started` time, and time 0 is the start of the interaction. Without logs, the file names are used, as before, and the manifest says so.
- **Reconnections are kept.** When a participant reconnects, DuckSoup writes a new file (`-c-2-`, `-c-3-`, ...). All of a participant's files are placed on one timeline; the gaps are filled with black frames and silence.
- **One encode, native resolution.** Lossless x264 with `lossless=True` (the default; large files, e.g. ~200 MB per 4-min 800x600 video), or x264 at `crf_single_pass` with `lossless=False` (default 18, visually lossless and much smaller), constant `target_fps`, no rescaling (if the resolution changes during a recording, the largest size is used and the aspect ratio is kept). A lossless WAV is written next to each mp4 (`write_wav`).
- **Manifest.** `trimed/<interaction>/alignment.csv` lists, for each output, the segments used, the gaps, the covered time and the time with audio, so analyses can mask the gaps.

The previous chain (re-encode to 1280x720 -> change fps -> `trim_folder`: three lossy encodes) is still available with `single_pass=False`. Its `trim_folder` took the recording starts from the file names, which shifted the partners relative to each other by the difference of their join times (0-10 s), and cut everything recorded before a reconnection.

Tests (synthetic interactions, need ffmpeg with libx264 and libopus):
```
python -m pytest tests/test_align.py
```
