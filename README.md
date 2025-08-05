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



