# prepro
This python module provides some scripts to simplify the data preprocessing of datasets collected with ducksoup

## Installation:
To install it please do the following:

First, create a new conda environment with requirements:
```
conda create --name ds_prepro python=3.9
conda activate ds_prepro
pip install scipy soundfile pyloudnorm pandas pyo numpy opencv-python
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

Add these to your path:
```
conda develop "$(pwd)/STIM"
conda develop "$(pwd)/prepro"
```

Now you put the raw data collected with ducksoup in a folder called "data/first_data_set and execute the process_video_example.py script:
```
python 

```



