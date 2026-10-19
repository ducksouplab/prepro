# models
Optional: the MediaPipe face landmarker model, used by the audio-video sync measurement when `mediapipe` is installed (`avsync.py`):
```
curl -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```
Or set the environment variable `PREPRO_FACE_MODEL` to the model's path. Without the model, the sync measurement uses OpenCV (or whole-frame motion).
